# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from .cp_utils import slice_log_prob_with_cp


def sampling_replay_response_indices(
    *,
    sample_index: int,
    total_lengths: list[int],
    response_lengths: list[int],
    local_seq_length: int,
    cp_size: int,
    cp_rank: int,
    allgather_cp: bool,
    cp_output_is_zigzag: bool,
    qkv_format: str,
    max_seq_len: int | None,
    padded_total_length: int | None,
) -> torch.Tensor:
    """CPU CSR row indices in exactly the order scored by get_responses."""
    total_length, response_length = total_lengths[sample_index], response_lengths[sample_index]
    indices = torch.arange(response_length, device="cpu")
    if cp_size == 1:
        return indices
    if allgather_cp and not cp_output_is_zigzag:
        # Contiguous CP shards are in predictor coordinates (target minus one).
        response_start = sum(total_lengths[:sample_index]) + total_length - response_length - 1
        start = max(response_start, cp_rank * local_seq_length)
        end = min(response_start + response_length, (cp_rank + 1) * local_seq_length)
        return indices[max(0, start - response_start) : max(0, end - response_start)]
    return slice_log_prob_with_cp(
        indices,
        total_length,
        response_length,
        qkv_format,
        max_seq_len,
        padded_total_length,
        dynamic_cp_size=cp_size,
        dynamic_cp_rank=cp_rank,
    )


def sampling_replay_keep_mask(
    ids: torch.Tensor,
    offsets: torch.Tensor,
    response_indices: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_start: int,
    local_vocab_size: int,
    vocab_size: int,
) -> torch.Tensor:
    """Expand only selected CSR rows and this TP rank's vocabulary shard.

    CSR and row selection stay on CPU. Only sparse local coordinates and the
    temporary bool mask move to the logits device. Validate targets BEFORE the
    logprob primitive, which otherwise implicitly retains absent targets.
    """
    lengths = offsets[response_indices + 1] - offsets[response_indices]
    rows = torch.repeat_interleave(torch.arange(len(response_indices), device="cpu"), lengths)
    starts = offsets[response_indices] - lengths.cumsum(0) + lengths
    positions = torch.arange(rows.numel(), device="cpu") + torch.repeat_interleave(starts, lengths)
    selected_ids = ids[positions].long()
    local = (selected_ids >= vocab_start) & (selected_ids < vocab_start + local_vocab_size)
    keep = torch.zeros((len(response_indices), local_vocab_size), dtype=torch.bool, device=targets.device)
    keep[rows[local].to(targets.device), (selected_ids[local] - vocab_start).to(targets.device)] = True
    local_targets = targets - vocab_start
    target_on_shard = (local_targets >= 0) & (local_targets < local_vocab_size)
    kept = keep.gather(1, local_targets.clamp(0, local_vocab_size - 1).unsqueeze(1)).squeeze(1)
    # Async device assertion avoids a GPU -> CPU synchronization on the hot path.
    torch._assert_async(
        ((targets >= 0) & (targets < vocab_size) & (~target_on_shard | kept)).all(),
        "sampled token is absent from rollout sampling support",
    )
    return keep
