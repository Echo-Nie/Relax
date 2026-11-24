# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tiny NCCL tests of the real CSR -> Actor scoring path (no model
training)."""

from argparse import Namespace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


TOKENS = [[6, 0, 3, 7, 2, 5], [4, 1, 6], [1, 0]]
SUPPORTS = [[[3, 1], [7, 0, 4], [2], [5, 1, 6]], [[6, 2, 0]], [[0, 5]]]
PADDED = [12, 4, 8]


def _worker(rank, tp, cp, layout, agentic, rendezvous):
    from megatron.core import mpu

    from relax.backends.megatron.loss import get_log_probs_and_entropy
    from relax.utils.sampling_mask import RolloutSamplingMask

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=tp * cp, timeout=timedelta(seconds=90)
    )
    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        cp_rank = mpu.get_context_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        device = torch.device("cuda", rank)
        full = torch.arange(11 * 8, device=device).reshape(11, 8).float().sin()
        reference = full.detach().clone().requires_grad_()
        lengths = [len(t) for t in TOKENS]
        supports = SUPPORTS
        masks = [RolloutSamplingMask.from_rows(rows) for rows in supports]
        if agentic:
            from types import SimpleNamespace

            from relax.agentic.session.state import SessionForest

            forest = SessionForest.create_empty(session_id="gpu", use_rollout_sampling_replay=True)
            parent = forest.append_obs(
                parent_state_hash=forest.root_state_hash,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "user", "content": "p"}],
                train_token_delta=[6, 0],
                rollout_token_delta=[6, 0],
            )
            first = forest.append_resp(
                parent_state_hash=parent.state_hash,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "assistant", "content": "a"}],
                token_delta=[3],
                logprob_delta=[-0.5],
                sampling_mask_delta=RolloutSamplingMask.from_rows([[3, 1]]),
            )
            observation = forest.append_obs(
                parent_state_hash=first.state_hash,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "tool", "tool_call_id": "call", "content": "r"}],
                train_token_delta=[7, 2],
                rollout_token_delta=[7, 2],
            )
            leaf = forest.append_resp(
                parent_state_hash=observation.state_hash,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "assistant", "content": "b"}],
                token_delta=[5],
                logprob_delta=[-1.0],
                sampling_mask_delta=RolloutSamplingMask.from_rows([[5, 1, 6]]),
            )
            sample = forest.build_sample(
                leaf_state_hash=leaf.state_hash, tokenizer=SimpleNamespace(decode=lambda tokens, **_: str(tokens))
            )
            assert sample.tokens == TOKENS[0]
            assert sample.loss_mask == [1, 0, 0, 1]
            assert sample.rollout_log_probs == [-0.5, 0.0, 0.0, -1.0]
            masks[0] = sample.rollout_sampling_mask
            # Independent hand-written expectation for the exported rows.
            supports = [[[3, 1], [7], [2], [5, 1, 6]], *SUPPORTS[1:]]
        response_lengths = [len(rows) for rows in supports]
        expected = []
        # Independent unsharded torch reference, using the saved ragged rows.
        start = 0
        predictors = []
        for tokens, rows in zip(TOKENS, supports, strict=True):
            prompt = len(tokens) - len(rows)
            predictors.append([start + prompt - 1 + t for t in range(len(rows))])
            expected.append(
                torch.stack(
                    [
                        (reference[start + prompt - 1 + t, support] / 0.7).log_softmax(0)[
                            support.index(tokens[prompt + t])
                        ]
                        for t, support in enumerate(rows)
                    ]
                )
            )
            start += len(tokens)
        sum(row.sum() for row in expected).backward()

        if cp == 1:
            positions = list(range(11))
            owned_rows = [list(range(n)) for n in response_lengths]
        elif layout == "allgather":
            positions = list(range(cp_rank * 6, (cp_rank + 1) * 6))
            # Final outputs are redistributed to per-sample zigzag ownership.
            owned_rows = []
            for predictor_rows, padded, seq_start in zip(predictors, PADDED, [0, 6, 9], strict=True):
                chunk = padded // 4
                local_positions = [
                    *range(cp_rank * chunk, (cp_rank + 1) * chunk),
                    *range((3 - cp_rank) * chunk, (4 - cp_rank) * chunk),
                ]
                owned_rows.append([i for i, p in enumerate(predictor_rows) if p - seq_start in local_positions])
        else:
            positions, owned_rows = [], []
            for predictor_rows, padded, seq_start, length in zip(predictors, PADDED, [0, 6, 9], lengths, strict=True):
                chunk = padded // 4
                local_positions = [
                    *range(cp_rank * chunk, (cp_rank + 1) * chunk),
                    *range((3 - cp_rank) * chunk, (4 - cp_rank) * chunk),
                ]
                positions.extend([seq_start + p if p < length else -1 for p in local_positions])
                owned_rows.append([i for i, p in enumerate(predictor_rows) if p - seq_start in local_positions])
        # Fill padding with conspicuously different logits so accidental scoring fails.
        positions = [p if p < 11 else -1 for p in positions]
        local_full = torch.stack([full[p] if p >= 0 else torch.full((8,), 19.0, device=device) for p in positions])
        lo, hi = tp_rank * (8 // tp), (tp_rank + 1) * (8 // tp)
        logits = local_full[:, lo:hi].clone().unsqueeze(0).requires_grad_()
        args = Namespace(
            qkv_format="thd",
            allgather_cp=layout in {"allgather", "already_zigzag"},
            rollout_temperature=0.7,
            log_probs_chunk_size=2,
            entropy_coef=0.0,
            advantage_estimator="grpo",
            loss_type="policy_loss",
        )
        unconcat_tokens = [torch.tensor(t, device=device) for t in TOKENS]
        mask_ids, mask_offsets = [m.ids for m in masks], [m.offsets for m in masks]
        if layout == "get_batch":
            from megatron.training.global_vars import set_args

            from relax.backends.megatron.data import DataIterator, get_batch

            set_args(args)
            source = {
                "tokens": unconcat_tokens,
                "total_lengths": lengths,
                "response_lengths": response_lengths,
                "loss_masks": [torch.ones(n, device=device) for n in response_lengths],
                "advantages": [torch.ones(n) for n in response_lengths],
                "rollout_sampling_mask_ids": mask_ids,
                "rollout_sampling_mask_offsets": mask_offsets,
            }
            packed = get_batch(DataIterator(source, micro_batch_size=3), list(source), pad_multiplier=1)
            assert packed["tokens"].device == device
            assert all(t.device == device for t in packed["advantages"])
            for field, original in (
                ("rollout_sampling_mask_ids", mask_ids),
                ("rollout_sampling_mask_offsets", mask_offsets),
            ):
                for actual, expected_csr in zip(packed[field], original, strict=True):
                    assert actual.device.type == "cpu"
                    assert actual.dtype == expected_csr.dtype
                    assert torch.equal(actual, expected_csr)
            unconcat_tokens = packed["unconcat_tokens"]
            lengths, response_lengths = packed["total_lengths"], packed["response_lengths"]
            mask_ids, mask_offsets = packed["rollout_sampling_mask_ids"], packed["rollout_sampling_mask_offsets"]
            pad = packed["tokens"].size(1) - logits.size(1)
            positions.extend([-1] * pad)
            logits = torch.cat([logits.detach(), logits.new_zeros(1, pad, hi - lo)], dim=1).requires_grad_()
        _, result = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=lengths,
            response_lengths=response_lengths,
            padded_total_lengths=PADDED if cp > 1 else None,
            cp_output_is_zigzag=layout == "already_zigzag",
            use_rollout_sampling_replay=True,
            rollout_sampling_mask_ids=mask_ids,
            rollout_sampling_mask_offsets=mask_offsets,
            with_entropy=True,
        )
        for actual, all_expected, indices in zip(result["log_probs"], expected, owned_rows, strict=True):
            torch.testing.assert_close(actual, all_expected[indices], atol=2e-6, rtol=2e-6)
        # Entropy remains full-vocabulary, despite support-normalized logprobs.
        for actual, predictor_rows, indices in zip(result["entropy"], predictors, owned_rows, strict=True):
            selected = [predictor_rows[i] for i in indices]
            lp = (full[selected] / 0.7).log_softmax(-1)
            torch.testing.assert_close(actual, -(lp.exp() * lp).sum(-1), atol=2e-6, rtol=2e-6)
        sum(row.sum() for row in result["log_probs"]).backward()
        expected_grad = torch.stack(
            [reference.grad[p, lo:hi] if p >= 0 else torch.zeros(hi - lo, device=device) for p in positions]
        ).unsqueeze(0)
        torch.testing.assert_close(logits.grad, expected_grad, atol=2e-6, rtol=2e-6)
        assert all(m.ids.device.type == "cpu" and m.offsets.device.type == "cpu" for m in masks)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp,cp,layout,agentic",
    [
        (2, 1, "get_batch", False),
        (2, 1, "zigzag", False),
        (1, 2, "zigzag", False),
        (1, 2, "allgather", False),
        (1, 2, "already_zigzag", False),
        (2, 2, "allgather", False),
        (2, 1, "get_batch", True),
        (1, 2, "zigzag", True),
        (2, 2, "allgather", True),
    ],
)
def test_sampling_replay_nccl_logprob_gradient_and_cp_alignment(tp, cp, layout, agentic, tmp_path):
    if torch.cuda.device_count() < tp * cp:
        pytest.skip(f"requires {tp * cp} GPUs for TP={tp}, CP={cp}")
    mp.spawn(_worker, args=(tp, cp, layout, agentic, str(tmp_path / "rendezvous")), nprocs=tp * cp, join=True)


def _absent_target_worker(rank):
    from relax.backends.megatron.sampling_replay import sampling_replay_keep_mask

    torch.cuda.set_device(rank)
    target = torch.tensor([3], device=f"cuda:{rank}")
    try:
        sampling_replay_keep_mask(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, 1]),
            torch.tensor([0]),
            target,
            vocab_start=0,
            local_vocab_size=8,
            vocab_size=8,
        )
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert "device-side assert" in str(error), str(error)
    else:
        raise AssertionError("GPU accepted a sampled token absent from saved support")


def test_sampling_replay_gpu_rejects_absent_target_in_isolated_context():
    if torch.cuda.device_count() < 1:
        pytest.skip("requires one GPU for the isolated device-assert test")
    # Device assertions poison their CUDA context. A fresh process contains that
    # expected failure and leaves the parent's and other tests' contexts intact.
    mp.spawn(_absent_target_worker, nprocs=1, join=True)
