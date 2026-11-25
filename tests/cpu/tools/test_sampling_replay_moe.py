# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from scripts.tools.sampling_replay_isolated import start_ticks
from scripts.tools.sampling_replay_moe import guard_cluster, model_inventory, validate_samples


def test_inventory_reads_real_bf16_storage_without_loading_model(tmp_path):
    save_file(
        {"weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )
    inventory = model_inventory(tmp_path)
    assert inventory["parameters"] == 6
    assert inventory["bf16_payload_bytes"] == 12
    assert set(inventory["shards"]) == {"model.safetensors"}


def test_submission_rejects_host_namespace_before_creating_run(tmp_path):
    instance = tmp_path / "ray"
    instance.mkdir()
    (instance / "state.json").write_text(
        json.dumps(dict(pid=os.getpid(), netns=os.readlink("/proc/self/ns/net"), start_ticks=start_ticks(os.getpid())))
    )
    with pytest.raises(ValueError, match="private network namespace"):
        guard_cluster(tmp_path, instance, "off")
    assert not (tmp_path / "smoke-off").exists()


def sample_pair():
    originals = [
        dict(
            index=i,
            tokens=[9, 3],
            loss_mask=[1],
            response_length=1,
            rollout_log_probs=[0.0],
            rollout_sampling_mask=dict(ids=[3], offsets=[0, 1]),
        )
        for i in range(32)
    ]
    samples = [
        dict(
            sample_indices=i,
            tokens=[9, 3],
            loss_masks=[1],
            response_lengths=1,
            rollout_log_probs=[0.0],
            advantages=[1.0],
            rollout_sampling_mask_ids=[3],
            rollout_sampling_mask_offsets=[0, 1],
        )
        for i in range(32)
    ]
    return samples, originals


def test_sampling_metadata_survives_transfer_and_detects_corruption():
    samples, originals = sample_pair()
    assert validate_samples(samples, originals, True) == 32
    corrupt = copy.deepcopy(samples)
    corrupt[0]["rollout_sampling_mask_ids"] = [2]
    with pytest.raises(ValueError, match="CSR changed"):
        validate_samples(corrupt, originals, True)
    corrupt = copy.deepcopy(samples)
    corrupt[0]["loss_masks"] = []
    with pytest.raises(ValueError, match="alignment"):
        validate_samples(corrupt, originals, True)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_samples(samples[:-1], originals, True)
    with pytest.raises(ValueError, match="OFF unexpectedly"):
        validate_samples(samples, originals, False)


def test_singleton_behavior_is_checked_after_both_sides_match():
    samples, originals = sample_pair()
    samples[0]["rollout_log_probs"] = [-0.5]
    originals[0]["rollout_log_probs"] = [-0.5]
    with pytest.raises(ValueError, match="Singleton"):
        validate_samples(samples, originals, True)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_static_cp_probability_transfer_uses_response_prediction_positions(cp_rank):
    # T=12,P=2,R=10: logits 1..10 intersect CP0 chunks [0,3),[9,12)
    # at response indices [0,1,8,9]; CP1 gets [2,3,4,5,6,7].
    # T=9,P=5,R=4: logits 4..7 lie entirely on CP1, so CP0 is empty.
    selectors = (([0, 1, 8, 9], []), ([2, 3, 4, 5, 6, 7], [0, 1, 2, 3]))
    samples, originals = [], []
    for i in range(32):
        prompt_length, n = (2, 10) if i % 2 == 0 else (5, 4)
        q = [-(j + 1) / 10 for j in range(n)]
        original = dict(
            index=i,
            tokens=[9] * prompt_length + [3] * n,
            loss_mask=[1] * n,
            response_length=n,
            rollout_log_probs=q,
            rollout_sampling_mask=dict(ids=[3, 7] * n, offsets=list(range(0, 2 * n + 1, 2))),
        )
        local_q = [q[j] for j in selectors[cp_rank][i % 2]]
        sample = dict(
            sample_indices=i,
            tokens=original["tokens"],
            loss_masks=original["loss_mask"],
            response_lengths=n,
            rollout_log_probs=local_q,
            advantages=[1.0] * len(local_q),
            rollout_sampling_mask_ids=original["rollout_sampling_mask"]["ids"],
            rollout_sampling_mask_offsets=original["rollout_sampling_mask"]["offsets"],
        )
        originals.append(original)
        samples.append(sample)
    assert validate_samples(samples, originals, True, cp_size=2, cp_rank=cp_rank) == (64 if cp_rank == 0 else 160)
    samples[0]["rollout_log_probs"][0] -= 0.1
    with pytest.raises(ValueError, match="Behavior transfer"):
        validate_samples(samples, originals, True, cp_size=2, cp_rank=cp_rank)


def test_analyze_rejects_rank_aliases_and_invalid_optimizer_evidence(tmp_path, monkeypatch):
    from argparse import Namespace

    from scripts.tools import sampling_replay_moe as moe

    run = tmp_path / "smoke-on"
    (run / "rollout_data").mkdir(parents=True)
    (run / "train_data").mkdir()
    (run / "preflight.json").write_text(json.dumps({"job_id": "owned-job"}))
    samples, originals = sample_pair()
    torch.save({"samples": originals}, run / "rollout_data/0.pt")
    for rank in range(8):
        local = copy.deepcopy(samples)
        if rank >= 4:
            for sample in local:
                sample["rollout_log_probs"] = []
                sample["advantages"] = []
        torch.save({"rollout_id": 0, "rank": rank, "samples": local}, run / f"train_data/0_{rank}.pt")
        row = dict(
            rank=rank,
            optimizer_step=0,
            optimizer_successful=True,
            weight_decay=0,
            finite=True,
            parameter_finite_after=True,
            optimizer_grad_norm=1.0,
            sampled_gradient_norm=0.0,
            delta_l2=0.0,
            max_abs_delta=0.0,
            changed_elements=0,
            sampled_gradient_nonzero=0,
        )
        (run / f"parameter-updates-rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    # Only the remote namespace-owner/status/log service is isolated. Artifact
    # loading, CP slicing and corruption checks execute the real analyzer.
    monkeypatch.setattr(moe, "checked_state", lambda _: {"netns": os.readlink("/proc/self/ns/net")})
    monkeypatch.setattr(
        moe, "fetch", lambda _url, path: {"logs": "done"} if path.endswith("/logs") else {"status": "SUCCEEDED"}
    )
    args = Namespace(root=tmp_path, mode="on", instance=tmp_path / "ray")
    target = run / "train_data/0_7.pt"
    saved = torch.load(target, weights_only=False)
    torch.save({**saved, "rank": 6}, target)
    with pytest.raises(ValueError, match="mislabeled rank"):
        moe.analyze(args)
    torch.save(saved, target)
    target = run / "parameter-updates-rank0.jsonl"
    row = json.loads(target.read_text())
    for changed in ({"rank": 1}, {"optimizer_step": 1}, {"finite": False}):
        target.write_text(json.dumps({**row, **changed}) + "\n")
        with pytest.raises(ValueError, match="Invalid rank/step"):
            moe.analyze(args)
