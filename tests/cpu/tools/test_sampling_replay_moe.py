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
    (instance / "config.json").write_text(json.dumps(dict(parent_netns=os.readlink("/proc/self/ns/net"))))
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
    monkeypatch.setattr(moe, "checked_state", lambda _, **kwargs: {"netns": os.readlink("/proc/self/ns/net")})
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


def test_job_runtime_env_merges_with_real_ray_driver_environment(tmp_path):
    from ray._private.worker import _merge_runtime_env

    from scripts.tools.sampling_replay_moe import job_runtime_env

    job = job_runtime_env(tmp_path)
    driver = {"env_vars": {"PYTHONPATH": "driver-path", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}
    assert _merge_runtime_env(job, driver, override=False) is None
    merged = _merge_runtime_env(job, driver, override=job["env_vars"].get("RAY_OVERRIDE_JOB_RUNTIME_ENV") == "1")
    assert merged["env_vars"]["PYTHONPATH"] == "driver-path"
    assert merged["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert merged["env_vars"]["HF_HOME"] == str(tmp_path / "cache/hf")


def test_native_job_metrics_require_one_real_step_and_the_frozen_batch():
    from scripts.tools.sampling_replay_moe import metrics_from_job_log

    record = "step 0: {'train/loss': 0.25, 'train/grad_norm': 0.5, 'train/step': 0, 'train/global_batch_size': 32}"
    assert metrics_from_job_log(record)["train/loss"] == [0.25]
    for bad in ("no metrics", record + "\n" + record, record.replace("size': 32", "size': 16")):
        with pytest.raises(ValueError):
            metrics_from_job_log(bad)


def test_reference_capture_checks_r3_routes_and_cp_local_sc_candidates():
    from scripts.tools.sampling_replay_moe import validate_reference_payload

    samples, originals = sample_pair()
    for o in originals:
        o["rollout_routed_experts"] = torch.arange(8, dtype=torch.int32).repeat(1, 48, 1)
    validate_reference_payload(samples, originals, "r3", 0)
    originals[0]["rollout_routed_experts"][0, 0, -1] = 0
    with pytest.raises(ValueError, match="Duplicate experts"):
        validate_reference_payload(samples, originals, "r3", 0)
    for o, sample in zip(originals, samples):
        o["rollout_topk_token_ids"] = [[3] + [-1] * 127]
        o["rollout_topk_log_probs"] = [[0.0] + [-float("inf")] * 127]
        o["rollout_topk_support_complete"] = [True]
        for k in ("rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_topk_support_complete"):
            sample[k] = o[k]
    assert validate_reference_payload(samples, originals, "sc", 0)["complete_support_rows"] == 32
    for sample in samples:
        for k in ("rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_topk_support_complete"):
            sample[k] = []
    validate_reference_payload(samples, originals, "sc", 1)
    samples[0]["rollout_topk_support_complete"] = [True]
    with pytest.raises(ValueError, match="CP transfer"):
        validate_reference_payload(samples, originals, "sc", 1)


def test_native_tis_changes_gradient_with_zero_to_two_clipping():
    from argparse import Namespace

    from relax.backends.megatron.loss import vanilla_tis_function

    pg = torch.ones(3, requires_grad=True)
    weighted, _, metrics = vanilla_tis_function(
        Namespace(tis_clip_low=0.0, tis_clip=2.0),
        pg_loss=pg,
        train_log_probs=[torch.tensor([0.025, 0.1, 0.4]).log()],
        rollout_log_probs=[torch.tensor([0.1, 0.1, 0.1]).log()],
        loss_masks=[torch.ones(3)],
    )
    weighted.sum().backward()
    torch.testing.assert_close(pg.grad, torch.tensor([0.25, 1.0, 2.0]))
    torch.testing.assert_close(metrics["tis_clipfrac"], torch.tensor([0.0, 0.0, 1.0]))
