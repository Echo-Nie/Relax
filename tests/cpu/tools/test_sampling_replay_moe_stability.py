# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import socket
import threading

import pytest

from relax.utils.sampling_mask import RolloutSamplingMask
from relax.utils.types import Sample
from scripts.tools.sampling_replay_moe_stability import verify_records
from scripts.tools.sampling_replay_tracking_proxy import ClearMLTunnel, TrackingServer
from scripts.training.text.sampling_replay_moe_stability_hooks import (
    evaluation_metrics,
    quality_collapse,
    rollout_metrics,
)


def test_eval_distinguishes_average_accuracy_from_any_correct_question():
    samples = [Sample(index=i, reward={"score": r}) for i, r in enumerate([1, -1, -1, -1])]
    info = dict(samples=samples, num_prompts=2, num_valid_prompts=2, group_size=2)
    result = evaluation_metrics(info, 2, 2)
    assert result["accuracy_avg_at_2"] == 0.25
    assert result["reward"] == -0.5
    assert result["correct_responses"] == result["correct_questions_at_least_one"] == 1
    samples[-1].index = 2
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation_metrics(info, 2, 2)


def test_train_metrics_preserve_support_and_mixed_reward_group_semantics():
    samples = [
        Sample(
            index=i,
            group_index=i // 8,
            reward={"score": 1 if i == 0 else -1},
            tokens=[5, 3, 4],
            response_length=2,
            rollout_log_probs=[0.0, -0.5],
            rollout_sampling_mask=RolloutSamplingMask.from_rows([[3], [4, 7]]),
        )
        for i in range(32)
    ]
    result = rollout_metrics(samples, "score", 8)
    assert result["accuracy"] == 1 / 32
    assert result["mixed_reward_group_fraction"] == 0.25
    assert result["nonzero_reward_advantage_token_fraction"] == 0.25
    assert result["singleton_fraction"] == 0.5
    assert result["support_mean"] == 1.5
    samples[0].rollout_log_probs = [0.0]
    with pytest.raises(ValueError, match="length mismatch"):
        rollout_metrics(samples, "score", 8)


def test_quality_rule_requires_two_evals_and_predeclared_absolute_drop():
    history = [{"accuracy_avg_at_2": value} for value in [0.5, 0.25, 0.25]]
    assert quality_collapse(history)
    assert not quality_collapse(history[:2])
    assert not quality_collapse([{"accuracy_avg_at_2": value} for value in [0.5, 0.25, 0.26]])


def test_verification_rejects_succeeded_job_with_missing_final_eval_or_rank_update():
    train = [{"rollout_id": i} for i in range(2)]
    evaluations = [dict(completed_updates=u, questions=2, responses=4, coverage=1) for u in (0, 2)]
    updates = {
        rank: [
            dict(
                optimizer_step=i,
                finite=True,
                optimizer_successful=True,
                sampled_gradient_nonzero=1,
                changed_elements=1,
            )
            for i in range(2)
        ]
        for rank in range(8)
    }
    assert verify_records("smoke", "SUCCEEDED", train, evaluations, updates)["effective_observed_updates"] == 2
    with pytest.raises(ValueError, match="evaluation update"):
        verify_records("smoke", "SUCCEEDED", train, evaluations[:-1], updates)
    corrupt = copy.deepcopy(updates)
    corrupt[7].pop()
    with pytest.raises(ValueError, match="optimizer update"):
        verify_records("smoke", "SUCCEEDED", train, evaluations, corrupt)


def test_tracking_tunnel_refuses_unconfigured_destination():
    with TrackingServer(("127.0.0.1", 0), ClearMLTunnel) as server:
        server.allowed_targets = {"configured-clearml.example:443"}
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=5) as client:
                client.sendall(b"CONNECT arbitrary.example:443 HTTP/1.1\r\n\r\n")
                assert client.recv(4096).startswith(b"HTTP/1.1 403 Forbidden")
        finally:
            server.shutdown()
            worker.join(timeout=5)
        assert not worker.is_alive()
