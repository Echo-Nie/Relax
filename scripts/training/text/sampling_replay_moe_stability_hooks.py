# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experiment-only metrics and predeclared quality stop; no loss
modification."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def reward_value(sample, key):
    reward = sample.reward[key] if isinstance(sample.reward, dict) else sample.reward
    value = float(reward)
    if value not in (-1.0, 1.0):
        raise ValueError("DAPO smoke/stability protocol requires finite score -1/+1")
    return value


def rollout_metrics(samples, reward_key, group_size):
    if len(samples) != 32 or len({sample.index for sample in samples}) != 32:
        raise ValueError("Expected 32 unique training responses")
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)
    if len(groups) != 4 or any(len(group) != group_size for group in groups.values()):
        raise ValueError("Incomplete training Group")
    rewards = [reward_value(s, reward_key) for s in samples]
    nonzero_tokens = 0
    mixed = 0
    for group in groups.values():
        if len({reward_value(s, reward_key) for s in group}) > 1:
            mixed += 1
            nonzero_tokens += sum(s.response_length for s in group)
    tokens = sum(s.response_length for s in samples)
    result = {
        "accuracy": sum(r > 0 for r in rewards) / len(rewards),
        "mixed_reward_group_fraction": mixed / len(groups),
        "nonzero_reward_advantage_token_fraction": nonzero_tokens / tokens if tokens else 0.0,
    }
    result["truncated_fraction"] = sum(s.status == s.Status.TRUNCATED for s in samples) / len(samples)
    result["response_length_mean"] = tokens / len(samples)
    masks = [s.rollout_sampling_mask for s in samples]
    if any(mask is not None for mask in masks):
        if any(mask is None for mask in masks):
            raise ValueError("Partial SR support")
        sizes = []
        for sample, mask in zip(samples, masks, strict=True):
            if len(mask) != sample.response_length or len(sample.rollout_log_probs) != sample.response_length:
                raise ValueError("Support/token/behavior length mismatch")
            sizes.extend((mask.offsets[1:] - mask.offsets[:-1]).tolist())
        if len(sizes) != tokens:
            raise ValueError("Support/token mismatch")
        percentiles = statistics.quantiles(sizes, n=100, method="inclusive")
        result.update(
            support_p50=percentiles[49],
            support_p95=percentiles[94],
            support_p99=percentiles[98],
            singleton_fraction=sum(size == 1 for size in sizes) / tokens,
            support_mean=sum(sizes) / tokens,
            support_max=max(sizes),
        )
    return result


def evaluation_metrics(info, expected_questions, samples_per_question):
    samples = info.get("samples", [])
    if (
        info.get("num_prompts") != expected_questions
        or info.get("num_valid_prompts") != expected_questions
        or info.get("group_size") != samples_per_question
        or len(samples) != expected_questions * samples_per_question
        or sorted(s.index for s in samples) != list(range(len(samples)))
    ):
        raise ValueError("Incomplete independent evaluation; do not accept Job status alone")
    ordered = sorted(samples, key=lambda s: s.index)
    rewards = [reward_value(s, "score") for s in ordered]
    correct = sum(value > 0 for value in rewards)
    solved = sum(
        any(value > 0 for value in rewards[i : i + samples_per_question])
        for i in range(0, len(rewards), samples_per_question)
    )
    return {
        "accuracy_avg_at_2": correct / len(rewards),
        "reward": sum(rewards) / len(rewards),
        "correct_responses": correct,
        "correct_questions_at_least_one": solved,
        "questions": expected_questions,
        "responses": len(rewards),
        "coverage": 1.0,
    }


def quality_collapse(history):
    """Stop only at a predeclared eval boundary, preserve the triggering
    evidence."""
    if len(history) < 3:
        return False
    best = max(row["accuracy_avg_at_2"] for row in history[:-2])
    return best >= 0.25 and all(row["accuracy_avg_at_2"] <= best - 0.25 for row in history[-2:])


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from relax.utils import tracking_utils

    values = rollout_metrics(samples, args.reward_key, args.n_samples_per_prompt)
    root = Path(args.rollout_result_dir)
    history_path = root / "audit-train.jsonl"
    _append(history_path, dict(rollout_id=rollout_id, completed_updates_before_batch=rollout_id, **values))
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    if len(history) >= 3 and all(row["truncated_fraction"] >= 0.8 and row["accuracy"] <= 0.1 for row in history[-3:]):
        (root / "quality-stop.json").write_text(
            json.dumps({"reason": "three batches: truncation >= .8 and accuracy <= .1", "rollout_id": rollout_id})
            + "\n"
        )
    tracking_utils.log(args, {**{f"audit_train/{k}": v for k, v in values.items()}, "step": rollout_id}, "step")
    return False  # Preserve every native metric and its existing reduction.


def eval_log(rollout_id, args, data, extra_metrics):
    from relax.utils import tracking_utils

    path = Path(args.rollout_result_dir) / "audit-eval.jsonl"
    history = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if not history and rollout_id != 0:
        raise ValueError("Missing initial independent evaluation")
    completed = rollout_id + 1 if history else 0
    if history and completed <= history[-1]["completed_updates"]:
        raise ValueError("Duplicate/out-of-order independent evaluation")
    expected = len(Path(args.eval_prompt_data[1]).read_text().splitlines())
    try:
        values = evaluation_metrics(data["moe-heldout"], expected, args.n_samples_per_eval_prompt)
    except ValueError as error:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "quality-stop.json").write_text(
            json.dumps({"reason": str(error), "rollout_id": rollout_id}) + "\n"
        )
        raise
    row = dict(native_rollout_id=rollout_id, completed_updates=completed, **values)
    _append(path, row)
    tracking_utils.log(args, {**{f"audit_eval/{k}": v for k, v in values.items()}, "step": completed}, "step")
    tracking_utils.flush_metrics(args, completed)
    # The rollout callback may be caught by Actor's HTTP warning path, so set
    # an explicit local stop marker that the next before_train hook checks.
    if quality_collapse(history + [row]):
        (path.parent / "quality-stop.json").write_text(json.dumps(row) + "\n")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite independent evaluation")
    return False
