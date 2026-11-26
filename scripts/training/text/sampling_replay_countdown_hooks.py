# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Countdown instrumentation; native generation, reward and loss stay
authoritative."""

import copy
import importlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


async def generate_bounded(args, sample, sampling_params, evaluation=False):
    bounded = importlib.import_module("examples.algorithms.score-centering.recipe").generate_bounded
    sc = args.advantage_estimator == "score_centering"
    capture_args = args
    if not sc and not evaluation:
        # Same-call native behavior q for every RLOO arm. Only SR sends CSR to
        # the Actor; OFF/TIS keep the full-vocabulary Actor and native losses.
        capture_args = copy.copy(args)
        capture_args.use_rollout_sampling_replay = True
        capture_args.custom_generate_function_path = None
    result = await bounded(capture_args, sample, sampling_params, evaluation=evaluation)
    if len(result.tokens) > 512:
        raise ValueError("Countdown total-token budget exceeded")
    if not sc and not evaluation:
        mask = result.rollout_sampling_mask
        if mask is None or len(mask) != result.response_length:
            raise ValueError("Countdown native behavior capture missing or misaligned")
        sizes = (mask.offsets[1:] - mask.offsets[:-1]).tolist()
        result.metadata = {**(result.metadata or {}), "countdown_support_sizes": sizes}
        if not args.use_rollout_sampling_replay:
            result.rollout_sampling_mask = None
    elif not evaluation and not result.rollout_topk_support_complete.all():
        raise ValueError("SC candidates must cover complete k64 support")
    return result


def signed_score(sample):
    value = float(sample.reward)
    if value not in (-1.0, 1.0):
        raise ValueError("Countdown requires the native signed -1/+1 reward")
    return value


def grouped_metrics(samples, n, expected):
    if len(samples) != n * expected or len({s.index for s in samples}) != len(samples):
        raise ValueError("Missing/duplicate Countdown samples")
    scores = [signed_score(s) for s in samples]
    return dict(
        accuracy=sum(r > 0 for r in scores) / len(scores),
        reward=statistics.mean(scores),
        correct_responses=sum(r > 0 for r in scores),
        samples=len(samples),
        truncated_fraction=sum(s.status == s.Status.TRUNCATED for s in samples) / len(samples),
        response_length_mean=statistics.mean(s.response_length for s in samples),
    )


def quality_stop(history):
    # Allow learning from a weak initial model. Do not call a low initial
    # reward alone a collapse: require persistent loss of earlier performance.
    if len(history) < 15:
        return False
    best = max(r["accuracy"] for r in history[:-5])
    return all(r["accuracy"] <= max(0.01, 0.2 * best) and r["truncated_fraction"] >= 0.9 for r in history[-5:])


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from relax.utils import tracking_utils

    root = Path(args.rollout_result_dir)
    expected = args.rollout_batch_size
    values = grouped_metrics(samples, 8, expected)
    groups = defaultdict(list)
    for sample in samples:
        if len(sample.tokens) > 512:
            raise ValueError("Training token budget exceeded")
        groups[sample.group_index].append(sample)
    if len(groups) != expected or any(len(g) != 8 for g in groups.values()):
        raise ValueError("Incomplete Countdown training Group")
    values["mixed_reward_group_fraction"] = (
        sum(len({signed_score(s) for s in g}) > 1 for g in groups.values()) / expected
    )
    sizes = [size for s in samples for size in (s.metadata or {}).get("countdown_support_sizes", [])]
    if sizes:
        q = statistics.quantiles(sizes, n=100, method="inclusive")
        values.update(
            singleton_fraction=sum(s == 1 for s in sizes) / len(sizes),
            support_mean=statistics.mean(sizes),
            support_p50=q[49],
            support_p95=q[94],
            support_p99=q[98],
            support_max=max(sizes),
        )
    path = root / "audit-train.jsonl"
    append(path, dict(completed_updates_before_batch=rollout_id, **values))
    history = [json.loads(line) for line in path.read_text().splitlines()]
    if quality_stop(history):
        (root / "stop.json").write_text(
            json.dumps(
                dict(
                    kind="quality_collapse",
                    confirmed_before_update=rollout_id,
                    rule="five post-warmup batches: accuracy <= max(.01,.2*earlier best) and truncated >= .9",
                )
            )
            + "\n"
        )
    tracking_utils.log(args, {**{f"audit_train/{k}": v for k, v in values.items()}, "step": rollout_id}, "step")
    return False


def eval_log(rollout_id, args, data, extra_metrics):
    from relax.utils import tracking_utils

    root = Path(args.rollout_result_dir)
    path = root / "audit-eval.jsonl"
    history = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = rollout_id + 1 if history else 0
    try:
        if (not history and rollout_id != 0) or (history and completed <= history[-1]["completed_updates"]):
            raise ValueError("Missing initial or duplicate evaluation")
        info = data["countdown-dev"]
        expected = len(Path(args.eval_prompt_data[1]).read_text().splitlines())
        if info["num_prompts"] != expected or info["num_valid_prompts"] != expected or info["group_size"] != 8:
            raise ValueError("Incomplete Countdown dev coverage")
        values = grouped_metrics(info["samples"], 8, expected)
        if sorted(s.index for s in info["samples"]) != list(range(expected * 8)):
            raise ValueError("Invalid dev sample indices")
        ordered = sorted(info["samples"], key=lambda s: s.index)
        values["correct_questions_at_least_one"] = sum(
            any(signed_score(s) > 0 for s in ordered[i : i + 8]) for i in range(0, len(ordered), 8)
        )
        values["questions"] = expected
        values["coverage"] = 1.0
    except (KeyError, TypeError, ValueError) as error:
        root.mkdir(parents=True, exist_ok=True)
        (root / "stop.json").write_text(json.dumps(dict(kind="incomplete_eval", error=str(error))) + "\n")
        raise
    row = dict(completed_updates=completed, native_rollout_id=rollout_id, **values)
    append(path, row)
    tracking_utils.log(args, {**{f"audit_eval/{k}": v for k, v in values.items()}, "step": completed}, "step")
    tracking_utils.flush_metrics(args, completed)
    return False


def before_train(args, rollout_id, step_id, model, optimizer, scheduler):
    import torch.distributed as dist

    from scripts.training.text.sampling_replay_moe_hooks import install_observer

    root = Path(args.rollout_result_dir)
    if (root / "stop.json").exists():
        raise ValueError("Countdown explicit stop marker; preserve evidence")
    install_observer(
        model,
        optimizer,
        output=root.parent,
        rank=dist.get_rank(group=dist.group.WORLD),
        weight_decay=args.weight_decay,
        require_optimizer_norm=False,
        pattern=re.compile(r"decoder\.layers\.0\.mlp\.linear_fc2\.weight$"),
        observation_scope="first 65536 elements of layer0 dense fc2; DP replicas, not independent expert shards",
    )
