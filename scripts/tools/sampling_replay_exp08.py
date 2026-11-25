# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen EXP-08 inputs and offline analysis.

Never submits training or inference.
"""

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts.tools.sampling_replay_evaluate import read_json, rows, sha, write_json


def prompt_key(row: dict) -> str:
    messages = [{**m, "content": " ".join(m["content"].split())} for m in row["prompt"]]
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def split_indices(data: list[dict], excluded: set[str], lengths: list[int]) -> dict[str, list[int]]:
    seen = set(excluded)
    eligible = []
    for i, row in enumerate(data):
        key = prompt_key(row)
        if key not in seen and lengths[i] <= 1024:
            seen.add(key)
            eligible.append(i)
    random.Random(20261125).shuffle(eligible)
    if len(eligible) < 384:
        raise ValueError("Need 128 training and 256 distinct held-out prompts")
    return {"train": eligible[:128], "eval": eligible[128:384]}


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    data = rows(args.data)
    smoke = rows(args.smoke)
    source_by_key = {prompt_key(row): row for row in data}
    if len(smoke) != 4 or any(source_by_key.get(prompt_key(row)) != row for row in smoke):
        raise ValueError("Smoke must contain four unmodified source questions")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lengths = [
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                ),
                add_special_tokens=False,
            )
        )
        for row in data
    ]
    excluded = {prompt_key(row) for row in rows(args.exclude)} | {prompt_key(row) for row in smoke}
    splits = split_indices(data, excluded, lengths)
    target = args.root / "data"
    target.mkdir(exist_ok=False)
    for name, selected in splits.items():
        (target / f"{name}.jsonl").write_text(
            "".join(json.dumps(data[i], ensure_ascii=False) + "\n" for i in selected)
        )
    shutil.copyfile(args.smoke, target / "smoke.jsonl")
    (target / "eval-smoke.jsonl").write_text(
        "".join(json.dumps(data[i], ensure_ascii=False) + "\n" for i in splits["eval"][:2])
    )
    base = read_json(args.base_argv)
    base = base[base.index("relax.entrypoints.train") + 1 :]
    write_json(args.root / "base-argv.json", base)
    manifest = dict(
        source=str(args.data.resolve()),
        source_sha256=sha(args.data),
        source_rows=len(data),
        model=str(args.model.resolve()),
        model_sha256={p.name: sha(p) for p in sorted(args.model.iterdir()) if p.is_file()},
        data_sha256={p.name: sha(p) for p in target.glob("*.jsonl")},
        source_indices=splits,
        split_seed=20261125,
        canonical_whitespace_overlap=0,
        excluded_source=str(args.exclude.resolve()),
        excluded_sha256=sha(args.exclude),
        base_argv_sha256=sha(args.root / "base-argv.json"),
        training=dict(updates=32, seed=1234, prompts=4, samples=8, global_batch=32, max_response=4096),
        evaluation=dict(
            questions=256,
            samples=4,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_response_len=4096,
            max_context_len=8192,
            thinking=False,
            completion_seeds=[1234, 1235, 1236, 1237],
            concurrency=8,
            updates=[0, 16, 32],
            common_u0=True,
            metric="mean of per-question mean correctness (Avg@4)",
        ),
    )
    write_json(args.root / "manifest.json", manifest)
    write_json(args.root / "evaluation-argv.json", training_argv(args.root, "off", "pilot"))
    manifest["evaluation_argv_sha256"] = sha(args.root / "evaluation-argv.json")
    write_json(args.root / "manifest.json", manifest)
    for name, selected in [("eval", splits["eval"]), ("eval-smoke", splits["eval"][:2])]:
        write_json(
            args.root / f"{name}-manifest.json",
            dict(
                source_indices=selected,
                data_sha256=sha(target / f"{name}.jsonl"),
                training_argv_sha256=manifest["evaluation_argv_sha256"],
            ),
        )


def training_argv(root: Path, mode: str, stage: str) -> list[str]:
    manifest = read_json(root / "manifest.json")
    if sha(root / "base-argv.json") != manifest["base_argv_sha256"]:
        raise ValueError("Base training argv changed")
    argv = list(read_json(root / "base-argv.json"))
    argv.remove("--use-rollout-sampling-replay")
    i = argv.index("--eval-prompt-data")
    del argv[i : i + 3]
    for key in (
        "--eval-interval",
        "--n-samples-per-eval-prompt",
        "--eval-temperature",
        "--eval-top-k",
        "--eval-top-p",
        "--eval-max-response-len",
        "--eval-max-context-len",
    ):
        i = argv.index(key)
        del argv[i : i + 2]
    run = root / f"{stage}-{mode}-s1234"
    values = {
        "--prompt-data": root / "data" / ("smoke.jsonl" if stage == "smoke" else "train.jsonl"),
        "--hf-checkpoint": manifest["model"],
        "--load": manifest["model"],
        "--num-rollout": 1 if stage == "smoke" else 32,
        "--rollout-batch-size": 4,
        "--n-samples-per-prompt": 8,
        "--global-batch-size": 32,
        "--rollout-temperature": 1,
        "--rollout-top-p": 0.95,
        "--rollout-top-k": 64,
        "--rollout-seed": 1234,
        "--sglang-random-seed": 1234,
        "--dump-details": run,
        "--tb-project-name": "sampling-replay-exp08",
        "--tb-experiment-name": f"sr-exp08-{stage}-{mode}-s1234",
        "--save": run / "checkpoints",
        "--save-interval": 1 if stage == "smoke" else 16,
    }
    for key, value in values.items():
        if key in argv:
            argv[argv.index(key) + 1] = str(value)
        else:
            argv.extend([key, str(value)])
    argv.append("--no-save-optim")
    if mode == "on":
        argv.append("--use-rollout-sampling-replay")
    return argv


def preflight(args: argparse.Namespace) -> None:
    from scripts.tools.sampling_replay_ab import preflight as checked_preflight
    from scripts.tools.sampling_replay_evaluate import process_identity
    from scripts.tools.sampling_replay_isolated import GPU_PAIRS

    instance = args.root / f"ray-{args.stage}-{args.mode}"
    owner = read_json(instance / "state.json")
    config = read_json(instance / "config.json")
    if (
        os.readlink("/proc/self/ns/net") != owner["netns"]
        or owner["netns"] == config["parent_netns"]
        or process_identity(owner["pid"]) != owner["start_ticks"]
    ):
        raise ValueError("Run preflight inside the recorded private experiment namespace")
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            GPU_PAIRS[args.mode],
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if any(
        memory > 100 or utilization
        for memory, utilization in (map(int, row.split(",")) for row in gpu_rows.strip().splitlines())
    ):
        raise ValueError("Experiment GPUs are busy; do not clean or share them")

    if shutil.disk_usage(args.root).free < 120 * 1024**3:
        raise ValueError("EXP-08 requires at least 120 GiB free for checkpoints, exports and dumps")
    if args.stage == "pilot" and subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Pilot requires a clean source checkout")
    manifest = read_json(args.root / "manifest.json")
    if sha(args.root / "base-argv.json") != manifest["base_argv_sha256"]:
        raise ValueError("Base training argv changed")
    checked_preflight(
        argparse.Namespace(
            root=args.root,
            model=Path(manifest["model"]),
            dashboard=args.dashboard,
            job=f"sr-exp08-{args.stage}-{args.mode}-s1234",
            run=args.root / f"{args.stage}-{args.mode}-s1234",
        )
    )


def summarize_avg(records: list[dict], expected: list[int], samples: int) -> dict:
    indexed = {(r["source_index"], r["completion_index"]): r for r in records}
    wanted = {(i, j) for i in expected for j in range(samples)}
    if len(indexed) != len(records) or set(indexed) != wanted:
        raise ValueError("Missing, duplicate or substituted evaluation completions")
    per_question = {
        str(i): sum(bool(indexed[i, j]["reward"]["acc"]) for j in range(samples)) / samples for i in expected
    }
    rewards = [r["reward"]["score"] for r in records]
    if not all(math.isfinite(v) for v in rewards):
        raise ValueError("Non-finite evaluation reward")
    return dict(
        questions=len(expected),
        completions=len(records),
        avg_accuracy=sum(per_question.values()) / len(expected),
        reward=sum(rewards) / len(rewards),
        truncated=sum(r["status"] == "truncated" for r in records),
        per_question_accuracy=per_question,
        any_correct=sum(v > 0 for v in per_question.values()),
    )


def analyze_eval(args: argparse.Namespace) -> None:
    manifest = read_json(args.root / "manifest.json")
    expected = manifest["source_indices"]["eval"][:2] if args.stage == "smoke" else manifest["source_indices"]["eval"]
    names = ["common-u0", "off-u16", "on-u16", "off-u32", "on-u32"] if args.stage == "pilot" else ["off-u1", "on-u1"]
    result = {}
    protocol = None
    for name in names:
        run = args.root / "evaluations" / args.stage / name
        status = read_json(run / "status.json")
        if status["state"] != "SUCCEEDED":
            raise ValueError(f"Incomplete evaluation: {name}")
        params = {
            k: status["arguments"][k]
            for k in (
                "samples_per_prompt",
                "temperature",
                "top_p",
                "top_k",
                "max_response_len",
                "max_context_len",
                "chat_template_kwargs",
                "concurrency",
                "fixed_sampling_seeds",
            )
        }
        if protocol is not None and protocol != params:
            raise ValueError("Evaluation protocols differ")
        protocol = params
        if (
            status["data_sha256"]
            != manifest["data_sha256"]["eval-smoke.jsonl" if args.stage == "smoke" else "eval.jsonl"]
        ):
            raise ValueError("Evaluation dataset changed")
        result[name] = summarize_avg(rows(run / "repeat-0.jsonl"), expected, 4)
    comparisons = {}
    if args.stage == "pilot":
        baseline = result["common-u0"]["avg_accuracy"]
        for step in (16, 32):
            on, off = result[f"on-u{step}"], result[f"off-u{step}"]
            comparisons[f"u{step}"] = dict(
                on_minus_off=on["avg_accuracy"] - off["avg_accuracy"],
                on_minus_u0=on["avg_accuracy"] - baseline,
                off_minus_u0=off["avg_accuracy"] - baseline,
                per_question_on_minus_off={
                    i: on["per_question_accuracy"][i] - off["per_question_accuracy"][i]
                    for i in on["per_question_accuracy"]
                },
            )
    write_json(
        args.root / f"{args.stage}-evaluation-analysis.json",
        dict(protocol=protocol, results=result, comparisons=comparisons),
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "per_question_accuracy"} for k, v in result.items()}, indent=2
        )
    )


def distribution(values: list[float]) -> dict:
    import torch

    x = torch.tensor(values, dtype=torch.float64)
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Empty/non-finite distribution")
    return dict(
        n=x.numel(),
        mean=float(x.mean()),
        p95=float(x.quantile(0.95)),
        p99=float(x.quantile(0.99)),
        min=float(x.min()),
        max=float(x.max()),
    )


def analyze_train(args: argparse.Namespace) -> None:
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    result = {}
    expected = 1 if args.stage == "smoke" else 32
    for mode in ("off", "on"):
        run = args.root / f"{args.stage}-{mode}-s1234"
        updates = [json.loads(x) for x in (run / "parameter-updates.jsonl").read_text().splitlines()]
        tb = EventAccumulator(
            str(run / "checkpoints/tensorboard_log"),
            size_guidance={"scalars": 0},
        )
        tb.Reload()
        metrics = {k: [dict(step=e.step, value=e.value) for e in tb.Scalars(k)] for k in tb.Tags()["scalars"]}
        if not all(math.isfinite(e["value"]) for v in metrics.values() for e in v):
            raise ValueError("Non-finite metric")
        all_delta = []
        ns_ratio = []
        acc = []
        nonzero = 0
        mixed = 0
        total_tokens = 0
        singleton = 0
        ns = []
        prompts = []
        truncated = 0
        csr_bytes = 0
        dumps = sorted((run / "train_data").glob("*.pt"), key=lambda p: int(p.stem.split("_")[0]))
        if len(dumps) != expected or len(updates) != expected or not all(u["optimizer_successful"] for u in updates):
            raise ValueError("Incomplete optimizer/data budget")
        for p in dumps:
            samples = torch.load(p, map_location="cpu", weights_only=False)["samples"]
            originals = torch.load(
                run / "rollout_data" / (p.stem.split("_")[0] + ".pt"), map_location="cpu", weights_only=False
            )["samples"]
            byid = {s["index"]: s for s in originals}
            groups = defaultdict(list)
            if len(samples) != 32 or len(byid) != 32 or {s["sample_indices"] for s in samples} != set(byid):
                raise ValueError("Incomplete training batch")
            for s in samples:
                o = byid[s["sample_indices"]]
                n = s["response_lengths"]
                if (
                    s["tokens"] != o["tokens"]
                    or s["loss_masks"] != o["loss_mask"]
                    or n != len(s["log_probs"])
                    or n != len(s["rollout_log_probs"])
                    or n != len(s["loss_masks"])
                ):
                    raise ValueError("Training token alignment changed")
                if not torch.equal(
                    torch.tensor(s["rollout_log_probs"], dtype=torch.float32),
                    torch.tensor(o["rollout_log_probs"], dtype=torch.float32),
                ):
                    raise ValueError("Behavior changed across transfer")
                adv = torch.tensor(s["advantages"])
                delta = torch.tensor(s["log_probs"]) - torch.tensor(s["rollout_log_probs"])
                if not torch.isfinite(adv).all() or not torch.isfinite(delta).all():
                    raise ValueError("Non-finite training data")
                all_delta.extend(delta.tolist())
                nonzero += int(torch.count_nonzero(adv))
                acc.append(o["reward"]["acc"])
                groups[o["group_index"]].append(o["reward"]["acc"])
                prompts.append(prompt_key({"prompt": [{"content": o["prompt"]}]}))
                truncated += o["status"] == "truncated"
                total_tokens += n
                if mode == "on":
                    ids, offsets = s["rollout_sampling_mask_ids"], s["rollout_sampling_mask_offsets"]
                    if (
                        ids != o["rollout_sampling_mask"]["ids"]
                        or offsets != o["rollout_sampling_mask"]["offsets"]
                        or len(offsets) != n + 1
                        or offsets[-1] != len(ids)
                    ):
                        raise ValueError("CSR changed across transfer")
                    sizes = torch.diff(torch.tensor(offsets))
                    ns_ratio.extend(delta[sizes > 1].exp().tolist())
                    singleton += int((sizes == 1).sum())
                    ns.extend(delta[sizes > 1].abs().tolist())
                    csr_bytes += 4 * len(ids) + 8 * len(offsets)
                    for i, t in enumerate(s["tokens"][-n:]):
                        if t not in ids[offsets[i] : offsets[i + 1]]:
                            raise ValueError("Sampled token outside support")
                elif s.get("rollout_sampling_mask_ids") is not None:
                    raise ValueError("OFF unexpectedly contains support")
            if len(groups) != 4 or any(len(v) != 8 for v in groups.values()):
                raise ValueError("Incomplete Advantage groups")
            mixed += sum(0 < sum(v) < len(v) for v in groups.values())
        status = read_json(run / "job-status.json")
        if status["status"] != "SUCCEEDED":
            raise ValueError("Job has not succeeded")
        peaks = [
            ast.literal_eval(match.group(1))
            for line in (run / "job.log").read_text().splitlines()
            if "Step-Memory-Peak key=actor-" in line
            and "phase=train" in line
            and (match := re.search(r"memory=(\{.*\})", line))
        ]
        if len(peaks) != expected:
            raise ValueError("Incomplete training memory events")
        result[mode] = dict(
            actor_peak_allocated_GiB=max(p["peak_allocated_GB"] for p in peaks),
            actor_peak_reserved_GiB=max(p["peak_reserved_GB"] for p in peaks),
            job_wall_seconds=(status["end_time"] - status["start_time"]) / 1000,
            updates=len(updates),
            accuracy=sum(acc) / len(acc),
            reward=2 * sum(acc) / len(acc) - 1,
            samples=len(acc),
            mixed_groups=mixed,
            nonzero_advantage_tokens=nonzero,
            nonzero_gradient_steps=sum(e["value"] > 0 for e in metrics["train/grad_norm"]),
            parameter_changed_steps=sum(u["changed_elements"] > 0 for u in updates),
            truncated=truncated,
            response_tokens=total_tokens,
            singleton_fraction=singleton / total_tokens if mode == "on" else None,
            csr_bytes=csr_bytes if mode == "on" else None,
            non_singleton_mismatch=distribution(ns) if ns else None,
            all_mismatch=distribution([abs(x) for x in all_delta]),
            all_ratio=distribution([math.exp(x) for x in all_delta]),
            non_singleton_ratio=distribution(ns_ratio) if ns_ratio else None,
            entropy_semantics="conditional saved support" if mode == "on" else "full vocabulary",
            metrics=metrics,
            prompt_order=prompts,
            initial_parameter_sha=updates[0]["before_sha256"],
        )
    if (
        result["on"]["prompt_order"] != result["off"]["prompt_order"]
        or result["on"]["initial_parameter_sha"] != result["off"]["initial_parameter_sha"]
    ):
        raise ValueError("ON/OFF initial state or data order differ")
    write_json(args.root / f"{args.stage}-training-analysis.json", result)
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a not in ("metrics", "prompt_order")} for k, v in result.items()},
            indent=2,
        )
    )


def snapshot(args: argparse.Namespace) -> None:
    from scripts.tools.sampling_replay_ab import fetch

    run = args.root / f"{args.stage}-{args.mode}-s1234"
    url = args.dashboard + f"/api/jobs/sr-exp08-{args.stage}-{args.mode}-s1234"
    write_json(run / "job-status.json", fetch(url, ""))
    (run / "job.log").write_text(fetch(url, "/logs")["logs"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "argv", "preflight", "analyze-train", "analyze-eval", "snapshot"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["on", "off"])
    p.add_argument("--stage", choices=["smoke", "pilot"], default="pilot")
    for name in ["data", "model", "smoke", "exclude", "base-argv"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument("--dashboard")
    a = p.parse_args()
    required = {
        "prepare": ["data", "model", "smoke", "exclude", "base_argv"],
        "argv": ["mode"],
        "preflight": ["mode", "dashboard"],
        "snapshot": ["mode", "dashboard"],
        "analyze-train": [],
        "analyze-eval": [],
    }
    for field in required[a.command]:
        if getattr(a, field) is None:
            p.error(f"{a.command} requires --{field.replace('_', '-')}")
    if a.command == "argv":
        print(json.dumps(training_argv(a.root, a.mode, a.stage)))
    else:
        {
            "prepare": prepare,
            "preflight": preflight,
            "analyze-train": analyze_train,
            "analyze-eval": analyze_eval,
            "snapshot": snapshot,
        }[a.command](a)


if __name__ == "__main__":
    main()
