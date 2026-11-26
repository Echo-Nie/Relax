# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen MoE stability preparation.

Training requires an explicit submit.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.tools import sampling_replay_moe as moe
from scripts.tools.sampling_replay_exp08 import prompt_key
from scripts.tools.sampling_replay_isolated import MODES


TIERS = {"smoke": (2, "1e-6", 2, 2), "baseline": (8, "1e-6", 4, 64), "pressure": (32, "1e-3", 8, 64)}
PROJECT = "sampling-replay-moe-stability"


def prepare(args):
    manifest = moe.read(args.reference / "manifest.json")
    root = args.root.resolve()
    for protected in (Path(manifest["model"]).resolve(), Path(manifest["source"]).resolve().parent):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Outputs must not overlap shared inputs")
    source = [json.loads(line) for line in Path(manifest["source"]).read_text().splitlines() if line.strip()]
    train = [json.loads(line) for line in (args.split / "train.jsonl").read_text().splitlines() if line.strip()]
    heldout = [json.loads(line) for line in (args.split / "eval.jsonl").read_text().splitlines() if line.strip()][:64]
    if len(train) != 128 or len(heldout) != 64:
        raise ValueError("Require existing 128 train / 64 held-out questions")
    keys = [prompt_key(row) for row in train + heldout]
    if len(set(keys)) != 192 or any(row not in source for row in train + heldout):
        raise ValueError("Split must be distinct, disjoint and unmodified source rows")
    if moe.model_inventory(Path(manifest["model"])) != manifest["model_inventory"]:
        raise ValueError("Model inventory changed")
    if moe.digest(Path(manifest["source"])) != manifest["source_sha256"]:
        raise ValueError("Source changed")
    if (args.root / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists; preserve it")
    args.root.mkdir(parents=True, exist_ok=True)
    data = args.root / "data"
    data.mkdir()
    for name, rows in (("train", train), ("heldout", heldout), ("eval-smoke", heldout[:2])):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    for tier in TIERS:
        for mode in MODES:
            child = args.root / tier / mode
            (child / "data").mkdir(parents=True)
            (child / "data/train.jsonl").symlink_to(data / "train.jsonl")
            (child / "cache").symlink_to(args.reference / "cache", target_is_directory=True)
            moe.write(
                child / "manifest.json",
                {
                    **manifest,
                    "source_indices": [source.index(row) for row in train],
                    "prompt_lengths": None,
                    "train_sha256": moe.digest(data / "train.jsonl"),
                },
            )
    moe.write(
        args.root / "plan.json",
        dict(
            project=PROJECT,
            tiers=TIERS,
            train_questions=128,
            heldout_questions=64,
            samples_per_eval_prompt=2,
            source_sha256=manifest["source_sha256"],
            data_sha256={p.name: moe.digest(p) for p in data.glob("*.jsonl")},
            split_source=str(args.split.resolve()),
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=moe.REPO, text=True).strip(),
            probability_semantics={
                "off": "legacy full-vocab",
                "on": "saved sampling support conditional",
                "r3": "legacy full-vocab, replayed routes",
                "tis": "legacy full-vocab, clipped IS weights 0..2",
                "sc": "native score centering; conditional behavior, candidate correction; different advantage/loss",
            },
            missing=[
                "per-token Actor ratio quantiles and non-singleton mismatch: not natively logged",
                "whole-model parameter delta: only local expert slices observed",
                "active GPU hours: report allocated 8-GPU wall hours instead",
            ],
        ),
    )
    print(args.root / "plan.json")


def training_argv(root: Path, tier: str, mode: str) -> list[str]:
    child = root / tier / mode
    argv = moe.training_argv(child, mode)
    updates, lr, interval, _ = TIERS[tier]
    run = child / f"smoke-{mode}"
    # No full .pt route/train dumps: R3 replication would consume the disk budget.
    for flag in ("--dump-details",):
        i = argv.index(flag)
        del argv[i : i + 2]
    values = {
        "--num-rollout": updates,
        "--lr": lr,
        "--eval-interval": interval,
        "--n-samples-per-eval-prompt": 2,
        "--eval-temperature": 1,
        "--eval-top-p": 0.95,
        "--eval-top-k": 64,
        "--eval-max-response-len": 4096,
        "--eval-max-context-len": 8192,
        "--eval-input-key": "prompt",
        "--eval-label-key": "label",
        "--tb-project-name": PROJECT,
        "--tb-experiment-name": f"{root.name}-{tier}-{mode}-s1234",
        "--rollout-result-dir": run / "results",
        "--custom-rollout-log-function-path": "scripts.training.text.sampling_replay_moe_stability_hooks.rollout_log",
        "--custom-eval-rollout-log-function-path": "scripts.training.text.sampling_replay_moe_stability_hooks.eval_log",
    }
    for key, value in values.items():
        if key in argv:
            argv[argv.index(key) + 1] = str(value)
        else:
            argv.extend([key, str(value)])
    argv.extend(
        [
            "--eval-prompt-data",
            "moe-heldout",
            str(root / "data" / ("eval-smoke.jsonl" if tier == "smoke" else "heldout.jsonl")),
            "--use-clearml",
            "--log-passrate",
        ]
    )
    return argv


def tracking_env(root: Path, tier: str, mode: str) -> dict:
    run = root / tier / mode / f"smoke-{mode}"
    proxy = "http://127.0.0.1:18080"
    bypass = "127.0.0.0/8,127.0.0.1,localhost," + os.environ.get("MASTER_ADDR", "127.0.0.1")
    return dict(
        HTTPS_PROXY=proxy,
        HTTP_PROXY=proxy,
        https_proxy=proxy,
        http_proxy=proxy,
        NO_PROXY=bypass,
        no_proxy=bypass,
        TENSORBOARD_DIR=str(run / "tensorboard_log"),
        CLEARML_CONFIG_FILE=os.environ.get("CLEARML_CONFIG_FILE", str(Path.home() / "clearml.conf")),
    )


def validate_plan(root: Path):
    plan = moe.read(root / "plan.json")
    if any(moe.digest(root / "data" / name) != value for name, value in plan["data_sha256"].items()):
        raise ValueError("Frozen data changed")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise ValueError("Require 20GiB free; no checkpoints/exports/full tensor dumps")
    return plan


def snapshot(args):
    from scripts.tools.sampling_replay_ab import fetch
    from scripts.tools.sampling_replay_isolated import checked_state

    checked_state(args.instance, inside=True)
    run = args.root / args.tier / args.mode / f"smoke-{args.mode}"
    job_id = moe.read(run / "preflight.json")["job_id"]
    state = fetch("http://127.0.0.1:8265", f"/api/jobs/{job_id}")
    logs = fetch("http://127.0.0.1:8265", f"/api/jobs/{job_id}/logs")
    moe.write(run / "job-status.json", state)
    (run / "job.log").write_text(logs["logs"])
    print(json.dumps({k: state.get(k) for k in ("submission_id", "status", "message")}))


def observe(args):
    import time

    from scripts.tools.sampling_replay_ab import fetch
    from scripts.tools.sampling_replay_isolated import checked_state

    checked_state(args.instance, inside=True)
    run = args.root / args.tier / args.mode / f"smoke-{args.mode}"
    job = moe.read(run / "preflight.json")["job_id"]
    with (run / "gpu.csv").open("x") as stream:
        stream.write("timestamp, index, memory.used [MiB], utilization.gpu [%]\n")
        previous = None
        while True:
            stream.write(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id=0,1,2,3,4,5,6,7",
                        "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
            )
            stream.flush()
            state = fetch("http://127.0.0.1:8265", f"/api/jobs/{job}")
            if state["status"] != previous:
                print(state["status"], flush=True)
                previous = state["status"]
            if state["status"] in ("SUCCEEDED", "FAILED", "STOPPED"):
                snapshot(args)
                return
            time.sleep(5)


def verify_records(tier, state, train, evaluations, updates):
    budget, _, interval, questions = TIERS[tier]
    expected_eval = [0, *range(interval, budget + 1, interval)]
    if state != "SUCCEEDED":
        raise ValueError(f"Job terminal state is {state}")
    if [row["rollout_id"] for row in train] != list(range(budget)):
        raise ValueError("Missing/duplicate training batches")
    if [row["completed_updates"] for row in evaluations] != expected_eval:
        raise ValueError("Missing/duplicate evaluation update points")
    if any(
        row["questions"] != questions or row["responses"] != questions * 2 or row["coverage"] != 1
        for row in evaluations
    ):
        raise ValueError("Incomplete independent evaluation")
    if set(updates) != set(range(8)):
        raise ValueError("Missing expert ranks")
    for rows in updates.values():
        if [row["optimizer_step"] for row in rows] != list(range(budget)):
            raise ValueError("Incorrect optimizer update count")
        if any(not row["finite"] or not row["optimizer_successful"] for row in rows):
            raise ValueError("Non-finite or unsuccessful optimizer")
    effective = sum(
        any(
            updates[rank][step]["sampled_gradient_nonzero"] > 0 and updates[rank][step]["changed_elements"] > 0
            for rank in updates
        )
        for step in range(budget)
    )
    return dict(
        updates=budget,
        effective_observed_updates=effective,
        evaluation_updates=expected_eval,
        learning_signal="observed" if effective else "insufficient; not an implementation failure",
    )


def analyze(args):
    import math
    import re

    from clearml import Task
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from scripts.tools.sampling_replay_isolated import checked_state

    checked_state(args.instance, inside=True)
    run = args.root / args.tier / args.mode / f"smoke-{args.mode}"

    def records(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    if (run / "results/quality-stop.json").exists():
        raise ValueError("Quality stop or incomplete evaluation; preserve trigger; not a normal budget completion")
    state = moe.read(run / "job-status.json")
    train = records(run / "results/audit-train.jsonl")
    evaluations = records(run / "results/audit-eval.jsonl")
    updates = {rank: records(run / f"parameter-updates-rank{rank}.jsonl") for rank in range(8)}
    summary = verify_records(args.tier, state["status"], train, evaluations, updates)
    for row in train:
        if len(records(run / f"results/train/{row['rollout_id']}.jsonl")) != 32:
            raise ValueError("Incomplete per-response training record")
    for row in evaluations:
        raw = records(run / f"results/eval/{row['native_rollout_id']}.jsonl")
        if len(raw) != row["responses"]:
            raise ValueError("Incomplete per-response evaluation record")
    accumulator = EventAccumulator(str(run / "tensorboard_log"), size_guidance={"scalars": 0})
    accumulator.Reload()
    metrics = {
        key: [dict(step=event.step, value=event.value) for event in accumulator.Scalars(key)]
        for key in accumulator.Tags()["scalars"]
    }
    if any(not math.isfinite(event["value"]) for rows in metrics.values() for event in rows):
        raise ValueError("Non-finite scalar; no successful acceptance")
    if [event["step"] for event in metrics.get("train/loss", [])] != list(range(TIERS[args.tier][0])):
        raise ValueError("Missing native training loss steps")
    text = (run / "job.log").read_text()
    ids = set(re.findall(r"ClearML Task: created new task id=([a-f0-9]{32})", text))
    if len(ids) != 1:
        raise ValueError("Expected exactly one actual ClearML Task")
    os.environ.update(tracking_env(args.root, args.tier, args.mode))
    task = Task.get_task(task_id=ids.pop())
    reported = task.get_reported_scalars()
    observed = reported.get("audit_eval", {}).get("accuracy_avg_at_2", {})
    if observed.get("x") != summary["evaluation_updates"]:
        raise ValueError("ClearML uploaded eval steps are missing or misaligned")
    if observed.get("y") != [row["accuracy_avg_at_2"] for row in evaluations]:
        raise ValueError("ClearML evaluation values differ from local evidence")
    summary.update(
        status="passed",
        task_id=task.id,
        task_url=task.get_output_log_web_page(),
        train=train,
        evaluations=evaluations,
        metrics=metrics,
        allocated_gpu_hours=8 * (state["end_time"] - state["start_time"]) / 3600000,
        note="Allocated GPU wall hours include init/eval; not measured kernel-active hours",
    )
    moe.write(run / "verification.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "updates", "effective_observed_updates", "task_url")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "argv", "submit", "snapshot", "analyze", "observe"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
        if command == "prepare":
            p.add_argument("--reference", type=Path, required=True)
            p.add_argument("--split", type=Path, required=True)
        else:
            p.add_argument("--tier", choices=TIERS, required=True)
            p.add_argument("--mode", choices=MODES, required=True)
            if command in ("submit", "snapshot", "analyze", "observe"):
                p.add_argument("--instance", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "observe":
        observe(args)
    elif args.command == "snapshot":
        snapshot(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        validate_plan(args.root)
        argv = training_argv(args.root, args.tier, args.mode)
        if args.command == "argv":
            print(json.dumps(argv, indent=2))
        else:
            child = args.root / args.tier / args.mode
            # Existing submit retains GPU/PID/netns/model/data/Serve/job guards.
            moe.submit(
                argparse.Namespace(root=child, mode=args.mode, instance=args.instance),
                argv=argv,
                extra_env=tracking_env(args.root, args.tier, args.mode),
            )


if __name__ == "__main__":
    main()
