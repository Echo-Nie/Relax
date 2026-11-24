# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare immutable paired datasets and inspect single, explicitly submitted
jobs."""

import argparse
import hashlib
import json
import math
import random
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt_key(row: dict) -> str:
    return json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)


def fetch(dashboard: str, endpoint: str) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(dashboard.rstrip("/") + endpoint, timeout=30) as response:
        return json.load(response)


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "manifest.json").exists():
        raise ValueError("Dataset manifest already exists; preserve it and use preflight to verify it")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    smoke = [json.loads(line) for line in args.smoke.read_text().splitlines() if line.strip()]
    original = {prompt_key(row): row for row in source}
    assert len(smoke) == 2 and all(original[prompt_key(row)] == row for row in smoke)
    excluded = {prompt_key(row) for row in smoke}
    candidates = []
    for index, row in enumerate(source):
        key = prompt_key(row)
        if key in excluded:
            continue
        excluded.add(key)
        tokens = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True)
        if len(tokens) <= 1024:
            candidates.append((index, row))
    random.Random(20261124).shuffle(candidates)
    assert len(candidates) >= 322
    splits = {"train": candidates[:256], "eval": candidates[256:320], "smoke-eval": candidates[320:322]}
    data_dir = args.root / "data"
    data_dir.mkdir()
    for name, selected in splits.items():
        (data_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row in selected)
        )
    (data_dir / "smoke.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in smoke))
    weights = sorted(args.model.glob("*.safetensors"))
    assert weights
    manifest = {
        "source": str(args.data.resolve()),
        "source_sha256": digest(args.data),
        "source_rows": len(source),
        "model": str(args.model.resolve()),
        "model_sha256": {p.name: digest(p) for p in weights},
        "data_sha256": {p.name: digest(p) for p in sorted(data_dir.glob("*.jsonl"))},
        "source_indices": {name: [i for i, _ in selected] for name, selected in splits.items()},
        "split_seed": 20261124,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"source_rows": len(source), "split_rows": {k: len(v) for k, v in splits.items()}}))


def preflight(args: argparse.Namespace) -> None:
    manifest = json.loads((args.root / "manifest.json").read_text())
    if args.job.startswith("sr-ab-formal-"):
        assert not subprocess.check_output(["git", "status", "--porcelain"], text=True), (
            "Formal runs require a clean source checkout"
        )
    assert str(args.model.resolve()) == manifest["model"]
    assert digest(Path(manifest["source"])) == manifest["source_sha256"]
    for name, expected in manifest["model_sha256"].items():
        assert digest(args.model / name) == expected, name
    for name, expected in manifest["data_sha256"].items():
        assert digest(args.root / "data" / name) == expected, name
    jobs = fetch(args.dashboard, "/api/jobs/")
    assert not [j for j in jobs if j["status"] in ("RUNNING", "PENDING")], "Ray has active jobs; do not share Serve"
    assert not any(j.get("submission_id") == args.job for j in jobs), "Job ID already used; preserve its evidence"
    serve = fetch(args.dashboard, "/api/serve/applications/")
    assert not serve["applications"], "Serve has applications; do not clean shared state"
    import ray

    ray.init(address="auto", logging_level="ERROR")
    available = ray.available_resources()
    ray.shutdown()
    assert available.get("GPU", 0) >= 2, available
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / "preflight.json").write_text(
        json.dumps(
            {
                "job_id": args.job,
                "manifest_sha256": digest(args.root / "manifest.json"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "available_resources": available,
            },
            indent=2,
        )
    )


def distribution(values: list[float]) -> dict:
    import torch

    t = torch.tensor(values, dtype=torch.float64)
    assert t.numel() and torch.isfinite(t).all()
    return {
        "n": len(values),
        "mean": float(t.mean()),
        "p95": float(t.quantile(0.95)),
        "p99": float(t.quantile(0.99)),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def analyze(args: argparse.Namespace) -> None:
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    plan = json.loads((args.run / "preflight.json").read_text())
    job = fetch(args.dashboard, "/api/jobs/" + plan["job_id"])
    (args.run / "job-status.json").write_text(json.dumps(job, indent=2))
    logs = fetch(args.dashboard, "/api/jobs/" + plan["job_id"] + "/logs")["logs"]
    (args.run / "job.log").write_text(logs)
    updates = [json.loads(line) for line in (args.run / "parameter-updates.jsonl").read_text().splitlines()]
    tb = EventAccumulator(str(args.tensorboard))
    tb.Reload()
    metrics = {
        tag: [{"step": e.step, "wall_time": e.wall_time, "value": e.value} for e in tb.Scalars(tag)]
        for tag in tb.Tags()["scalars"]
    }
    assert all(math.isfinite(e["value"]) for entries in metrics.values() for e in entries)
    (args.run / "tensorboard.json").write_text(json.dumps(metrics, indent=2))
    rows = []
    nonzero_advantages = 0
    sample_count = 0
    replay = "-on-" in plan["job_id"]
    for path in sorted((args.run / "train_data").glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)["samples"]
        originals = {
            s["index"]: s
            for s in torch.load(
                args.run / "rollout_data" / (path.stem.split("_")[0] + ".pt"), map_location="cpu", weights_only=False
            )["samples"]
        }
        for s in data:
            original = originals[s["sample_indices"]]
            assert s["tokens"] == original["tokens"] and s["loss_masks"] == original["loss_mask"]
            assert torch.equal(
                torch.tensor(s["rollout_log_probs"], dtype=torch.float32),
                torch.tensor(original["rollout_log_probs"], dtype=torch.float32),
            )
            n = s["response_lengths"]
            assert n == original["response_length"]
            assert n == len(s["loss_masks"]) == len(s["log_probs"]) == len(s["rollout_log_probs"])
            assert len(s["tokens"]) == s["total_lengths"]
            lp = torch.tensor(s["log_probs"], dtype=torch.float64)
            q = torch.tensor(s["rollout_log_probs"], dtype=torch.float64)
            valid = torch.tensor(s["loss_masks"], dtype=torch.bool)
            if replay:
                ids, offsets = s["rollout_sampling_mask_ids"], s["rollout_sampling_mask_offsets"]
                assert ids == original["rollout_sampling_mask"]["ids"]
                assert offsets == original["rollout_sampling_mask"]["offsets"]
                assert len(offsets) == n + 1 and offsets[0] == 0 and offsets[-1] == len(ids)
                for i, token in enumerate(s["tokens"][-n:]):
                    assert token in ids[offsets[i] : offsets[i + 1]]
            else:
                assert s.get("rollout_sampling_mask_ids") is None
            delta = (lp - q)[valid]
            adv = torch.tensor(s["advantages"], dtype=torch.float64)[valid]
            assert torch.isfinite(delta).all() and torch.isfinite(adv).all()
            nonzero_advantages += int(torch.count_nonzero(adv))
            sample_count += 1
            rows.extend(delta.tolist())
    assert rows
    effective = [
        u for u in updates if u["optimizer_successful"] and u["changed_elements"] > 0 and u["weight_decay"] == 0
    ]
    result = {
        "status": job["status"],
        "optimizer_steps": len(updates),
        "sample_count": sample_count,
        "nonzero_advantage_tokens": nonzero_advantages,
        "effective_parameter_updates": len(effective),
        "abs_logprob_diff": distribution([abs(x) for x in rows]),
        "ratio": distribution([math.exp(x) for x in rows]),
        "metrics": metrics,
    }
    (args.run / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    assert job["status"] == "SUCCEEDED", job["status"]
    assert effective, "No actual parameter update with zero weight decay; this is not an effective smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("root", "model", "data", "smoke"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("preflight")
    for name in ("root", "model", "run"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--dashboard", required=True)
    p.add_argument("--job", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--tensorboard", type=Path, required=True)
    p.add_argument("--dashboard", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "preflight": preflight, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
