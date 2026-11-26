# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare or explicitly submit one isolated eight-GPU MoE smoke.

Never auto-retry.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

from scripts.tools.sampling_replay_ab import digest, fetch
from scripts.tools.sampling_replay_isolated import MODES, checked_state


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def model_inventory(model):
    """Read headers only: no CUDA, tensor loading, conversion or model
    writes."""
    inventory = {}
    count = 0
    payload = 0
    for path in sorted(model.glob("*.safetensors")):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header_bytes = stream.read(length)
        header = json.loads(header_bytes)
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] != "BF16":
                raise ValueError("Smoke requires the existing BF16 model")
            count += math.prod(tensor["shape"])
            payload += tensor["data_offsets"][1] - tensor["data_offsets"][0]
        inventory[path.name] = dict(
            bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,
            header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        )
    if not inventory:
        raise ValueError("No safetensors shards")
    return dict(parameters=count, bf16_payload_bytes=payload, shards=inventory)


def prepare(args):
    from transformers import AutoTokenizer

    model, data, smoke = args.model.resolve(), args.data.resolve(), args.smoke_data.resolve()
    root = args.root.resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (model, data.parent)):
        raise ValueError("Outputs must not overlap shared inputs")
    source = [json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    selected = [json.loads(x) for x in smoke.read_text().splitlines() if x.strip()]
    if len(selected) != 4 or len({json.dumps(x["prompt"], sort_keys=True) for x in selected}) != 4:
        raise ValueError("Provide exactly four distinct unmodified source prompts")
    indices = [source.index(row) for row in selected]
    config = read(model / "config.json")
    if (
        config.get("model_type"),
        config.get("num_hidden_layers"),
        config.get("num_experts"),
        config.get("num_experts_per_tok"),
    ) != ("qwen3_moe", 48, 128, 8):
        raise ValueError("This smoke is for Qwen3-30B-A3B")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    lengths = [
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                ),
                add_special_tokens=False,
            )
        )
        for x in selected
    ]
    if max(lengths) > 1024:
        raise ValueError("Smoke prompt exceeds the frozen 1024-token budget")
    inventory = model_inventory(model)
    root.mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    (root / "data/train.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected))
    n = inventory["parameters"]
    write(
        root / "manifest.json",
        dict(
            model=str(model),
            source=str(data),
            source_sha256=digest(data),
            source_rows=len(source),
            source_indices=indices,
            prompt_lengths=lengths,
            train_sha256=digest(root / "data/train.jsonl"),
            model_inventory=inventory,
            model_metadata_sha256={
                p.name: digest(p) for p in model.iterdir() if p.suffix in (".json", ".txt", ".jinja")
            },
            budget=dict(
                smoke_checkpoint_bytes=0,
                smoke_optimizer_disk_bytes=0,
                smoke_export_bytes=0,
                weights_payload_bytes=n * 2,
                fp32_master_and_adam_moments_bytes=n * 12,
                full_optimizer_checkpoint_payload_bytes=n * 14,
                live_weights_grad_master_adam_bytes=n * 18,
                note="Logical BF16 weights + FP32 gradient/master/m/v. Actual DCP includes padding, replication and metadata.",
            ),
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        ),
    )
    print(json.dumps(read(root / "manifest.json")["budget"], indent=2))


def training_argv(root, mode):
    if mode not in MODES:
        raise ValueError("Unknown smoke method")
    manifest = read(root / "manifest.json")
    raw = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
            "model-config",
            str(REPO / "scripts/models/qwen3-30B-A3B.sh"),
        ]
    )
    model_args = raw.decode().rstrip("\x00").split("\x00")
    run = root / f"smoke-{mode}"
    options = {
        "--resource": '{"actor":[1,8],"rollout":[1,8]}',
        "--max-staleness": "0",
        "--num-data-storage-units": "1",
        "--hf-checkpoint": manifest["model"],
        "--load": manifest["model"],
        "--megatron-to-hf-mode": "bridge",
        "--tensor-model-parallel-size": "4",
        "--pipeline-model-parallel-size": "1",
        "--context-parallel-size": "2",
        "--expert-model-parallel-size": "8",
        "--expert-tensor-parallel-size": "1",
        "--moe-token-dispatcher-type": "flex",
        "--moe-flex-dispatcher-backend": "deepep",
        "--moe-router-load-balancing-type": "none",
        "--prompt-data": str(root / "data/train.jsonl"),
        "--input-key": "prompt",
        "--label-key": "label",
        "--apply-chat-template-kwargs": '{"enable_thinking":false}',
        "--rm-type": "dapo",
        "--reward-key": "score",
        "--num-rollout": "1",
        "--rollout-batch-size": "4",
        "--n-samples-per-prompt": "8",
        "--global-batch-size": "32",
        "--micro-batch-size": "1",
        "--rollout-max-response-len": "4096",
        "--rollout-max-context-len": "8192",
        "--rollout-temperature": "1",
        "--rollout-top-p": "0.95",
        "--rollout-top-k": "64",
        "--rollout-seed": "1234",
        "--seed": "1234",
        "--advantage-estimator": "grpo",
        "--eps-clip": "0.2",
        "--eps-clip-high": "0.28",
        "--kl-loss-coef": "0",
        "--entropy-coef": "0",
        "--clip-grad": "1",
        "--optimizer": "adam",
        "--lr": "1e-6",
        "--lr-decay-style": "constant",
        "--weight-decay": "0",
        "--adam-beta1": "0.9",
        "--adam-beta2": "0.999",
        "--max-tokens-per-gpu": "8192",
        "--log-probs-max-tokens-per-gpu": "8192",
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": "1",
        "--rollout-num-gpus-per-engine": "8",
        "--sglang-sampling-backend": "pytorch",
        "--sglang-random-seed": "1234",
        "--sglang-mem-fraction-static": "0.5",
        "--sglang-context-length": "8256",
        "--sglang-max-running-requests": "8",
        "--attention-dropout": "0",
        "--hidden-dropout": "0",
        "--attention-backend": "flash",
        "--custom-megatron-before-train-step-hook-path": "scripts.training.text.sampling_replay_moe_hooks.before_train",
        "--dump-details": str(run),
        "--tensorboard-dir": str(run / "tensorboard_log"),
        "--tb-project-name": "sampling-replay-moe",
        "--tb-experiment-name": f"smoke-{mode}",
    }
    flags = [
        "--colocate",
        "--sequence-parallel",
        "--apply-chat-template",
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
        "--calculate-per-token-loss",
        "--use-dynamic-batch-size",
        "--sglang-enable-fp32-lm-head",
        "--sglang-disable-cuda-graph",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ]
    argv = [
        "python",
        "-m",
        "relax.entrypoints.train",
        *model_args,
        *flags,
        *[item for key, value in options.items() for item in (key, str(value))],
    ]
    if mode == "on":
        argv.append("--use-rollout-sampling-replay")
    elif mode == "r3":
        argv.extend(["--use-rollout-routing-replay", "--use-slime-router"])
    elif mode == "tis":
        argv.extend(["--use-tis", "--tis-clip-low", "0", "--tis-clip", "2"])
    elif mode == "sc":
        argv[argv.index("--advantage-estimator") + 1] = "score_centering"
        argv.extend(
            [
                "--rollout-top-logprobs-num",
                "128",
                "--score-centering-is",
                "none",
                "--kl-coef",
                "0",
                "--use-rollout-logprobs",
                "--get-mismatch-metrics",
            ]
        )
    return argv


def guard_cluster(root, instance, mode):
    """Check ownership before any HTTP/Ray interaction or output creation."""
    state = checked_state(instance, inside=True)
    if os.readlink("/proc/self/ns/net") != state["netns"]:
        raise ValueError("Submit only through isolated.py exec into this private namespace")
    config = read(instance / "config.json")
    if config["mode"] != mode:
        raise ValueError("Instance mode does not match submission")
    if (root / "active.json").exists():
        raise ValueError("Previous smoke requires terminal snapshot and successful acceptance before next submission")
    if config.get("gpus") != "0,1,2,3,4,5,6,7":
        raise ValueError("MoE smoke requires its own eight-GPU namespace")
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise ValueError("Smoke needs >=10GiB free; checkpoints/exports are disabled")
    if (root / f"smoke-{mode}").exists():
        raise FileExistsError("Preserve existing results; no overwrite or automatic retry")
    return config


def job_runtime_env(root):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "MASTER_ADDR",
            "NCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME",
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME",
        )
    }
    env["PYTHONPATH"] = str(REPO) + ":" + env.get("PYTHONPATH", "")
    env.update(
        RAY_OVERRIDE_JOB_RUNTIME_ENV="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(root / "cache/hf"),
        XDG_CACHE_HOME=str(root / "cache/xdg"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/torch_extensions"),
        SGLANG_CACHE_DIR=str(root / "cache/sglang"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
    )
    return dict(env_vars=env)


def submit(args, *, argv=None, extra_env=None):
    import fcntl

    with (args.root / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = guard_cluster(args.root, args.instance, args.mode)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise ValueError("Smoke requires a clean committed source checkout")
        from scripts.tools.sampling_replay_isolated import check_gpus_idle

        check_gpus_idle(config["gpus"])
        m = read(args.root / "manifest.json")
        if (
            digest(Path(m["source"])) != m["source_sha256"]
            or digest(args.root / "data/train.jsonl") != m["train_sha256"]
        ):
            raise ValueError("Frozen data changed")
        if model_inventory(Path(m["model"])) != m["model_inventory"] or any(
            digest(Path(m["model"]) / p) != h for p, h in m["model_metadata_sha256"].items()
        ):
            raise ValueError("Frozen model changed")
        dashboard = "http://127.0.0.1:8265"
        if (
            any(j["status"] in ("PENDING", "RUNNING") for j in fetch(dashboard, "/api/jobs/"))
            or fetch(dashboard, "/api/serve/applications/")["applications"]
        ):
            raise ValueError("Private cluster is not empty; do not clean services")
        import ray

        ray.init(address="auto", logging_level="ERROR")
        try:
            available = ray.available_resources()
            if available.get("GPU", 0) != 8:
                raise ValueError(f"Need 8 available GPUs: {available}")
        finally:
            ray.shutdown()
        run = args.root / f"smoke-{args.mode}"
        run.mkdir()
        job = f"sr-moe-smoke-{args.mode}-s1234-{hashlib.sha256(str(args.root.resolve()).encode()).hexdigest()[:8]}"
        command = [
            "ray",
            "job",
            "submit",
            "--no-wait",
            "--address",
            dashboard,
            "--submission-id",
            job,
            "--runtime-env-json",
            json.dumps({"env_vars": {**job_runtime_env(args.root)["env_vars"], **(extra_env or {})}}),
            "--",
            *(argv if argv is not None else training_argv(args.root, args.mode)),
        ]
        write(run / "command.json", command)
        write(
            run / "preflight.json",
            dict(
                job_id=job,
                available=available,
                namespace=checked_state(args.instance, inside=True),
                config=config,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                manifest_sha256=digest(args.root / "manifest.json"),
            ),
        )
        write(args.root / "active.json", dict(mode=args.mode, job_id=job, instance=str(args.instance.resolve())))
        with (run / "submit.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Submitted only {job}; inspect {run}/submit.log")


def validate_samples(samples, originals, replay, *, cp_size=1, cp_rank=0):
    import torch

    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    byid = {o["index"]: o for o in originals}
    if (
        len(originals) != 32
        or len(byid) != 32
        or {s["sample_indices"] for s in samples} != set(byid)
        or len(samples) != 32
    ):
        raise ValueError("Incomplete/duplicate smoke samples")
    nonzero = 0
    for s in samples:
        o = byid[s["sample_indices"]]
        n = s["response_lengths"]
        if (
            s["tokens"] != o["tokens"]
            or s["loss_masks"] != o["loss_mask"]
            or n != o["response_length"]
            or n != len(s["loss_masks"])
            or n != len(o["rollout_log_probs"])
        ):
            raise ValueError("Token/response/loss-mask alignment changed")
        # Static CP slices token probabilities at ingestion; tokens/masks/CSR
        # remain full-response fields in the debug dump. Never compare a local
        # probability vector with the unsliced rollout or index it by full CSR rows.
        full_q = torch.tensor(o["rollout_log_probs"], dtype=torch.float32)
        expected_q = slice_log_prob_with_cp(
            full_q,
            len(o["tokens"]),
            n,
            qkv_format="thd",
            padded_total_length=s.get("padded_total_lengths"),
            dynamic_cp_size=cp_size,
            dynamic_cp_rank=cp_rank,
        )
        q = torch.tensor(s["rollout_log_probs"], dtype=torch.float32)
        adv = torch.tensor(s["advantages"], dtype=torch.float32)
        if (
            not torch.equal(q, expected_q)
            or adv.shape != q.shape
            or not torch.isfinite(full_q).all()
            or not torch.isfinite(q).all()
            or not torch.isfinite(adv).all()
        ):
            raise ValueError("Behavior transfer changed or non-finite advantages")
        nonzero += int(torch.count_nonzero(adv))
        if replay:
            ids = s["rollout_sampling_mask_ids"]
            offsets = s["rollout_sampling_mask_offsets"]
            if (
                ids != o["rollout_sampling_mask"]["ids"]
                or offsets != o["rollout_sampling_mask"]["offsets"]
                or len(offsets) != n + 1
                or offsets[0] != 0
                or offsets[-1] != len(ids)
            ):
                raise ValueError("CSR changed across transfer")
            for i, t in enumerate(s["tokens"][-n:]):
                row = ids[offsets[i] : offsets[i + 1]]
                if not 1 <= len(row) <= 64 or len(set(row)) != len(row) or t not in row:
                    raise ValueError("Invalid sampling support")
                if len(row) == 1 and full_q[i] != 0:
                    raise ValueError("Singleton behavior must be zero")
        elif s.get("rollout_sampling_mask_ids") is not None:
            raise ValueError("OFF unexpectedly contains support")
    return nonzero


def validate_reference_payload(samples, originals, mode, cp_rank):
    import torch

    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    byid = {o["index"]: o for o in originals}
    complete = 0
    for sample in samples:
        o = byid[sample["sample_indices"]]
        n = o["response_length"]
        if mode == "r3":
            routes = torch.as_tensor(o["rollout_routed_experts"])
            if (
                routes.dtype != torch.int32
                or routes.shape != (len(o["tokens"]) - 1, 48, 8)
                or not ((routes >= 0) & (routes < 128)).all()
            ):
                raise ValueError("Invalid native routed expert capture")
            ordered = routes.sort(-1).values
            if (ordered[..., 1:] == ordered[..., :-1]).any():
                raise ValueError("Duplicate experts in a native top8 route")
        elif mode == "sc":
            ids = torch.as_tensor(o["rollout_topk_token_ids"], dtype=torch.int64)
            q = torch.as_tensor(o["rollout_topk_log_probs"], dtype=torch.float32)
            coverage = torch.as_tensor(o["rollout_topk_support_complete"], dtype=torch.bool)
            if ids.shape != (n, 128) or q.shape != ids.shape or coverage.shape != (n,):
                raise ValueError("Invalid SC candidate dimensions")
            valid = ids >= 0
            if (
                not (ids[~valid] == -1).all()
                or not torch.isneginf(q[~valid]).all()
                or not torch.isfinite(q[valid]).all()
                or (q[valid] > 1e-6).any()
                or (ids[valid] >= 151936).any()
            ):
                raise ValueError("Invalid SC candidate IDs or probabilities")
            for i in range(n):
                row = ids[i][valid[i]]
                if row.unique().numel() != row.numel():
                    raise ValueError("Duplicate SC candidates")
                if coverage[i]:
                    loc = (ids[i] == o["tokens"][-n + i]).nonzero().flatten()
                    if (
                        loc.numel() != 1
                        or not torch.isclose(q[i, valid[i]].exp().sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
                        or not torch.isclose(
                            q[i, loc[0]], torch.tensor(o["rollout_log_probs"][i]), atol=1e-6, rtol=1e-5
                        )
                    ):
                        raise ValueError("SC complete support does not match behavior")
            complete += int(coverage.sum())
            for key, full in (
                ("rollout_topk_token_ids", ids),
                ("rollout_topk_log_probs", q),
                ("rollout_topk_support_complete", coverage),
            ):
                expected = slice_log_prob_with_cp(
                    full, len(o["tokens"]), n, "thd", dynamic_cp_size=2, dynamic_cp_rank=cp_rank
                )
                actual = torch.as_tensor(sample[key], dtype=full.dtype)
                if actual.numel() != expected.numel() or not torch.equal(actual.reshape(expected.shape), expected):
                    raise ValueError("SC candidates changed across CP transfer")
    return dict(method=mode, complete_support_rows=complete if mode == "sc" else None)


def metrics_from_job_log(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = re.findall(r"step 0: (\{.*\})", clean)
    if len(rows) != 1:
        raise ValueError("Expected exactly one native train step-0 metrics record")
    values = ast.literal_eval(rows[0])
    if values.get("train/step") != 0 or values.get("train/global_batch_size") != 32:
        raise ValueError("Unexpected native train step/batch budget")
    return {key: [value] for key, value in values.items()}


def analyze(args):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    state = checked_state(args.instance, inside=True)
    if os.readlink("/proc/self/ns/net") != state["netns"]:
        raise ValueError("Analyze through the recorded isolated namespace")
    run = args.root / f"smoke-{args.mode}"
    plan = read(run / "preflight.json")
    dashboard = "http://127.0.0.1:8265"
    job = fetch(dashboard, "/api/jobs/" + plan["job_id"])
    write(run / "job-status.json", job)
    (run / "job.log").write_text(fetch(dashboard, "/api/jobs/" + plan["job_id"] + "/logs")["logs"])
    if job["status"] != "SUCCEEDED":
        raise ValueError("Smoke job has not succeeded")
    originals = torch.load(run / "rollout_data/0.pt", map_location="cpu", weights_only=False)["samples"]
    dumps = list((run / "train_data").glob("*.pt"))
    if {p.name for p in dumps} != {f"0_{rank}.pt" for rank in range(8)}:
        raise ValueError("Expected all eight rank dumps")
    all_nonzero = []
    reference_payload = []
    for p in sorted(dumps):
        data = torch.load(p, map_location="cpu", weights_only=False)
        rank = int(p.stem.split("_")[1])
        if data["rollout_id"] != 0 or data["rank"] != rank:
            raise ValueError("Unexpected rollout budget or duplicated/mislabeled rank dump")
        # Frozen eight-rank TP4/CP2/PP1/DP1 layout: TP is the fastest axis.
        all_nonzero.append(
            validate_samples(data["samples"], originals, args.mode == "on", cp_size=2, cp_rank=rank // 4)
        )
        if args.mode in ("r3", "sc"):
            reference_payload.append(validate_reference_payload(data["samples"], originals, args.mode, rank // 4))
    updates = []
    for rank in range(8):
        rows = [json.loads(x) for x in (run / f"parameter-updates-rank{rank}.jsonl").read_text().splitlines()]
        if len(rows) != 1:
            raise ValueError("Incomplete/failed optimizer update")
        row = rows[0]
        if (
            row["rank"] != rank
            or row["optimizer_step"] != 0
            or not row["optimizer_successful"]
            or row["weight_decay"] != 0
            or not row["finite"]
            or not row["parameter_finite_after"]
            or any(
                row[key] is None or not math.isfinite(row[key])
                for key in ("optimizer_grad_norm", "sampled_gradient_norm", "delta_l2", "max_abs_delta")
            )
        ):
            raise ValueError("Invalid rank/step or non-finite/failed optimizer update")
        if args.mode == "r3" and not row.get("routing_replay", {}).get("fully_consumed", False):
            raise ValueError("R3 routing consumption was not verified")
        updates.extend(rows)
    if (run / "tensorboard_log").is_dir():
        tb = EventAccumulator(str(run / "tensorboard_log"), size_guidance={"scalars": 0})
        tb.Reload()
        metrics = {k: [e.value for e in tb.Scalars(k)] for k in tb.Tags()["scalars"]}
        metrics_source = "tensorboard_log"
    else:
        metrics = metrics_from_job_log((run / "job.log").read_text())
        metrics_source = "job.log native train step 0"
    if not metrics or any(not math.isfinite(x) for v in metrics.values() for x in v):
        raise ValueError("Missing/non-finite metrics")
    if args.mode == "tis" and any(key not in metrics for key in ("train/tis", "train/tis_abs", "train/tis_clipfrac")):
        raise ValueError("Native TIS weight metrics are missing")
    if args.mode == "sc" and any(
        key not in metrics for key in ("train/sc_correction", "train/sc_support_aligned", "train/sc_importance_weight")
    ):
        raise ValueError("Native SC objective metrics are missing")
    # Some versions place TB under the default dump directory; require actual loss evidence.
    if not metrics.get("train/loss") or not any(x > 0 for x in metrics.get("train/grad_norm", [])):
        raise ValueError("No finite training loss/nonzero global gradient evidence")
    if not any(x > 0 for x in all_nonzero) or not any(
        u["changed_elements"] > 0 and u["sampled_gradient_nonzero"] > 0 for u in updates
    ):
        raise ValueError("No effective learning signal: retain evidence, do not retry automatically")
    write(
        run / "acceptance.json",
        dict(
            status="passed",
            samples=32,
            rank_dumps=8,
            nonzero_advantage_tokens_by_rank=all_nonzero,
            updates=updates,
            metrics=metrics,
            metrics_source=metrics_source,
            reference_payload=reference_payload,
            note="Observed expert slices only; CP/TP math has separate reference tests. No additional diagnostic forward.",
        ),
    )
    active = read(args.root / "active.json")
    if active["job_id"] != plan["job_id"]:
        raise ValueError("Active job ownership changed")
    (args.root / "active.json").unlink()
    print(json.dumps(dict(status="passed", samples=32, rank_dumps=8)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "argv", "submit", "analyze"):
        c = sub.add_parser(command)
        c.add_argument("--root", type=Path, required=True)
        if command == "prepare":
            for flag in ("model", "data", "smoke-data"):
                c.add_argument("--" + flag, type=Path, required=True)
        else:
            c.add_argument("--mode", choices=MODES, required=True)
        if command in ("submit", "analyze"):
            c.add_argument("--instance", type=Path, required=True)
    a = p.parse_args()
    if a.command == "argv":
        print(json.dumps(training_argv(a.root, a.mode), indent=2))
    else:
        {"prepare": prepare, "submit": submit, "analyze": analyze}[a.command](a)


if __name__ == "__main__":
    main()
