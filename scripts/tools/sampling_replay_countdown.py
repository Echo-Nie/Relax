# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-arm Countdown experiments.

No batch submission or shared cleanup.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.tools import sampling_replay_moe as base
from scripts.tools.sampling_replay_ab import fetch
from scripts.tools.sampling_replay_isolated import check_gpus_idle, checked_state


REPO = Path(__file__).resolve().parents[2]
ARMS = ("off", "on", "tis", "sc", "sc-tis")
PROJECT = "sampling-replay-countdown-fp8"
GENERATOR = "scripts.training.text.sampling_replay_countdown_hooks.generate_bounded"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(a):
    from importlib import import_module

    from transformers import AutoTokenizer

    model = a.model.resolve()
    data = a.data.resolve()
    root = a.root.resolve()
    if root.is_relative_to(model) or root.is_relative_to(data):
        raise ValueError("Outputs must be separate from inputs")
    source = base.read(data / "manifest.json")
    t = AutoTokenizer.from_pretrained(model, local_files_only=True)
    all_keys = []
    summary = {}
    root.mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    for split, expected in (("train", 19936), ("dev", 64)):
        path = data / f"countdown-{split}.jsonl"
        values = rows(path)
        if len(values) != expected or base.digest(path) != source["files"][path.name]["sha256"]:
            raise ValueError("Countdown source manifest mismatch")
        lengths = []
        keys = []
        recipe = import_module("examples.algorithms.score-centering.recipe")
        for row in values:
            label = json.loads(row["label"])
            if row["prompt"] != recipe.messages(label):
                raise ValueError("Prompt differs from native Countdown recipe")
            tokens = t.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
            )
            if 512 - len(tokens) != row["max_response_tokens"] or len(tokens) >= 512:
                raise ValueError("512 total-token budget mismatch")
            if (
                recipe.reward(
                    None, type("S", (), dict(label=row["label"], response=f"<answer>{label['solution']}</answer>"))()
                )
                != 1
            ):
                raise ValueError("Native parser rejects source solution")
            lengths.append(len(tokens))
            keys.append((tuple(sorted(label["numbers"])), label["target"]))
        all_keys.append(set(keys))
        summary[split] = dict(rows=len(values), prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths))
        shutil.copyfile(path, root / "data" / path.name)
    if all_keys[0] & all_keys[1]:
        raise ValueError("Train/dev Countdown question overlap")
    (root / "data/dev-smoke.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows(root / "data/countdown-dev.jsonl")[:2])
    )
    base.write(
        root / "plan.json",
        dict(
            model=str(model),
            fp8=str(a.fp8.resolve()),
            model_sha256={
                p.name: base.digest(p)
                for p in model.iterdir()
                if p.is_file() and p.suffix in (".json", ".safetensors", ".txt")
            },
            data_sha256={p.name: base.digest(p) for p in (root / "data").glob("*.jsonl")},
            summary=summary,
            seed=0,
            updates=100,
            gpus="0,1,2,3,4,5,6,7",
            batch=512,
            dev_samples=512,
            project=PROJECT,
            total_tokens=512,
        ),
    )
    print(json.dumps(summary))


def validate(root, quantized=False):
    plan = base.read(root / "plan.json")
    for name, digest in plan["model_sha256"].items():
        if base.digest(Path(plan["model"]) / name) != digest:
            raise ValueError("BF16 input changed")
    for name, digest in plan["data_sha256"].items():
        if base.digest(root / "data" / name) != digest:
            raise ValueError("Frozen data changed")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise ValueError("Require 20GiB free")
    if quantized:
        seal = base.read(root / "fp8-seal.json")
        if seal["source_sha256"] != plan["model_sha256"]["model.safetensors"]:
            raise ValueError("FP8 source differs")
        for name, digest in seal["files"].items():
            if base.digest(Path(plan["fp8"]) / name) != digest:
                raise ValueError("FP8 checkpoint changed")
    return plan


def convert(a):
    plan = validate(a.root)
    target = Path(plan["fp8"])
    if target.exists():
        raise FileExistsError("Refuse to overwrite FP8 output")
    check_gpus_idle(str(a.gpu))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/tools/convert_hf_to_fp8.py"),
            "--model-dir",
            plan["model"],
            "--save-dir",
            str(target),
            "--strategy",
            "block",
            "--block-size",
            "128",
            "128",
            "--max-workers",
            "1",
        ],
        env=env,
        check=True,
    )
    config = base.read(target / "config.json")
    if config["quantization_config"]["quant_method"] != "fp8" or config["quantization_config"][
        "weight_block_size"
    ] != [128, 128]:
        raise ValueError("Unexpected FP8 format")
    base.write(
        a.root / "fp8-seal.json",
        dict(
            source_sha256=plan["model_sha256"]["model.safetensors"],
            files={p.name: base.digest(p) for p in target.iterdir() if p.is_file()},
            config=config["quantization_config"],
        ),
    )
    print("FP8_CONVERSION_SEALED", target)


def run_dir(a):
    return a.root / a.stage / a.mode


def argv(a):
    p = base.read(a.root / "plan.json")
    run = run_dir(a)
    raw = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
            "model",
            str(REPO / "scripts/models/qwen3-0.6B.sh"),
        ]
    )
    model = raw.decode().rstrip("\0").split("\0")
    opts = {
        "--resource": '{"actor":[1,8],"rollout":[1,8]}',
        "--hf-checkpoint": p["model"],
        "--load": p["model"],
        "--ref-load": p["model"],
        "--sglang-hf-checkpoint": p["fp8"],
        "--megatron-to-hf-mode": "bridge",
        "--max-staleness": 0,
        "--num-data-storage-units": 1,
        "--prompt-data": a.root / "data/countdown-train.jsonl",
        "--input-key": "prompt",
        "--label-key": "label",
        "--apply-chat-template-kwargs": '{"enable_thinking":false}',
        "--custom-rm-path": "examples.algorithms.score-centering.recipe.signed_reward",
        "--custom-generate-function-path": GENERATOR,
        "--num-rollout": 100 if a.stage == "formal" else 2,
        "--rollout-batch-size": 64,
        "--n-samples-per-prompt": 8,
        "--global-batch-size": 512,
        "--micro-batch-size": 1,
        "--rollout-seed": 0,
        "--seed": 0,
        "--rollout-max-response-len": 512,
        "--rollout-max-context-len": 512,
        "--rollout-temperature": 1,
        "--rollout-top-p": 0.95,
        "--rollout-top-k": 64,
        "--eval-interval": 10 if a.stage == "formal" else 2,
        "--n-samples-per-eval-prompt": 8,
        "--eval-temperature": 1,
        "--eval-top-p": 0.95,
        "--eval-top-k": 64,
        "--eval-max-response-len": 512,
        "--optimizer": "sgd",
        "--sgd-momentum": 0,
        "--lr": 0.01,
        "--lr-decay-style": "constant",
        "--weight-decay": 0,
        "--clip-grad": 0,
        "--kl-coef": 0,
        "--entropy-coef": 0,
        "--advantage-estimator": "score_centering" if a.mode.startswith("sc") else "rloo",
        "--tensor-model-parallel-size": 1,
        "--pipeline-model-parallel-size": 1,
        "--context-parallel-size": 1,
        "--expert-model-parallel-size": 1,
        "--expert-tensor-parallel-size": 1,
        "--max-tokens-per-gpu": 4096,
        "--log-probs-max-tokens-per-gpu": 4096,
        "--log-probs-chunk-size": 4,
        "--update-weight-buffer-size": 134217728,
        "--train-memory-margin-bytes": 536870912,
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": 1,
        "--rollout-num-gpus-per-engine": 1,
        "--sglang-mem-fraction-static": 0.3,
        "--sglang-context-length": 512,
        "--sglang-max-running-requests": 64,
        "--sglang-random-seed": 0,
        "--sglang-sampling-backend": "pytorch",
        "--sglang-fp8-gemm-backend": "triton",
        "--sglang-kv-cache-dtype": "fp8_e4m3",
        "--attention-dropout": 0,
        "--hidden-dropout": 0,
        "--attention-backend": "flash",
        "--reward-num-workers": 4,
        "--reward-max-concurrency": 8,
        "--tb-project-name": PROJECT,
        "--tb-experiment-name": f"{a.root.name}-{a.stage}-{a.mode}-s0",
        "--rollout-result-dir": run / "results",
        "--tensorboard-dir": run / "tensorboard_log",
        "--custom-megatron-before-train-step-hook-path": "scripts.training.text.sampling_replay_countdown_hooks.before_train",
        "--custom-rollout-log-function-path": "scripts.training.text.sampling_replay_countdown_hooks.rollout_log",
        "--custom-eval-rollout-log-function-path": "scripts.training.text.sampling_replay_countdown_hooks.eval_log",
    }
    flags = [
        "--colocate",
        "--bf16",
        "--apply-chat-template",
        "--balance-data",
        "--calculate-per-token-loss",
        "--use-dynamic-batch-size",
        "--recompute-loss-function",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--use-clearml",
        "--use-metrics-service",
        "--log-passrate",
    ]
    result = [
        "python",
        "-m",
        "relax.entrypoints.train",
        *model,
        *flags,
        *[str(v) for k, value in opts.items() for v in (k, value)],
        "--eval-prompt-data",
        "countdown-dev",
        str(a.root / "data" / ("countdown-dev.jsonl" if a.stage == "formal" else "dev-smoke.jsonl")),
    ]
    if a.mode == "on":
        result.append("--use-rollout-sampling-replay")
    if a.mode == "tis":
        result.extend(["--use-tis", "--tis-clip-low", "0", "--tis-clip", "2"])
    if a.mode.startswith("sc"):
        result.extend(
            [
                "--rollout-top-logprobs-num",
                "128",
                "--use-rollout-logprobs",
                "--get-mismatch-metrics",
                "--score-centering-is",
                "tis" if a.mode == "sc-tis" else "none",
            ]
        )
        if a.mode == "sc-tis":
            result.extend(["--score-centering-tis-clip", "2"])
    return result


def tracking_env(root, run):
    env = base.job_runtime_env(root)["env_vars"]
    proxy = "http://127.0.0.1:18080"
    bypass = "127.0.0.0/8,localhost,127.0.0.1," + os.environ.get("MASTER_ADDR", "127.0.0.1")
    env.update(
        HTTPS_PROXY=proxy,
        HTTP_PROXY=proxy,
        https_proxy=proxy,
        http_proxy=proxy,
        NO_PROXY=bypass,
        no_proxy=bypass,
        TENSORBOARD_DIR=str(run / "tensorboard_log"),
        SGLANG_RETURN_ORIGINAL_LOGPROB="0",
        PYTHONUNBUFFERED="1",
        CLEARML_CONFIG_FILE=str(Path.home() / "clearml.conf"),
    )
    env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
        [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "NO_PROXY",
            "no_proxy",
            "TENSORBOARD_DIR",
            "CLEARML_CONFIG_FILE",
            "SGLANG_RETURN_ORIGINAL_LOGPROB",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "XDG_CACHE_HOME",
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "SGLANG_CACHE_DIR",
        ]
    )
    return env


def submit(a):
    import fcntl

    from scripts.tools.sampling_replay_isolated import checked_state

    with (a.root / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_state(a.instance, inside=True)
        config = base.read(a.instance / "config.json")
        if config["mode"] != a.mode or config["gpus"] != "0,1,2,3,4,5,6,7":
            raise ValueError("Wrong private instance")
        validate(a.root, True)
        check_gpus_idle(config["gpus"])
        if (a.root / "active.json").exists():
            raise ValueError("Previous arm needs terminal analysis before next submission")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise ValueError("Require clean committed source")
        if (
            any(j["status"] in ("PENDING", "RUNNING") for j in fetch("http://127.0.0.1:8265", "/api/jobs/"))
            or fetch("http://127.0.0.1:8265", "/api/serve/applications/")["applications"]
        ):
            raise ValueError("Private Ray is not empty; no cleanup")
        run = run_dir(a)
        run.mkdir(parents=True, exist_ok=False)
        job = f"sr-countdown-{a.stage}-{a.mode}-s0-{hashlib.sha256(str(a.root).encode()).hexdigest()[:8]}"
        cmd = [
            "ray",
            "job",
            "submit",
            "--no-wait",
            "--address",
            "http://127.0.0.1:8265",
            "--submission-id",
            job,
            "--runtime-env-json",
            json.dumps(dict(env_vars=tracking_env(a.root, run))),
            "--",
            *argv(a),
        ]
        base.write(
            run / "preflight.json",
            dict(
                job_id=job,
                git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                instance=str(a.instance),
                stage=a.stage,
                mode=a.mode,
            ),
        )
        base.write(run / "command.json", cmd)
        base.write(a.root / "active.json", dict(job_id=job, run=str(run)))
        with (run / "submit.log").open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=REPO)
        print("SUBMITTED", job, flush=True)


def snapshot(a):
    checked_state(a.instance, inside=True)
    run = run_dir(a)
    job = base.read(run / "preflight.json")["job_id"]
    state = fetch("http://127.0.0.1:8265", f"/api/jobs/{job}")
    base.write(run / "job-status.json", state)
    (run / "job.log").write_text(fetch("http://127.0.0.1:8265", f"/api/jobs/{job}/logs")["logs"])
    return state


def analyze(a):
    checked_state(a.instance, inside=True)
    run = run_dir(a)
    state = base.read(run / "job-status.json")
    train = rows(run / "results/audit-train.jsonl") if (run / "results/audit-train.jsonl").exists() else []
    evaluations = rows(run / "results/audit-eval.jsonl") if (run / "results/audit-eval.jsonl").exists() else []
    updates = {
        r: rows(run / f"parameter-updates-rank{r}.jsonl")
        if (run / f"parameter-updates-rank{r}.jsonl").exists()
        else []
        for r in range(8)
    }
    stop = base.read(run / "results/stop.json") if (run / "results/stop.json").exists() else None
    text = (run / "job.log").read_text()
    budget = 100 if a.stage == "formal" else 2
    if stop:
        kind = stop["kind"]
    elif "CUDA out of memory" in text:
        kind = "system_oom"
    elif any(not r["finite"] or not r["optimizer_successful"] for records in updates.values() for r in records):
        kind = "numeric_failure"
    elif state["status"] == "SUCCEEDED":
        kind = "budget_completed"
    else:
        kind = "system_or_contract_failure"
    result = dict(
        kind=kind,
        job_status=state["status"],
        updates_by_rank={r: len(v) for r, v in updates.items()},
        train=train,
        evaluations=evaluations,
        stop=stop,
    )
    base.write(run / "analysis.json", result)
    if kind == "budget_completed":
        expected = [0, *range(10 if a.stage == "formal" else 2, budget + 1, 10 if a.stage == "formal" else 2)]
        if (
            len(train) != budget
            or any(len(v) != budget for v in updates.values())
            or [e["completed_updates"] for e in evaluations] != expected
        ):
            raise ValueError("Missing updates/evals; Job SUCCEEDED is insufficient")
        for row in train:
            if len(rows(run / f"results/train/{row['completed_updates_before_batch']}.jsonl")) != 512:
                raise ValueError("Incomplete training responses")
        for row in evaluations:
            if len(rows(run / f"results/eval/{row['native_rollout_id']}.jsonl")) != row["samples"]:
                raise ValueError("Incomplete eval file")
        os.environ.update(tracking_env(a.root, run))  # before importing SDK: no proxy bootstrap retry
        print("Local validation passed; reading ClearML", flush=True)
        from clearml import Task

        ids = set(re.findall(r"ClearML Task: created new task id=([a-f0-9]{32})", text))
        if len(ids) != 1:
            raise ValueError("Missing/ambiguous ClearML Task")
        t = Task.get_task(task_id=ids.pop())
        scalars = t.get_reported_scalars()
        series = scalars.get("audit_eval", {}).get("accuracy", {})
        if series.get("x") != expected or series.get("y") != [e["accuracy"] for e in evaluations]:
            raise ValueError("ClearML eval step/value mismatch")
        result.update(
            task_id=t.id,
            url=t.get_output_log_web_page(),
            scalars=scalars,
            allocated_gpu_hours=8 * (state["end_time"] - state["start_time"]) / 3600000,
        )
        base.write(run / "analysis.json", result)
    if kind in ("budget_completed", "quality_collapse", "numeric_failure"):
        active = base.read(a.root / "active.json")
        if active["job_id"] == base.read(run / "preflight.json")["job_id"]:
            (a.root / "active.json").unlink()
    print(json.dumps({k: v for k, v in result.items() if k not in ("train", "evaluations", "scalars")}), flush=True)


def observe(a):
    run = run_dir(a)
    job = base.read(run / "preflight.json")["job_id"]
    with (run / "gpu.csv").open("x") as f:
        f.write("timestamp,index,memory.used,utilization.gpu\n")
        while True:
            f.write(
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
            f.flush()
            state = fetch("http://127.0.0.1:8265", f"/api/jobs/{job}")
            if state["status"] in ("SUCCEEDED", "FAILED", "STOPPED"):
                snapshot(a)
                analyze(a)
                return
            time.sleep(5)


def start(a):
    validate(a.root, True)
    if (a.root / "active.json").exists():
        raise ValueError("An arm is already active or awaits analysis")
    instance = a.root / f"ray-{a.stage}-{a.mode}"
    a.instance = instance
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/tools/sampling_replay_isolated.py"),
            "create",
            "--instance",
            str(instance),
            "--mode",
            a.mode,
            "--gpus",
            "0,1,2,3,4,5,6,7",
            "--host-ip",
            f"127.96.51.{ARMS.index(a.mode) + 1}",
            "--ray-tmpdir",
            str(a.ray_tmpdir),
        ],
        check=True,
    )
    logs = a.root / f"bridge-{a.stage}-{a.mode}.log"
    with logs.open("x") as f:
        proxy = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts/tools/sampling_replay_tracking_proxy.py"),
                "--instance",
                str(instance),
            ],
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    prefix = [
        sys.executable,
        str(REPO / "scripts/tools/sampling_replay_isolated.py"),
        "exec",
        "--instance",
        str(instance),
        "--",
    ]
    for _ in range(60):
        if proxy.poll() is not None:
            raise RuntimeError(f"Bridge failed: {logs}")
        ready = subprocess.run(
            [*prefix, "curl", "--fail", "--silent", "--max-time", "2", "http://127.0.0.1:8265/api/version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ready.returncode == 0 and "bridge ready" in logs.read_text():
            break
        time.sleep(1)
    else:
        raise TimeoutError("Private cluster not ready; preserve instance")
    subprocess.run(
        [
            *prefix,
            sys.executable,
            str(Path(__file__).resolve()),
            "submit",
            "--root",
            str(a.root),
            "--stage",
            a.stage,
            "--mode",
            a.mode,
            "--instance",
            str(instance),
        ],
        check=True,
        cwd=REPO,
    )
    run = run_dir(a)
    job = base.read(run / "preflight.json")["job_id"]
    processes = {}
    for name, cmd in (
        ("logs", [*prefix, "ray", "job", "logs", "--address", "http://127.0.0.1:8265", "--follow", job]),
        (
            "monitor",
            [
                *prefix,
                sys.executable,
                str(Path(__file__).resolve()),
                "observe",
                "--root",
                str(a.root),
                "--stage",
                a.stage,
                "--mode",
                a.mode,
                "--instance",
                str(instance),
            ],
        ),
    ):
        with (run / f"{name}.log").open("x") as f:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO
            )
            processes[name] = proc.pid
    base.write(run / "background.json", dict(job_id=job, pids=processes, proxy_pid=proxy.pid))
    print("BACKGROUND_SINGLE_ARM", job, "LOG", run / "logs.log", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    for cmd in ("prepare", "convert", "argv", "start", "submit", "snapshot", "observe", "analyze"):
        c = s.add_parser(cmd)
        c.add_argument("--root", required=True, type=Path)
        if cmd == "prepare":
            for arg in ("model", "fp8", "data"):
                c.add_argument("--" + arg, type=Path, required=True)
        elif cmd == "convert":
            c.add_argument("--gpu", type=int, required=True)
        else:
            c.add_argument("--mode", choices=ARMS, required=True)
            c.add_argument("--stage", choices=("formal", "smoke"), default="formal")
            if cmd == "start":
                c.add_argument("--ray-tmpdir", type=Path, required=True)
            elif cmd not in ("argv",):
                c.add_argument("--instance", type=Path, required=True)
    a = p.parse_args()
    a.root = a.root.resolve()
    if a.command == "argv":
        print(json.dumps(argv(a), indent=2))
    else:
        {
            "prepare": prepare,
            "convert": convert,
            "start": start,
            "submit": submit,
            "snapshot": snapshot,
            "observe": observe,
            "analyze": analyze,
        }[a.command](a)


if __name__ == "__main__":
    main()
