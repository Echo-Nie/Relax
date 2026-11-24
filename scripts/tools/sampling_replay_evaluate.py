"""Fixed-checkpoint evaluation through Relax's standard tokenizer, rollout and
DAPO reward.

No Ray, training, weight synchronization or shared-service cleanup. Each run
owns one local SGLang process group. Full evaluation is an explicit user
command.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def prompt_key(row: dict[str, Any]) -> str:
    return json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True)


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    original = read_json(args.base / "manifest.json")
    source = Path(original["source"])
    if sha(source) != original["source_sha256"]:
        raise ValueError("Source dataset checksum changed")
    data = rows(source)
    tokenizer = AutoTokenizer.from_pretrained(original["model"], local_files_only=True)
    train_keys = {prompt_key(data[i]) for i in original["source_indices"]["train"]}
    selected = list(original["source_indices"]["eval"])
    seen = {prompt_key(data[i]) for i in selected}
    if seen & train_keys or len(seen) != len(selected):
        raise ValueError("Original held-out set overlaps training or contains duplicate prompts")
    order = list(range(len(data)))
    random.Random(20261124).shuffle(order)
    for index in order:
        key = prompt_key(data[index])
        if key in train_keys or key in seen:
            continue
        text = tokenizer.apply_chat_template(data[index]["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 1024:
            continue
        selected.append(index)
        seen.add(key)
        if len(selected) == 256:
            break
    if len(selected) != 256:
        raise ValueError("Insufficient independent prompts")
    args.root.mkdir(parents=True, exist_ok=True)
    target = args.root / "eval256.jsonl"
    target.write_text("".join(json.dumps(data[i], ensure_ascii=False) + "\n" for i in selected))
    manifest = dict(
        source=str(source),
        source_sha256=sha(source),
        data=str(target),
        data_sha256=sha(target),
        model=original["model"],
        model_sha256=original["model_sha256"],
        source_indices=selected,
        excluded_training_indices=original["source_indices"]["train"],
        selection_seed=20261124,
        count=256,
        legacy_eval_count=64,
        train_overlap_by_exact_prompt=0,
        overlap_scope="Exact canonical prompt; semantic near-duplicates not audited",
    )
    write_json(args.root / "manifest.json", manifest)
    smoke = args.root / "smoke.jsonl"
    smoke.write_text("".join(json.dumps(data[i], ensure_ascii=False) + "\n" for i in selected[:2]))
    write_json(
        args.root / "smoke-manifest.json",
        dict(manifest, data=str(smoke), data_sha256=sha(smoke), source_indices=selected[:2], count=2),
    )


def compare(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare complete per-question records, rejecting silent
    drops/duplicates."""

    def indexed(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        result = {row["source_index"]: row for row in records}
        if len(result) != len(records):
            raise ValueError("Duplicate question IDs")
        return result

    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError("Different question sets")
    return {
        "n": len(a),
        "input_tokens_changed": [i for i in a if a[i]["input_ids"] != b[i]["input_ids"]],
        "output_tokens_changed": [i for i in a if a[i]["output_ids"] != b[i]["output_ids"]],
        "text_changed": [i for i in a if a[i]["response"] != b[i]["response"]],
        "reward_changed": [i for i in a if a[i]["reward"] != b[i]["reward"]],
        "reward_changed_same_text": [
            i for i in a if a[i]["response"] == b[i]["response"] and a[i]["reward"] != b[i]["reward"]
        ],
        "correct_left": sorted(i for i in a if a[i]["reward"]["acc"]),
        "correct_right": sorted(i for i in b if b[i]["reward"]["acc"]),
    }


def audit(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from relax.engine.rewards.math_dapo_utils import compute_score

    manifest = read_json(args.base / "manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    eval_rows = rows(args.base / "data/eval.jsonl")
    mapping = {
        tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True): i
        for i, row in zip(manifest["source_indices"]["eval"], eval_rows, strict=True)
    }
    records, mismatches = {}, []
    for seed in (1234, 5678):
        for mode in ("on", "off"):
            name = f"{mode}-s{seed}"
            samples = torch.load(
                args.base / f"formal-{name}/rollout_data/eval_0.pt", map_location="cpu", weights_only=False
            )["samples"]
            result = []
            for sample in samples:
                index = mapping[sample["prompt"]]
                n = sample["response_length"]
                prefix = sample["tokens"][:-n] if n else sample["tokens"]
                if prefix != tokenizer.encode(sample["prompt"], add_special_tokens=False):
                    raise ValueError(f"Historical prefix mismatch: {name}/{index}")
                rescored = [compute_score(sample["response"], sample["label"]) for _ in range(3)]
                if any(score != sample["reward"] for score in rescored):
                    mismatches.append([name, index, sample["reward"], rescored])
                result.append(
                    dict(
                        source_index=index,
                        input_ids=prefix,
                        output_ids=sample["tokens"][-n:] if n else [],
                        response=sample["response"],
                        reward=sample["reward"],
                        status=sample["status"],
                        weight_versions=sample["weight_versions"],
                    )
                )
            records[name] = sorted(result, key=lambda row: row["source_index"])
    args.root.mkdir(parents=True, exist_ok=True)
    write_json(args.root / "historical-initial-records.json", records)
    names = list(records)
    write_json(
        args.root / "historical-audit.json",
        dict(
            parser_rechecks=256 * 3,
            parser_mismatches=mismatches,
            prefixes_checked=256,
            comparisons={
                f"{a}_vs_{b}": compare(records[a], records[b]) for i, a in enumerate(names) for b in names[i + 1 :]
            },
        ),
    )


def process_identity(pid: int) -> str | None:
    try:
        # Fields after comm start with state (field 3); starttime is field 22.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except FileNotFoundError:
        return None


def status_or_stop(args: argparse.Namespace) -> None:
    status = read_json(args.output / "status.json")
    alive = process_identity(status["pid"]) == status["starttime"]
    if args.command == "stop" and alive:
        cmdline = Path(f"/proc/{status['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if str(args.output.resolve()) not in cmdline or "sampling_replay_evaluate.py" not in cmdline:
            raise ValueError("PID does not belong to the specified evaluation")
        os.kill(status["pid"], signal.SIGTERM)
    print(json.dumps(dict(status, process_alive=alive), ensure_ascii=False, indent=2))


async def local_dapo(args: argparse.Namespace, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Same pure DAPO scorer, inline async dispatch without shared Ray
    workers."""
    from relax.engine.rewards.math_dapo_utils import compute_score

    return compute_score(sample.response, sample.label)


async def evaluate(args: argparse.Namespace) -> None:
    import httpx

    from relax.engine.rollout.sglang_rollout import eval_rollout_single_dataset
    from relax.utils.arguments import parse_args
    from relax.utils.http_utils import init_http_client

    argv = read_json(args.base / "formal-on-s1234-argv.json")
    argv = [token for token in argv if token != "--use-rollout-sampling-replay"]
    sys.argv = [sys.argv[0], *argv, "--sglang-server-concurrency", str(args.concurrency)]
    config = parse_args()
    config.sglang_router_ip, config.sglang_router_port = "127.0.0.1", args.port
    config.use_distributed_post = False
    if config.rm_type != "dapo" or config.custom_rm_path:
        raise ValueError("This evaluation recipe requires the original built-in DAPO reward")
    config.custom_rm_path = "scripts.tools.sampling_replay_evaluate.local_dapo"
    # Keep the original tokenizer, chat template, standard generation and DAPO scorer.
    dataset = dataclasses.replace(config.eval_datasets[0], name="exp06", path=str(args.data))
    init_http_client(config)
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        (await client.post(f"http://127.0.0.1:{args.port}/flush_cache")).raise_for_status()
        for repeat in range(args.repeats):
            if repeat:
                (await client.post(f"http://127.0.0.1:{args.port}/flush_cache")).raise_for_status()
            started = time.time()
            result = (await eval_rollout_single_dataset(config, 0, dataset))["exp06"]
            samples = result["samples"]
            if len(samples) != len(rows(args.data)) or result["num_valid_prompts"] != len(samples):
                raise ValueError("Incomplete evaluation; refusing partial accuracy")
            manifest = read_json(args.manifest)
            indices = manifest["source_indices"]
            if isinstance(indices, dict):
                indices = indices["eval"]
            records = []
            for sample in samples:
                n = sample.response_length
                records.append(
                    dict(
                        source_index=indices[sample.index],
                        index=sample.index,
                        input_ids=sample.tokens[:-n] if n else sample.tokens,
                        output_ids=sample.tokens[-n:] if n else [],
                        response=sample.response,
                        label=sample.label,
                        reward=sample.reward,
                        status=sample.status.value,
                        prefix_cache_info=dataclasses.asdict(sample.prefix_cache_info),
                        metadata=sample.metadata,
                    )
                )
            target = args.output / f"repeat-{repeat}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            write_json(
                args.output / f"repeat-{repeat}-summary.json",
                dict(
                    n=len(records),
                    accuracy=sum(row["reward"]["acc"] for row in records) / len(records),
                    reward=sum(row["reward"]["score"] for row in records) / len(records),
                    truncated=sum(row["status"] == "truncated" for row in records),
                    seconds=time.time() - started,
                    result_sha256=sha(target),
                ),
            )


def preflight(args: argparse.Namespace) -> None:
    # The per-run port and GPU must be free; never steal or clean shared services.
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory, utilization = map(int, gpu.strip().split(","))
    if memory > 100 or utilization:
        raise ValueError(f"GPU {args.gpu} is busy: {gpu.strip()}")
    with socket.socket() as sock:
        # Match SGLang's listener: TIME_WAIT is reusable, a live listener is not.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))


def run(args: argparse.Namespace) -> None:
    import httpx

    args.output = args.output.resolve()
    manifest = read_json(args.manifest)
    indices = manifest["source_indices"]
    if isinstance(indices, dict):
        indices = indices["eval"]
        expected_sha = manifest["data_sha256"]["eval.jsonl"]
    else:
        expected_sha = manifest["data_sha256"]
    if len(indices) != len(rows(args.data)) or sha(args.data) != expected_sha:
        raise ValueError("Dataset differs from the fixed manifest")
    preflight(args)
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(
        pid=os.getpid(),
        starttime=process_identity(os.getpid()),
        state="RUNNING",
        started=time.time(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        data_sha256=sha(args.data),
        manifest_sha256=sha(args.manifest),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        tool_sha256=sha(__file__),
        model_files_sha256={
            path.name: sha(path)
            for path in sorted(args.model.iterdir())
            if path.suffix in (".safetensors", ".json", ".jinja", ".txt")
        },
    )
    write_json(args.output / "status.json", state)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--random-seed",
        "1234",
        "--sampling-backend",
        "pytorch",
        "--enable-fp32-lm-head",
        "--disable-cuda-graph",
        "--max-running-requests",
        "16",
        "--context-length",
        "2112",
        "--mem-fraction-static",
        "0.2",
    ]
    write_json(args.output / "server-command.json", command)
    server = None

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)),
                start_new_session=True,
            )
            state["server_pid"] = server.pid
            write_json(args.output / "status.json", state)
            with httpx.Client(timeout=2, trust_env=False) as client:
                deadline = time.monotonic() + 420
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Owned SGLang server exited; inspect server.log")
                    try:
                        if client.get(f"http://127.0.0.1:{args.port}/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Owned SGLang startup timed out")
                    time.sleep(1)
            asyncio.run(evaluate(args))
        state["state"] = "SUCCEEDED"
    except BaseException as exc:
        state.update(state="FAILED", error=repr(exc))
        raise
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        state["finished"] = time.time()
        write_json(args.output / "status.json", state)


def analyze(args: argparse.Namespace) -> None:
    runs = sorted(args.root.glob("*/repeat-*.jsonl"))
    if not runs:
        raise ValueError("No evaluation results; formal evaluation may still be pending")
    if args.expected_runs and len({path.parent for path in runs}) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} complete evaluation runs")
    manifest = read_json(args.manifest)
    indices = manifest["source_indices"]
    expected = set(indices["eval"] if isinstance(indices, dict) else indices)
    for path in runs:
        if read_json(path.parent / "status.json")["state"] != "SUCCEEDED":
            raise ValueError(f"Evaluation has not succeeded: {path.parent}")
        if {row["source_index"] for row in rows(path)} != expected:
            raise ValueError(f"Incomplete or different evaluation set: {path}")
    records = {str(path.relative_to(args.root)): rows(path) for path in runs}
    summaries = {}
    for name, data in records.items():
        compare(data, data)
        summaries[name] = dict(
            n=len(data),
            accuracy=sum(row["reward"]["acc"] for row in data) / len(data),
            reward=sum(row["reward"]["score"] for row in data) / len(data),
            correct=sorted(row["source_index"] for row in data if row["reward"]["acc"]),
        )
    write_json(
        args.root / "analysis.json",
        dict(
            summaries=summaries,
            comparisons={
                f"{a}_vs_{b}": compare(records[a], records[b])
                for i, a in enumerate(records)
                for b in list(records)[i + 1 :]
                if {r["source_index"] for r in records[a]} == {r["source_index"] for r in records[b]}
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit", "preflight", "run", "status", "stop", "analyze"])
    parser.add_argument("--base", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--expected-runs", type=int, default=0)
    args = parser.parse_args()
    required = dict(
        preflight=["gpu", "port"],
        prepare=["base", "root"],
        audit=["base", "root"],
        run=["base", "model", "data", "manifest", "output", "gpu", "port"],
        status=["output"],
        stop=["output"],
        analyze=["root", "manifest"],
    )
    for field in required[args.command]:
        if getattr(args, field) is None:
            parser.error(f"{args.command} requires --{field}")
    if args.repeats < 1 or args.concurrency < 1 or args.expected_runs < 0:
        parser.error("repeats/concurrency must be positive; expected-runs must be nonnegative")
    if args.command in ("status", "stop"):
        status_or_stop(args)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
