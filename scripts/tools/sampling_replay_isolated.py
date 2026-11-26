# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Give each experiment its own network, Ray cluster and selected GPUs.

This does not submit training. Use exec explicitly to run the existing launcher
or inspect one private cluster. Never run PID-scanning cleanup in a network
namespace: the host PID space is deliberately unchanged.
"""

import argparse
import fcntl
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


GPU_PAIRS = {"off": "2,3", "on": "4,5"}


def parse_gpus(value: str) -> str:
    parts = value.split(",")
    if not all(part.isascii() and part.isdecimal() for part in parts):
        raise ValueError("GPUs must be a comma-separated list of integers in 0..7")
    ids = [int(part) for part in parts]
    if len(set(ids)) != len(ids) or any(gpu not in range(8) for gpu in ids):
        raise ValueError("GPUs must be unique integers in 0..7")
    return ",".join(map(str, ids))


def check_gpus_idle(gpus: str) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpus,
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [tuple(map(int, line.split(","))) for line in output.splitlines() if line.strip()]
    if any(len(row) != 3 for row in rows) or sorted(row[0] for row in rows) != sorted(map(int, gpus.split(","))):
        raise ValueError("nvidia-smi did not return exactly the selected GPUs")
    for gpu, memory, utilization in rows:
        if memory > 100 or utilization:
            raise ValueError(f"GPU {gpu} is busy: {memory} MiB, utilization {utilization}%")


def start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]


def checked_state(instance: Path, *, inside: bool = False) -> dict:
    state = json.loads((instance / "state.json").read_text())
    pid = state["pid"]
    current = os.readlink(f"/proc/{pid}/ns/net")
    caller = os.readlink("/proc/self/ns/net")
    if current != state["netns"]:
        raise ValueError("Target must be the recorded private network namespace")
    if inside:
        parent = json.loads((instance / "config.json").read_text())["parent_netns"]
        if current == parent or caller != current:
            raise ValueError("Caller must be inside the recorded private network namespace, distinct from its parent")
    elif current == caller:
        raise ValueError("Target must be the recorded private network namespace")
    if start_ticks(pid) != state["start_ticks"]:
        raise ValueError("Namespace holder PID has been reused")
    return state


def instance_env(config: dict) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=config.get("gpus", GPU_PAIRS[config["mode"]]),
        TMPDIR=config["tmpdir"],
        RAY_TMPDIR=config["tmpdir"],
        RAY_ADDRESS="127.0.0.1:6379",
        RAY_DASHBOARD="http://127.0.0.1:8265",
        MASTER_ADDR=config["host_ip"],
        RELAX_SKIP_CLUSTER_CLEANUP="1",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="lo",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        RAY_USAGE_STATS_ENABLED="0",
        NO_PROXY="*",
        no_proxy="*",
    )
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(name, None)
    return env


def setup_loopback(host_ip: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack("16sh", b"lo", 0)
        flags = struct.unpack("16sh", fcntl.ioctl(sock, 0x8913, request))[1]
        fcntl.ioctl(sock, 0x8914, struct.pack("16sh", b"lo", flags | 1))
        for operation, address in ((0x8916, host_ip), (0x891C, "255.255.255.255")):
            request = struct.pack(
                "16sH2s4s8s", b"lo:sr", socket.AF_INET, b"\0\0", socket.inet_aton(address), b"\0" * 8
            )
            fcntl.ioctl(sock, operation, request)


def hold(instance: Path) -> None:
    config = json.loads((instance / "config.json").read_text())
    netns = os.readlink("/proc/self/ns/net")
    if netns == config["parent_netns"]:
        raise ValueError("Refusing to configure the host network")
    setup_loopback(config["host_ip"])
    pid = os.getpid()
    state = {"pid": pid, "netns": netns, "start_ticks": start_ticks(pid)}
    temporary = instance / "state.pending"
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(instance / "state.json")
    argv = [
        "ray",
        "start",
        "--head",
        "--block",
        "--disable-usage-stats",
        "--node-ip-address",
        config["host_ip"],
        "--port",
        "6379",
        "--dashboard-host",
        "127.0.0.1",
        "--dashboard-port",
        "8265",
        "--num-gpus",
        str(len(config.get("gpus", GPU_PAIRS[config["mode"]]).split(","))),
        "--num-cpus",
        "32",
        "--temp-dir",
        config.get("ray_tmpdir", config["tmpdir"] + "/ray"),
    ]
    os.execvpe(argv[0], argv, instance_env(config))


def check_ray_socket_path(ray_tmpdir: Path) -> None:
    # Linux AF_UNIX paths are limited to 107 bytes. Ray adds a timestamp,
    # PID and dashboard module name beneath the supplied temporary directory.
    pid_max = Path("/proc/sys/kernel/pid_max").read_text().strip()
    session = f"session_2000-01-01_00-00-00_000000_{pid_max}"
    path = ray_tmpdir.resolve() / session / "sockets" / "dash_MetricsHead"
    if len(os.fsencode(path)) > 107:
        raise OSError(f"AF_UNIX path length exceeds 107 bytes: {path}")


def create(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        raise ValueError("Network namespace setup requires root with CAP_SYS_ADMIN/CAP_NET_ADMIN")
    socket.inet_aton(args.host_ip)
    selected = getattr(args, "gpus", None)
    gpus = parse_gpus(GPU_PAIRS[args.mode] if selected is None else selected)
    if args.instance.exists():
        raise FileExistsError(f"Instance already exists: {args.instance}")
    if args.ray_tmpdir is not None and args.ray_tmpdir.resolve().exists():
        raise FileExistsError(f"Cache directory already exists: {args.ray_tmpdir}")
    check_gpus_idle(gpus)
    if args.ray_tmpdir is not None:
        check_ray_socket_path(args.ray_tmpdir)
    args.instance.mkdir(parents=True, exist_ok=False)
    if args.ray_tmpdir is None:
        ray_tmpdir = Path(tempfile.mkdtemp(prefix=f"sr-{args.mode}5678-", dir="/tmp"))
    else:
        ray_tmpdir = args.ray_tmpdir.resolve()
        ray_tmpdir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": args.mode,
        "gpus": gpus,
        "host_ip": args.host_ip,
        "parent_netns": os.readlink("/proc/self/ns/net"),
        "tmpdir": str(ray_tmpdir),
        "ray_tmpdir": str(ray_tmpdir if args.ray_tmpdir is not None else ray_tmpdir / "ray"),
    }
    (args.instance / "config.json").write_text(json.dumps(config, indent=2))
    with (args.instance / "ray-head.log").open("w") as log:
        process = subprocess.Popen(
            [
                "unshare",
                "--net",
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "_hold",
                "--instance",
                str(args.instance.resolve()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=instance_env(config),
        )
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"Private cluster exited; inspect {args.instance / 'ray-head.log'}")
        if (args.instance / "state.json").exists():
            state = checked_state(args.instance)
            print(json.dumps({**state, **config}, indent=2))
            return
        time.sleep(0.1)
    raise TimeoutError(f"Namespace startup pending; inspect {args.instance / 'ray-head.log'}")


def execute(args: argparse.Namespace) -> None:
    state = checked_state(args.instance)
    config = json.loads((args.instance / "config.json").read_text())
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("Provide an explicit command; training is never submitted automatically")
    argv = ["nsenter", f"--net=/proc/{state['pid']}/ns/net", "--", *command]
    os.execvpe(argv[0], argv, instance_env(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("create", help="Start only a private Ray cluster, not training")
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("--mode", choices=GPU_PAIRS, required=True)
    p.add_argument("--gpus", help="Physical GPU IDs, e.g. 0,1,2,3,4,5,6,7; defaults to the mode GPU pair")
    p.add_argument("--host-ip", required=True)
    p.add_argument("--ray-tmpdir", type=Path, help="New private cache directory; existing paths are refused")
    p = sub.add_parser("exec", help="Run one explicit command inside the recorded namespace")
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("_hold")
    p.add_argument("--instance", type=Path, required=True)
    args = parser.parse_args()
    {"create": create, "exec": execute, "_hold": lambda a: hold(a.instance)}[args.action](args)


if __name__ == "__main__":
    main()
