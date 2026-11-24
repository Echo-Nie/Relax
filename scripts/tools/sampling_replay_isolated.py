# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Give each s5678 experiment its own network, Ray cluster and GPU pair.

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


def start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]


def checked_state(instance: Path) -> dict:
    state = json.loads((instance / "state.json").read_text())
    pid = state["pid"]
    current = os.readlink(f"/proc/{pid}/ns/net")
    if current == os.readlink("/proc/self/ns/net") or current != state["netns"]:
        raise ValueError("Target must be the recorded private network namespace")
    if start_ticks(pid) != state["start_ticks"]:
        raise ValueError("Namespace holder PID has been reused")
    return state


def instance_env(config: dict) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=GPU_PAIRS[config["mode"]],
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
        "2",
        "--num-cpus",
        "32",
        "--temp-dir",
        config["tmpdir"] + "/ray",
    ]
    os.execvpe(argv[0], argv, instance_env(config))


def create(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        raise ValueError("Network namespace setup requires root with CAP_SYS_ADMIN/CAP_NET_ADMIN")
    socket.inet_aton(args.host_ip)
    args.instance.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": args.mode,
        "host_ip": args.host_ip,
        "parent_netns": os.readlink("/proc/self/ns/net"),
        "tmpdir": tempfile.mkdtemp(prefix=f"sr-{args.mode}5678-", dir="/tmp"),
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
    p.add_argument("--host-ip", required=True)
    p = sub.add_parser("exec", help="Run one explicit command inside the recorded namespace")
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("_hold")
    p.add_argument("--instance", type=Path, required=True)
    args = parser.parse_args()
    {"create": create, "exec": execute, "_hold": lambda a: hold(a.instance)}[args.action](args)


if __name__ == "__main__":
    main()
