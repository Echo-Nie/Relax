# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The experiment launcher must never enter the caller's shared network."""

import argparse
import json
import os
import sys

import pytest

from scripts.tools import sampling_replay_isolated
from scripts.tools.sampling_replay_isolated import checked_state, start_ticks


@pytest.mark.skipif(sys.platform != "linux", reason="Linux network namespace metadata required")
def test_checked_state_rejects_shared_network_namespace(tmp_path):
    pid = os.getpid()
    (tmp_path / "state.json").write_text(
        json.dumps({"pid": pid, "netns": os.readlink("/proc/self/ns/net"), "start_ticks": start_ticks(pid)})
    )
    with pytest.raises(ValueError, match="private network namespace"):
        checked_state(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux network namespace metadata required")
def test_hold_rejects_host_network_before_interface_changes(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"parent_netns": os.readlink("/proc/self/ns/net"), "host_ip": "127.0.0.1"})
    )

    def forbidden_network_change(*args):
        pytest.fail("Attempted to modify the caller's shared network")

    monkeypatch.setattr(sampling_replay_isolated, "setup_loopback", forbidden_network_change)
    with pytest.raises(ValueError, match="Refusing to configure the host network"):
        sampling_replay_isolated.hold(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux namespace creation required")
def test_create_refuses_existing_cache_before_starting_cluster(tmp_path, monkeypatch):
    cache = tmp_path / "other-task-cache"
    cache.mkdir()
    sentinel = cache / "sentinel"
    sentinel.write_text("owned by another task")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Started a cluster using an existing task cache")

    monkeypatch.setattr(sampling_replay_isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "Popen", forbidden_launch)
    args = argparse.Namespace(instance=tmp_path / "instance", mode="off", host_ip="127.0.0.1", ray_tmpdir=cache)
    with pytest.raises(FileExistsError):
        sampling_replay_isolated.create(args)
    assert sentinel.read_text() == "owned by another task"
    assert not (args.instance / "config.json").exists()


@pytest.mark.parametrize("gpus", ["", "0,,1", "-1", "0,0", "0,00", "8", "0,1.0", "0, 1", "０"])
def test_create_rejects_invalid_gpu_selection_before_query_or_launch(tmp_path, monkeypatch, gpus):
    def forbidden_external(*args, **kwargs):
        pytest.fail("Invalid GPU selection reached an external command")

    monkeypatch.setattr(sampling_replay_isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "check_output", forbidden_external)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "Popen", forbidden_external)
    args = argparse.Namespace(
        instance=tmp_path / "instance", mode="off", host_ip="127.0.0.1", ray_tmpdir=None, gpus=gpus
    )
    with pytest.raises(ValueError, match="GPUs must"):
        sampling_replay_isolated.create(args)
    assert not args.instance.exists()


@pytest.mark.parametrize("busy", ["7, 101, 0", "7, 0, 1"])
def test_create_rejects_busy_selected_gpu_without_creating_cluster(tmp_path, monkeypatch, busy):
    def query(command, **kwargs):
        selected = next(arg.removeprefix("--id=") for arg in command if arg.startswith("--id="))
        return "\n".join(busy if gpu == "7" else f"{gpu}, 0, 0" for gpu in selected.split(","))

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Started a cluster with a busy selected GPU")

    monkeypatch.setattr(sampling_replay_isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "check_output", query)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "Popen", forbidden_launch)
    args = argparse.Namespace(
        instance=tmp_path / "instance",
        mode="off",
        host_ip="127.0.0.1",
        ray_tmpdir=tmp_path / "cache",
        gpus="0,1,2,3,4,5,6,7",
    )
    with pytest.raises(ValueError, match="GPU 7 is busy"):
        sampling_replay_isolated.create(args)
    assert not args.instance.exists()
    assert not args.ray_tmpdir.exists()


def test_gpu_query_rejects_missing_selected_device(monkeypatch):
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "check_output", lambda *a, **k: "0, 0, 0\n")
    with pytest.raises(ValueError, match="exactly the selected GPUs"):
        sampling_replay_isolated.check_gpus_idle("0,1")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process identity required")
def test_inner_namespace_accepts_recorded_caller_but_preserves_external_entry_guard(tmp_path, monkeypatch):
    pid = os.getpid()
    state = dict(pid=pid, netns="net:[private]", start_ticks=start_ticks(pid))
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "config.json").write_text(json.dumps(dict(parent_netns="net:[parent]")))
    # Only the kernel namespace lookup is isolated. PID identity and the entire
    # production validator run normally; this reproduces exec -> inner submit.
    monkeypatch.setattr(sampling_replay_isolated.os, "readlink", lambda path: "net:[private]")
    with pytest.raises(ValueError, match="private network namespace"):
        checked_state(tmp_path)
    assert checked_state(tmp_path, inside=True) == state


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process identity required")
@pytest.mark.parametrize("fault", ["parent", "caller", "holder", "pid_reused"])
def test_inner_namespace_rejects_wrong_ownership_or_identity(tmp_path, monkeypatch, fault):
    pid = os.getpid()
    state = dict(pid=pid, netns="net:[private]", start_ticks=start_ticks(pid))
    if fault == "pid_reused":
        state["start_ticks"] = "wrong-start-time"
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "config.json").write_text(
        json.dumps(dict(parent_netns="net:[private]" if fault == "parent" else "net:[parent]"))
    )

    def namespace(path):
        if path == "/proc/self/ns/net" and fault == "caller":
            return "net:[parent]"
        if path == f"/proc/{pid}/ns/net" and fault == "holder":
            return "net:[replacement]"
        return "net:[private]"

    monkeypatch.setattr(sampling_replay_isolated.os, "readlink", namespace)
    with pytest.raises(ValueError, match="namespace|PID"):
        checked_state(tmp_path, inside=True)


def test_create_rejects_oversized_ray_socket_before_starting_cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling_replay_isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "check_output", lambda *a, **k: "0, 0, 0\n")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Started Ray with a socket path that cannot bind")

    monkeypatch.setattr(sampling_replay_isolated.subprocess, "Popen", forbidden_launch)
    args = argparse.Namespace(
        instance=tmp_path / "instance",
        mode="off",
        gpus="0",
        host_ip="127.0.0.1",
        ray_tmpdir=tmp_path / ("long-cache-" * 12),
    )
    with pytest.raises(OSError, match="AF_UNIX path length"):
        sampling_replay_isolated.create(args)
    assert not args.instance.exists()
    assert not args.ray_tmpdir.exists()


@pytest.mark.parametrize("mode", ["r3", "tis", "sc"])
def test_new_mode_requires_explicit_gpus_before_query_or_launch(tmp_path, monkeypatch, mode):
    def forbidden_external(*args, **kwargs):
        pytest.fail("A new mode without an explicit GPU allocation reached an external command")

    monkeypatch.setattr(sampling_replay_isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "check_output", forbidden_external)
    monkeypatch.setattr(sampling_replay_isolated.subprocess, "Popen", forbidden_external)
    args = argparse.Namespace(
        instance=tmp_path / "instance", mode=mode, host_ip="127.0.0.1", ray_tmpdir=None, gpus=None
    )
    with pytest.raises(ValueError, match="--gpus is required"):
        sampling_replay_isolated.create(args)
    assert not args.instance.exists()


@pytest.mark.parametrize("mode", ["r3", "tis", "sc"])
def test_new_mode_exec_preserves_explicit_gpu_isolation(tmp_path, monkeypatch, mode):
    # Exercise the production exec boundary: the old eager dictionary fallback
    # raised KeyError for these modes even when explicit GPUs were configured.
    gpus = "0,1,2,3,4,5,6,7"
    (tmp_path / "config.json").write_text(
        json.dumps(dict(mode=mode, gpus=gpus, tmpdir=str(tmp_path), host_ip="127.9.0.1"))
    )
    monkeypatch.setattr(sampling_replay_isolated, "checked_state", lambda instance: {"pid": 123})

    def capture_exec(executable, argv, env):
        assert executable == "nsenter"
        assert env["CUDA_VISIBLE_DEVICES"] == gpus
        assert argv[-2:] == ["python", "--version"]
        raise SystemExit(0)

    monkeypatch.setattr(sampling_replay_isolated.os, "execvpe", capture_exec)
    with pytest.raises(SystemExit) as stopped:
        sampling_replay_isolated.execute(argparse.Namespace(instance=tmp_path, command=["--", "python", "--version"]))
    assert stopped.value.code == 0
