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
