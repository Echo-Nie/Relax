# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The experiment launcher must never enter the caller's shared network."""

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
