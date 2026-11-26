# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.utils import post_process_env
from scripts.tools.sampling_replay_countdown import tracking_env
from scripts.training.text.sampling_replay_countdown_hooks import quality_stop
from tests.cpu.utils.test_sft_runtime_env import _make_args


def test_countdown_job_proxy_reaches_detached_actor_runtime(monkeypatch, tmp_path):
    env = tracking_env(tmp_path, tmp_path / "off")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    actual = post_process_env(_make_args(), {"env_vars": {}})["env_vars"]
    assert actual["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert actual["https_proxy"] == actual["HTTPS_PROXY"]
    assert actual["TENSORBOARD_DIR"] == str(tmp_path / "off/tensorboard_log")
    assert actual["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "0"


def test_countdown_quality_rule_does_not_stop_low_initial_reward():
    history = [dict(accuracy=0.03, truncated_fraction=0.95) for _ in range(15)]
    assert not quality_stop(history)
    history[-5:] = [dict(accuracy=0.0, truncated_fraction=1.0) for _ in range(5)]
    assert quality_stop(history)


def test_entry_restores_frozen_runtime_after_native_initialization(monkeypatch):
    import json
    import os
    import sys
    from types import SimpleNamespace

    from scripts.training.text.sampling_replay_countdown_entry import run

    frozen = {"HTTPS_PROXY": "http://127.0.0.1:18080", "NO_PROXY": "127.96.51.1"}
    monkeypatch.setenv("RELAX_COUNTDOWN_RUNTIME_ENV", json.dumps(frozen))
    monkeypatch.setenv("HTTPS_PROXY", "http://old-proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "localhost")
    sentinel = object()
    seen = []

    def parse_args():
        assert os.environ["HTTPS_PROXY"] == frozen["HTTPS_PROXY"]
        os.environ.update(HTTPS_PROXY="http://old-proxy.invalid:3128", NO_PROXY="localhost")
        return sentinel

    def main(args):
        assert args is sentinel
        seen.append({key: os.environ[key] for key in frozen})

    monkeypatch.setitem(sys.modules, "relax.entrypoints.train", SimpleNamespace(main=main, parse_args=parse_args))
    run()
    assert seen == [frozen]
