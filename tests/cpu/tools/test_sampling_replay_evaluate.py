"""Protect evaluation comparisons against dropped questions and parser
confounding."""

import argparse
import json
import socket
import subprocess
import sys

import pytest

from scripts.tools.sampling_replay_evaluate import analyze, compare, preflight, process_identity, status_or_stop


def record(index, tokens, response, correct):
    return dict(
        source_index=index,
        input_ids=[1, 2],
        output_ids=tokens,
        response=response,
        reward=dict(acc=correct, score=1 if correct else -1),
    )


def test_compare_distinguishes_output_changes_from_reward_changes():
    left = [record(10, [3], "Answer: 3", False), record(20, [4], "Answer: 4", True)]
    right = [record(20, [4], "Answer: 4", False), record(10, [5], "Answer: 5", True)]
    result = compare(left, right)
    assert result["output_tokens_changed"] == [10]
    assert result["text_changed"] == [10]
    assert result["reward_changed_same_text"] == [20]
    assert result["correct_left"] == [20]
    assert result["correct_right"] == [10]


@pytest.mark.parametrize("right", [[], [record(11, [3], "x", False)]])
def test_compare_rejects_missing_or_substituted_questions(right):
    with pytest.raises(ValueError, match="Different question sets"):
        compare([record(10, [3], "x", False)], right)


def test_compare_rejects_duplicate_questions():
    row = record(10, [3], "x", False)
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        compare([row, row], [row])


@pytest.mark.parametrize("stale_identity", [True, False])
def test_stop_does_not_signal_unrelated_or_reused_pid(tmp_path, stale_identity):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "status.json").write_text(
            json.dumps(
                dict(pid=child.pid, starttime="wrong-starttime" if stale_identity else process_identity(child.pid))
            )
        )
        args = argparse.Namespace(command="stop", output=tmp_path)
        if stale_identity:
            status_or_stop(args)
        else:
            with pytest.raises(ValueError, match="does not belong"):
                status_or_stop(args)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_preflight_allows_tcp_time_wait_but_rejects_a_live_listener(monkeypatch):
    # Only the external GPU query is isolated; bind/listen/FIN use real TCP.
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0, 0")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        args = argparse.Namespace(gpu=0, port=listener.getsockname()[1])
        with pytest.raises(OSError):
            preflight(args)
        with socket.create_connection(("127.0.0.1", args.port), timeout=2) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    preflight(args)


def test_analysis_rejects_missing_evaluation_runs(tmp_path):
    run = tmp_path / "completed-run"
    run.mkdir()
    (run / "repeat-0.jsonl").write_text(json.dumps(record(10, [3], "x", False)) + "\n")
    args = argparse.Namespace(root=tmp_path, expected_runs=2)
    with pytest.raises(ValueError, match="Expected 2 complete evaluation runs"):
        analyze(args)
