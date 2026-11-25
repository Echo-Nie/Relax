# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from scripts.tools.sampling_replay_exp08 import prompt_key, split_indices, summarize_avg


def test_split_excludes_whitespace_duplicates_and_long_prompts():
    data = [{"prompt": [{"role": "user", "content": f"question {i}"}]} for i in range(390)]
    data += [{"prompt": [{"role": "user", "content": "question   0"}]}]
    excluded = {prompt_key(data[0])}
    lengths = [10] * len(data)
    lengths[1] = 1025
    split = split_indices(data, excluded, lengths)
    train = {prompt_key(data[i]) for i in split["train"]}
    heldout = {prompt_key(data[i]) for i in split["eval"]}
    assert len(train) == 128 and len(heldout) == 256
    assert not train & heldout and not (train | heldout) & excluded
    assert 1 not in split["train"] + split["eval"]
    assert split == split_indices(data, excluded, lengths)


def records():
    return [
        dict(
            source_index=i,
            completion_index=j,
            reward=dict(acc=correct, score=1 if correct else -1),
            status="completed",
        )
        for i, answers in [(10, [True, False, False, False]), (20, [True, True, False, False])]
        for j, correct in enumerate(answers)
    ]


def test_avg_reports_mean_correctness_not_any_success_rate():
    result = summarize_avg(records(), [10, 20], 4)
    assert result["avg_accuracy"] == 0.375
    assert result["reward"] == -0.25
    assert result["per_question_accuracy"] == {"10": 0.25, "20": 0.5}
    assert result["any_correct"] == 2


@pytest.mark.parametrize("fault", ["drop", "duplicate", "substitute"])
def test_avg_rejects_incomplete_or_aliased_completions(fault):
    data = records()
    if fault == "drop":
        data.pop()
    elif fault == "duplicate":
        data[-1] = dict(data[0])
    else:
        data[-1]["source_index"] = 30
    with pytest.raises(ValueError, match="Missing, duplicate or substituted"):
        summarize_avg(data, [10, 20], 4)


def test_training_preflight_rejects_host_namespace_before_external_work(tmp_path):
    import argparse
    import json
    import os
    import sys

    from scripts.tools.sampling_replay_evaluate import process_identity
    from scripts.tools.sampling_replay_exp08 import preflight

    if sys.platform != "linux":
        pytest.skip("Linux namespace metadata required")
    instance = tmp_path / "ray-smoke-off"
    instance.mkdir()
    net = os.readlink("/proc/self/ns/net")
    (instance / "state.json").write_text(
        json.dumps(dict(pid=os.getpid(), netns=net, start_ticks=process_identity(os.getpid())))
    )
    (instance / "config.json").write_text(json.dumps(dict(parent_netns=net)))
    with pytest.raises(ValueError, match="recorded private experiment namespace"):
        preflight(argparse.Namespace(root=tmp_path, stage="smoke", mode="off"))


def test_evaluation_rejects_changed_argv_before_gpu_or_output_creation(tmp_path):
    import argparse
    import hashlib
    import json

    from scripts.tools.sampling_replay_evaluate import run

    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt": [], "label": "1"}\n')
    argv = tmp_path / "argv.json"
    argv.write_text("[]")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_indices=[10],
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                training_argv_sha256="different-fixed-config",
            )
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Evaluation argv differs"):
        run(argparse.Namespace(output=output, manifest=manifest, data=data, training_argv=argv))
    assert not output.exists()
