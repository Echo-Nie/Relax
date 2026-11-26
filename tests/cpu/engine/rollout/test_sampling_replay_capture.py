# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import asyncio
import copy
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from relax.engine.rollout import sglang_rollout as rollout
from relax.utils.arguments import apply_custom_config_overrides, get_slime_extra_args_provider
from relax.utils.sampling_replay import (
    configure_sampling_replay_request,
    parse_sampling_replay_response,
    validate_sampling_replay_args,
)
from relax.utils.types import Sample


def args_for_replay(**overrides):
    return argparse.Namespace(
        **{
            "use_rollout_sampling_replay": True,
            "sglang_sampling_backend": "pytorch",
            "rollout_temperature": 0.7,
            "rollout_top_k": 4,
            "rollout_top_p": 0.8,
            "advantage_estimator": "grpo",
            "ci_test": False,
            "use_opd": False,
            "use_rollout_routing_replay": False,
            "use_rollout_indexer_replay": False,
            "use_slime_router": False,
            "slime_router_sticky": False,
            "sglang_router_policy": "round_robin",
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 1,
            "lora_rank": 0,
            "sglang_speculative_algorithm": None,
            "partial_rollout": True,
            "mask_offpolicy_in_partial_rollout": True,
            "custom_generate_function_path": None,
            **overrides,
        }
    )


def response(tokens=(3, 1), *, finish="abort", text=""):
    rows = [[3, 2], [1]] if len(tokens) == 2 else [[2, 1, 3]]
    probabilities = (
        [[math.log(0.6), math.log(0.4)], [0.0]]
        if len(tokens) == 2
        else [[math.log(0.2), math.log(0.5), math.log(0.3)]]
    )
    return {
        "text": text,
        "output_ids": list(tokens),
        "meta_info": {
            "output_token_logprobs": [[-9.0, token, None] for token in tokens],
            "output_token_sampling_mask": rows,
            "output_token_sampling_logprobs": probabilities,
            "output_token_sampling_mask_length": len(tokens),
            "completion_tokens": len(tokens),
            "finish_reason": {"type": finish},
        },
    }


@pytest.fixture
def generate_boundary(monkeypatch):
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: [0]),
        processor=None,
        aborted=False,
        semaphore=asyncio.Semaphore(1),
        dp_rank_context=nullcontext,
    )
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    calls, outputs = [], []

    async def post(args, state, url, payload, headers, evaluation):
        calls.append(copy.deepcopy(payload))
        value = outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(rollout, "_post_generation", post)
    return calls, outputs, state


def generate(sample, args=None, max_new_tokens=5, *, evaluation=False):
    args = args or args_for_replay()
    return asyncio.run(
        rollout.generate(
            args,
            sample,
            {
                "max_new_tokens": max_new_tokens,
                "temperature": args.rollout_temperature,
                "top_k": args.rollout_top_k,
                "top_p": args.rollout_top_p,
            },
            evaluation=evaluation,
        )
    )


def test_sampling_replay_capture_uses_behavior_and_resumes_empty_text_abort(generate_boundary):
    calls, outputs, _ = generate_boundary
    outputs.extend([response(), response((2,), finish="length")])
    sample = generate(Sample(prompt="p"))
    assert sample.status == Sample.Status.ABORTED
    assert sample.response == "" and sample.response_length == 2
    assert sample.rollout_log_probs == [math.log(0.6), 0.0]
    assert calls[0]["return_sampling_mask"] is True
    assert calls[0]["sampling_logprobs_mode"] == "support"
    first = sample.rollout_sampling_mask
    sample.loss_mask = [0, 0]
    generate(sample)
    assert calls[1]["input_ids"] == [0, 3, 1]
    assert calls[1]["sampling_params"]["max_new_tokens"] == 3
    assert sample.tokens == sample.rollout_tokens == [0, 3, 1, 2]
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 2, 1, 2, 1, 3]
    assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 3, 6]
    assert first.ids.tolist() == [3, 2, 1]
    assert sample.rollout_log_probs == [math.log(0.6), 0.0, math.log(0.2)]
    assert sample.loss_mask == [0, 0, 1]
    assert sample.status == Sample.Status.TRUNCATED
    from relax.utils.utils import convert_samples_to_train_data

    sample.index, sample.reward = 0, 0.0
    data = convert_samples_to_train_data(
        args_for_replay(
            custom_reward_post_process_path=None,
            reward_key=None,
            rewards_normalization=False,
            multimodal_keys=None,
            debug_train_only=False,
        ),
        [sample],
    )
    assert data["rollout_sampling_mask_offsets"][0].tolist() == [0, 2, 3, 6]
    assert data["rollout_sampling_mask_ids"][0].tolist() == [3, 2, 1, 2, 1, 3]


@pytest.mark.parametrize("evaluation", [False, True])
def test_sampling_replay_disabled_or_evaluation_keeps_legacy_behavior(generate_boundary, evaluation):
    calls, outputs, _ = generate_boundary
    outputs.append(response(finish="stop", text="answer"))
    sample = generate(
        Sample(prompt="p"), args_for_replay(use_rollout_sampling_replay=evaluation), evaluation=evaluation
    )
    assert sample.rollout_sampling_mask is None
    assert sample.rollout_log_probs == [-9.0, -9.0]
    assert "return_sampling_mask" not in calls[0]
    assert "sampling_logprobs_mode" not in calls[0]


def test_sampling_replay_zero_budget_is_empty_csr_without_request(generate_boundary):
    calls, _, _ = generate_boundary
    sample = generate(Sample(prompt="p"), max_new_tokens=0)
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.tokens == [0]
    assert sample.rollout_sampling_mask.ids.tolist() == []
    assert sample.rollout_sampling_mask.offsets.tolist() == [0]
    assert sample.rollout_log_probs == []
    assert not calls


def test_sampling_replay_missing_prefix_fails_before_request(generate_boundary):
    calls, _, _ = generate_boundary
    with pytest.raises(ValueError, match="missing its prefix support"):
        generate(Sample(prompt="p", tokens=[0, 3], response_length=1))
    assert not calls


@pytest.mark.parametrize("failure", ["missing", "rows", "count", "output_ids", "duplicate", "absent", "mass", "empty"])
def test_sampling_replay_parser_rejects_invalid_server_data(failure):
    output = response()
    meta = output["meta_info"]
    if failure == "missing":
        del meta["output_token_sampling_logprobs"]
    elif failure == "rows":
        meta["output_token_sampling_mask"].pop()
    elif failure == "count":
        meta["completion_tokens"] = 3
    elif failure == "output_ids":
        output["output_ids"] = [2, 1]
    elif failure == "duplicate":
        meta["output_token_sampling_mask"][0] = [3, 3]
    elif failure == "absent":
        meta["output_token_sampling_mask"][0] = [0, 2]
    elif failure == "mass":
        meta["output_token_sampling_logprobs"][0] = [0.0, 0.0]
    else:
        meta["output_token_sampling_mask"][0] = []
    with pytest.raises(ValueError, match="sampling replay"):
        parse_sampling_replay_response(output, top_k=4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sglang_sampling_backend": "flashinfer"},
        {"sglang_speculative_algorithm": "EAGLE"},
        {"rollout_top_k": -1},
        {"rollout_temperature": 0.0},
        {"train_backend": "fsdp"},
        {"custom_generate_function_path": "custom.generate"},
    ],
)
def test_sampling_replay_configuration_fails_closed(overrides):
    with pytest.raises(ValueError, match="sampling replay"):
        validate_sampling_replay_args(args_for_replay(**overrides))
    validate_sampling_replay_args(args_for_replay(use_rollout_sampling_replay=False, **overrides))


@pytest.mark.parametrize(
    "sampling", [{"temperature": 1.0}, {"frequency_penalty": 0.1}, {"logit_bias": {"1": 2}}, {"no_stop_trim": False}]
)
def test_sampling_replay_rejects_changed_request_distribution(sampling):
    with pytest.raises(ValueError, match="sampling replay"):
        configure_sampling_replay_request(args_for_replay(), {"sampling_params": sampling})


def test_sampling_replay_cli_defaults_off_and_effective_yaml_is_validated(tmp_path):
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    assert not parser.parse_args([]).use_rollout_sampling_replay
    assert parser.parse_args(["--use-rollout-sampling-replay"]).use_rollout_sampling_replay
    config = tmp_path / "override.yaml"
    config.write_text("sglang_sampling_backend: flashinfer\n")
    args = args_for_replay(custom_config_path=str(config), loss_type="policy_loss", kl_coef=0.0, kl_loss_coef=0.0)
    validate_sampling_replay_args(args)
    with pytest.raises(ValueError, match="sampling replay requires --sglang-sampling-backend pytorch"):
        apply_custom_config_overrides(args)


@pytest.mark.parametrize("failure", ["metadata", "http_abort"])
def test_sampling_replay_incomplete_abort_preserves_committed_prefix(generate_boundary, failure):
    _, outputs, _ = generate_boundary
    outputs.append(response())
    sample = generate(Sample(prompt="p"))
    original = sample.to_dict()
    if failure == "metadata":
        invalid = response((2,))
        del invalid["meta_info"]["output_token_sampling_mask"]
        outputs.append(invalid)
        expected = ValueError
    else:
        outputs.append(rollout.GenerationAborted())
        expected = rollout.GenerationAborted
    with pytest.raises(expected):
        generate(sample)
    assert sample.to_dict() == original


def test_sampling_replay_rejects_client_token_sanitization(generate_boundary):
    _, outputs, state = generate_boundary
    state.tokenizer.image_token_id, state.tokenizer.pad_token_id = 3, 0
    outputs.append(response())
    with pytest.raises(ValueError, match="unchanged generated token IDs"):
        generate(Sample(prompt="p"))


def test_sampling_replay_capture_preserves_score_centering_candidates(generate_boundary):
    calls, outputs, _ = generate_boundary
    output = response(finish="stop")
    output["meta_info"]["output_token_logprobs"] = [[math.log(0.6), 3, None], [0.0, 1, None]]
    output["meta_info"]["output_top_logprobs"] = [
        [[math.log(0.6), 3, None], [math.log(0.4), 2, None]],
        [[0.0, 1, None], [float("-inf"), 0, None]],
    ]
    outputs.append(output)
    sample = generate(
        Sample(prompt="p"),
        args_for_replay(
            advantage_estimator="score_centering", rollout_top_logprobs_num=2, score_centering_vocab_size=4
        ),
    )
    assert calls[0]["top_logprobs_num"] == 2
    assert calls[0]["sampling_logprobs_mode"] == "support"
    assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 3]
    assert sample.rollout_topk_token_ids.tolist() == [[3, 2], [1, -1]]
    assert sample.rollout_topk_support_complete.tolist() == [True, True]
    assert sample.rollout_log_probs == [math.log(0.6), 0.0]


@pytest.mark.parametrize("unsupported", ["media", "partial_genrm", "sample_generator"])
def test_sampling_replay_rejects_token_rewriting_paths_before_request(generate_boundary, unsupported):
    calls, _, state = generate_boundary
    sample = Sample(prompt="p")
    if unsupported == "media":
        sample.multimodal_inputs = {"images": ["image"]}
    elif unsupported == "partial_genrm":
        state.partial_genrm = object()
    else:
        sample.generate_function_path = "custom.generate"
    with pytest.raises(ValueError, match="sampling replay"):
        if unsupported == "sample_generator":
            asyncio.run(rollout._dispatch_generate(state, args_for_replay(), sample, {"max_new_tokens": 1}))
        else:
            generate(sample)
    assert not calls


@pytest.mark.parametrize("mode", ["off", "on", "tis"])
def test_countdown_budget_delegate_preserves_behavior_and_only_sr_exports_csr(generate_boundary, mode):
    from relax.utils.sampling_replay import COUNTDOWN_GENERATOR

    calls, outputs, state = generate_boundary
    outputs.append(response(finish="stop"))
    args = args_for_replay(
        use_rollout_sampling_replay=mode == "on",
        use_tis=mode == "tis",
        advantage_estimator="rloo",
        custom_generate_function_path=COUNTDOWN_GENERATOR,
    )
    sample = asyncio.run(
        rollout._dispatch_generate(
            state, args, Sample(prompt="p"), dict(max_new_tokens=512, temperature=0.7, top_k=4, top_p=0.8)
        )
    )
    assert calls[0]["sampling_params"]["max_new_tokens"] == 511
    assert calls[0]["return_sampling_mask"] is True
    assert calls[0]["sampling_logprobs_mode"] == "support"
    assert sample.rollout_log_probs == [math.log(0.6), 0.0]
    assert (sample.rollout_sampling_mask is not None) == (mode == "on")
    assert sample.metadata["countdown_support_sizes"] == [2, 1]
    assert args.use_rollout_sampling_replay == (mode == "on")
    assert args.custom_generate_function_path == COUNTDOWN_GENERATOR
