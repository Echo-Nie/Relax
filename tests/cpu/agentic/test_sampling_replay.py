# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute Agentic capture/export and real scoring; isolate model HTTP only."""

import asyncio
import json
import math
import multiprocessing
import tempfile
import time
from argparse import Namespace
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import torch

from relax.agentic.pipeline import TrainingFieldArtifact, runtime
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest
from relax.backends.megatron import data, loss
from relax.utils.sampling_mask import RolloutSamplingMask
from relax.utils.training.data_fields import build_data_fields
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data


TOKENS = [6, 0, 3, 7, 4, 6, 2, 5]
ROWS = [[3, 1], [7, 0, 4], [4], [6], [2, 6], [5, 1, 6]]
BEHAVIOR = [math.log(0.6), math.log(0.2), 0.0, 0.0, math.log(0.25), math.log(0.5)]


class Tokenizer:
    pad_token_id = None

    def decode(self, tokens, **_kwargs):
        return "".join(chr(65 + token) for token in tokens)


def replay_args(**overrides):
    return Namespace(
        **{
            "use_rollout_sampling_replay": True,
            "use_agentic_rollout": True,
            "sglang_sampling_backend": "pytorch",
            "rollout_temperature": 0.7,
            "rollout_top_k": 4,
            "rollout_top_p": 0.8,
            "sglang_router_ip": "localhost",
            "sglang_router_port": 1,
            "sglang_router_policy": "round_robin",
            "slime_router_sticky": False,
            "use_rollout_routing_replay": False,
            "agentic_reasoning_parser": None,
            "agentic_tool_call_parser": None,
            "advantage_estimator": "grpo",
            "loss_type": "policy_loss",
            "qkv_format": "thd",
            "allgather_cp": False,
            "log_probs_chunk_size": 2,
            "entropy_coef": 0.0,
            "fully_async": False,
            "hybrid": False,
            "kl_coef": 0.0,
            "use_opd": False,
            "opd_token_selection": "student_sampled",
            "use_kl_loss": False,
            "multimodal_keys": None,
            "reward_key": None,
            "rewards_normalization": False,
            "custom_reward_post_process_path": None,
            "debug_train_only": False,
            **overrides,
        }
    )


def backend(args):
    adapter = object.__new__(runtime.SGLangBackendAdapter)
    adapter._args = args
    adapter._session_lifecycle = False
    adapter.tokenizer = Tokenizer()
    adapter.compiler = SimpleNamespace(processor=None)
    return adapter


def output(tokens, rows, probabilities, finish="stop"):
    return {
        "output_ids": list(tokens),
        "meta_info": {
            "output_token_logprobs": [[-9.0, token, None] for token in tokens],
            "output_token_sampling_mask": rows,
            "output_token_sampling_logprobs": [[math.log(p) for p in row] for row in probabilities],
            "output_token_sampling_mask_length": len(tokens),
            "completion_tokens": len(tokens),
            "finish_reason": {"type": finish},
        },
    }


def forest_with_prompt(replay=True):
    forest = SessionForest.create_empty(session_id="session", index=0, use_rollout_sampling_replay=replay)
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "prompt"}],
        train_token_delta=[6, 0],
        rollout_token_delta=[6, 0],
        tools=[],
        chat_template_kwargs={},
    )
    return forest, prompt


def shard_and_session(args, forest):
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._generation_backend = backend(args)
    return shard, SimpleNamespace(forest=forest, resp_state_hash_by_request_id={})


def request(parent, name="request", replay=True):
    return InflightRequest(
        request_id=name,
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
        use_rollout_sampling_replay=replay,
        pending_status="completed",
    )


async def two_turn_sample():
    args = replay_args()
    forest, parent = forest_with_prompt()
    shard, session = shard_and_session(args, forest)
    responses = [
        output([3, 7], [[3, 1], [7, 0, 4]], [[0.6, 0.4], [0.2, 0.3, 0.5]]),
        output([2, 5], [[2, 6], [5, 1, 6]], [[0.25, 0.75], [0.5, 0.2, 0.3]]),
    ]
    calls = []

    async def post(_url, payload, **_kwargs):
        calls.append(payload)
        return responses.pop(0)

    with patch.object(runtime, "post", post):
        for turn in range(2):
            ir = request(parent, f"turn-{turn}")
            result = await shard._generation_backend.generate(
                input_ids=forest.build_execution_prefix(parent.state_hash).rollout_token_prefix,
                sampling_params={"max_new_tokens": 2},
                session_id="session",
                request_id=ir.request_id,
            )
            shard._apply_generate_result(ir, result)
            shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
            parent = forest.nodes_by_hash[session.resp_state_hash_by_request_id[ir.request_id]]
            if turn == 0:
                parent = forest.append_obs(
                    parent_state_hash=parent.state_hash,
                    rollout_id=0,
                    abort_count=0,
                    messages_delta=[{"role": "tool", "tool_call_id": "call", "content": "result"}],
                    train_token_delta=[4, 6],
                    rollout_token_delta=[4, 6],
                )
    assert calls[0]["return_sampling_mask"] is True
    assert calls[0]["sampling_logprobs_mode"] == "support"
    assert calls[1]["input_ids"] == [6, 0, 3, 7, 4, 6]
    sample = forest.build_sample(leaf_state_hash=parent.state_hash, tokenizer=Tokenizer())
    sample.reward = 1.0
    return sample


def assert_alignment(sample):
    assert sample.tokens == TOKENS
    assert sample.response_length == 6
    assert sample.loss_mask == [1, 1, 0, 0, 1, 1]
    assert sample.rollout_log_probs == BEHAVIOR
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 1, 7, 0, 4, 4, 6, 2, 6, 5, 1, 6]
    assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 5, 6, 7, 9, 12]
    assert len(sample.rollout_sampling_mask) == len(sample.rollout_log_probs) == len(sample.loss_mask) == 6


@contextmanager
def cpu_actor_runtime(args):
    with ExitStack() as stack:
        for name, value in {
            "get_context_parallel_world_size": 1,
            "get_context_parallel_rank": 0,
            "get_tensor_model_parallel_world_size": 1,
            "get_tensor_model_parallel_rank": 0,
            "get_tensor_model_parallel_group": None,
            "is_pipeline_last_stage": True,
        }.items():
            stack.enter_context(patch.object(loss.mpu, name, return_value=value))
        stack.enter_context(patch.object(data, "get_args", return_value=args))
        yield


def reference_logprobs(logits):
    # Independent hand-specified supports, target coordinates and normalization.
    return torch.stack(
        [(logits[0, 1 + t, row] / 0.7).log_softmax(0)[row.index(TOKENS[2 + t])] for t, row in enumerate(ROWS)]
    )


def assert_scoring(received):
    args = replay_args()
    with cpu_actor_runtime(args):
        packed = data.get_batch(
            data.DataIterator(received, micro_batch_size=1),
            list(received),
            pad_multiplier=1,
            pack_device=torch.device("cpu"),
        )
        logits = torch.arange(64).reshape(1, 8, 8).float().sin().requires_grad_()
        reference = logits.detach().clone().requires_grad_()
        _, result = loss.get_log_probs_and_entropy(logits, args=args, use_rollout_sampling_replay=True, **packed)
        actual = result["log_probs"][0]
        expected = reference_logprobs(reference)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert torch.equal(actual[2:4], torch.zeros(2))
        # Nonzero observation weights prove singleton gradients vanish without masking.
        weights = torch.tensor([1.0, -0.5, 1.0, 1.0, 2.0, 0.25])
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(logits.grad, reference.grad, atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(logits.grad[0, 3:5]) == 0
        assert torch.count_nonzero(logits.grad[0, 1]) > 0


def test_sampling_replay_agentic_two_turn_alignment_logprob_and_gradient():
    sample = asyncio.run(two_turn_sample())
    assert_alignment(sample)
    td = convert_samples_to_train_data(replay_args(), [sample])
    received = {key: list(value) if value.is_nested else value.tolist() for key, value in td.items()}
    assert_scoring(received)


def transfer_and_score(payload, ray_temp_dir):
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf

    from relax.utils.data.stream_dataloader import get_data_from_transfer_queue

    ray.init(
        address="local",
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=80 * 1024**2,
        _temp_dir=ray_temp_dir,
    )
    try:
        tq.init(
            OmegaConf.create(
                {"backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"num_data_storage_units": 1}}}
            )
        )
        sample = TrainingFieldArtifact(sample_payload=payload).to_sample()
        assert_alignment(sample)
        client = tq.get_client()
        args = replay_args()
        client.put(convert_samples_to_train_data(args, [sample]), partition_id="agentic_sampling")
        received, meta = get_data_from_transfer_queue(
            args=args,
            tq_client=client,
            data_fields=build_data_fields(args),
            batch_size=1,
            partition_id="agentic_sampling",
            task_name="actor",
            sampling_config={},
            batch_index=0,
            per_rank_fetch=True,
            synchronize_per_rank_fetch=False,
            post_process=False,
        )
        assert meta.size == 1
        assert received["loss_masks"][0].tolist() == [1, 1, 0, 0, 1, 1]
        assert received["rollout_sampling_mask_offsets"][0].tolist() == [0, 2, 5, 6, 7, 9, 12]
        assert_scoring(received)
    finally:
        tq.close()
        ray.shutdown()


def test_sampling_replay_agentic_real_transfer_queue_and_serialization():
    sample = asyncio.run(two_turn_sample())
    payload = TrainingFieldArtifact.from_sample(sample).sample_payload
    with tempfile.TemporaryDirectory(prefix="replay-tq-") as directory:
        process = multiprocessing.get_context("spawn").Process(target=transfer_and_score, args=(payload, directory))
        process.start()
        try:
            process.join(timeout=120)
            assert not process.is_alive(), "Agentic TransferQueue timed out"
            assert process.exitcode == 0, "Agentic TransferQueue/scoring worker failed"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()


@pytest.mark.parametrize("replay,evaluation", [(False, False), (True, True)])
async def test_sampling_replay_agentic_off_and_eval_preserve_raw_logprobs(replay, evaluation, monkeypatch):
    calls = []

    async def post(_url, payload, **_kwargs):
        calls.append(payload)
        return output([3], [[3, 1]], [[0.6, 0.4]])

    monkeypatch.setattr(runtime, "post", post)
    result = await backend(replay_args(use_rollout_sampling_replay=replay)).generate(
        input_ids=[6, 0],
        sampling_params={"max_new_tokens": 1},
        session_id="s",
        request_id="r",
        return_logprob=True,
        collect_sampling_replay=not evaluation,
        collect_score_centering=not evaluation,
    )
    assert "return_sampling_mask" not in calls[0]
    assert "sampling_logprobs_mode" not in calls[0]
    assert result.new_sampling_mask is None
    assert result.new_log_probs == [-9.0]
    forest, prompt = forest_with_prompt(replay=False)
    leaf = forest.append_resp(
        parent_state_hash=prompt.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "D"}],
        token_delta=result.new_tokens,
        logprob_delta=result.new_log_probs,
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=Tokenizer())
    assert sample.rollout_sampling_mask is None
    assert sample.rollout_log_probs == [-9.0]


@pytest.mark.parametrize("failure", ["missing", "misaligned", "sanitized", "media", "sampling_override"])
async def test_sampling_replay_agentic_adapter_rejects_invalid_capture(failure, monkeypatch):
    response = output([3], [[3, 1]], [[0.6, 0.4]])
    if failure == "missing":
        del response["meta_info"]["output_token_sampling_mask"]
    elif failure == "misaligned":
        response["meta_info"]["output_token_sampling_mask"].append([3])
    adapter = backend(replay_args())
    if failure == "sanitized":
        adapter.tokenizer = SimpleNamespace(pad_token_id=0, image_token_id=3)
    calls = []

    async def post(_url, payload, **_kwargs):
        calls.append(payload)
        return response

    monkeypatch.setattr(runtime, "post", post)
    sampling = {"max_new_tokens": 1}
    if failure == "sampling_override":
        sampling["temperature"] = 1.0
    with pytest.raises(ValueError, match="sampling replay"):
        await adapter.generate(
            input_ids=[6, 0],
            sampling_params=sampling,
            session_id="s",
            request_id="r",
            image_data=["media"] if failure == "media" else None,
        )
    assert len(calls) == (0 if failure in {"media", "sampling_override"} else 1)


async def test_sampling_replay_agentic_abort_resume_and_http499_preserve_support(monkeypatch):
    forest, prompt = forest_with_prompt()
    shard, session = shard_and_session(replay_args(), forest)
    ir = request(prompt)
    responses = [output([3], [[3, 1]], [[0.6, 0.4]], "abort"), "499", output([7], [[7, 0, 4]], [[0.2, 0.3, 0.5]])]
    prefixes = []

    async def post(_url, payload, **_kwargs):
        prefixes.append(payload["input_ids"])
        response = responses.pop(0)
        if response == "499":
            req = httpx.Request("POST", "http://test/generate")
            res = httpx.Response(499, request=req, json={"error": {"message": "Request r was aborted"}})
            raise httpx.HTTPStatusError("aborted", request=req, response=res)
        return response

    monkeypatch.setattr(runtime, "post", post)
    for _ in range(3):
        result = await shard._generation_backend.generate(
            input_ids=[6, 0] + ir.pending_token_delta,
            sampling_params={"max_new_tokens": 2 - len(ir.pending_token_delta)},
            session_id="s",
            request_id="r",
        )
        shard._apply_generate_result(ir, result)
    assert prefixes == [[6, 0], [6, 0, 3], [6, 0, 3]]
    assert [len(mask) for mask in ir.pending_sampling_masks] == [1, 0, 1]
    shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id["request"], tokenizer=Tokenizer()
    )
    assert sample.tokens == [6, 0, 3, 7]
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 1, 7, 0, 4]
    assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 5]
    assert sample.rollout_log_probs == [math.log(0.6), math.log(0.2)]


@pytest.mark.parametrize("failure", ["missing", "support_length", "logprob_length"])
async def test_sampling_replay_agentic_invalid_attempt_preserves_accepted_prefix(failure):
    forest, prompt = forest_with_prompt()
    shard, _session = shard_and_session(replay_args(), forest)
    ir = request(prompt)
    good = runtime.BackendGenerateResult(
        [3], [-0.5], "abort", {}, 0.1, new_sampling_mask=RolloutSamplingMask.from_rows([[3, 1]])
    )
    shard._apply_generate_result(ir, good)
    bad = runtime.BackendGenerateResult(
        [7], [-1.0], "stop", {}, 0.1, new_sampling_mask=RolloutSamplingMask.from_rows([[7, 0]])
    )
    if failure == "missing":
        bad.new_sampling_mask = None
    elif failure == "support_length":
        bad.new_sampling_mask = RolloutSamplingMask.from_rows([[7], [0]])
    else:
        bad.new_log_probs = []
    with pytest.raises(ValueError, match="sampling replay"):
        shard._apply_generate_result(ir, bad)
    assert ir.pending_token_delta == [3]
    assert ir.pending_logprob_delta == [-0.5]
    assert len(ir.pending_sampling_masks) == 1
    with pytest.raises(ValueError, match="sampling replay"):
        forest.append_resp(
            parent_state_hash=prompt.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": "H"}],
            token_delta=bad.new_tokens,
            logprob_delta=bad.new_log_probs,
            sampling_mask_delta=bad.new_sampling_mask,
        )
    assert len(forest.nodes_by_hash) == 2


def test_sampling_replay_agentic_multiturn_branch_exports_own_serialized_support():
    forest, parent = forest_with_prompt()

    def response(parent, token):
        return forest.append_resp(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": chr(65 + token)}],
            token_delta=[token],
            logprob_delta=[math.log(0.5)],
            sampling_mask_delta=RolloutSamplingMask.from_rows([[token, 0]]),
        )

    def observation(parent, tokens):
        return forest.append_obs(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "user", "content": str(tokens)}],
            train_token_delta=tokens,
            rollout_token_delta=list(tokens),
        )

    first = response(parent, 3)
    second = response(observation(first, [4, 6]), 7)
    branch_point = observation(second, [1, 6, 4])
    leaves = [response(branch_point, token) for token in (2, 5)]
    samples = [forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=Tokenizer()) for leaf in leaves]
    for sample, token in zip(samples, (2, 5), strict=True):
        assert sample.tokens == [6, 0, 3, 4, 6, 7, 1, 6, 4, token]
        assert sample.loss_mask == [1, 0, 0, 1, 0, 0, 0, 1]
        assert sample.rollout_log_probs == [math.log(0.5), 0.0, 0.0, math.log(0.5), 0.0, 0.0, 0.0, math.log(0.5)]
        assert sample.rollout_sampling_mask.ids.tolist() == [3, 0, 4, 6, 7, 0, 1, 6, 4, token, 0]
        assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 3, 4, 6, 7, 8, 9, 11]
        assert sample.response_length == len(sample.rollout_sampling_mask) == 8
        assert sample.custom_advantage == [[0, 1], [3, 4], [7, 8]]
    restored = Sample.from_dict(json.loads(json.dumps(samples[0].to_dict())))
    compact = TrainingFieldArtifact.from_sample(samples[0]).to_sample()
    for sample in (restored, compact):
        assert torch.equal(sample.rollout_sampling_mask.ids, samples[0].rollout_sampling_mask.ids)
        assert sample.loss_mask == samples[0].loss_mask
    for sample in (samples[0], restored, compact):
        sample.rollout_sampling_mask.ids[0] = 99
        sample.rollout_sampling_mask.offsets[1] = 1
        sample.rollout_log_probs[0] = -99.0
    assert samples[1].rollout_sampling_mask.ids[0] == 3
    assert samples[1].rollout_sampling_mask.offsets[1] == 2
    assert first.sampling_mask_delta.ids.tolist() == [3, 0]
    assert first.logprob_delta == [math.log(0.5)]
    tail = forest.build_sample(leaf_state_hash=branch_point.state_hash, tokenizer=Tokenizer())
    assert tail.status == Sample.Status.TRUNCATED
    assert tail.loss_mask[-3:] == [0, 0, 0]
    assert tail.rollout_sampling_mask.ids.tolist() == [3, 0, 4, 6, 7, 0, 1, 6, 4]


async def test_sampling_replay_agentic_score_centering_keeps_candidate_probabilities(monkeypatch):
    response = output([3], [[3, 1]], [[0.6, 0.4]])
    response["meta_info"]["output_token_logprobs"] = [[math.log(0.06), 3, None]]
    response["meta_info"]["output_top_logprobs"] = [[[math.log(0.06), 3, None], [math.log(0.04), 1, None]]]

    async def post(_url, _payload, **_kwargs):
        return response

    monkeypatch.setattr(runtime, "post", post)
    adapter = backend(
        replay_args(advantage_estimator="score_centering", rollout_top_logprobs_num=2, score_centering_vocab_size=8)
    )
    result = await adapter.generate(
        input_ids=[6, 0],
        sampling_params={"max_new_tokens": 1, "temperature": 0.7, "top_p": 0.8, "top_k": 4},
        session_id="s",
        request_id="r",
    )
    assert result.new_log_probs == [math.log(0.6)]
    assert result.new_sampling_mask.ids.tolist() == [3, 1]
    np.testing.assert_array_equal(result.new_topk_token_ids, [[3, 1]])
    np.testing.assert_allclose(result.new_topk_log_probs, [[math.log(0.6), math.log(0.4)]], atol=1e-7)


def test_sampling_replay_agentic_policy_loss_masks_observations():
    sample = asyncio.run(two_turn_sample())
    args = replay_args(
        true_on_policy_mode=True,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_rollout_logprobs=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )
    td = convert_samples_to_train_data(args, [sample])
    received = {key: list(value) if value.is_nested else value.tolist() for key, value in td.items()}
    received["advantages"] = [torch.tensor([1.0, -0.5, 100.0, -70.0, 2.0, 0.25])]
    with cpu_actor_runtime(args):
        packed = data.get_batch(
            data.DataIterator(received, micro_batch_size=1),
            list(received),
            pad_multiplier=1,
            pack_device=torch.device("cpu"),
        )
        reduce = loss.get_sum_of_sample_mean(
            packed["total_lengths"],
            packed["response_lengths"],
            packed["loss_masks"],
            sample_denoms=packed.get("sample_index_mask_sums"),
        )
        logits = torch.arange(64).reshape(1, 8, 8).float().sin().requires_grad_()
        reference = logits.detach().clone().requires_grad_()
        objective, _ = loss.policy_loss_function(args, packed, logits, reduce)
        objective.backward()
        active = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        advantages = received["advantages"][0]
        torch.testing.assert_close(objective, -(advantages * active).sum() / 4)
        (-(reference_logprobs(reference) * advantages * active).sum() / 4).backward()
        torch.testing.assert_close(logits.grad, reference.grad, atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(logits.grad[0, 3:5]) == 0


def canonical_response(
    forest, parent, *, tokens=(3, 7), rows=((3, 1), (2, 7, 0)), logprobs=(-0.5, -1.0), **candidates
):
    return forest.append_resp(
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "same canonical response"}],
        token_delta=list(tokens),
        logprob_delta=list(logprobs),
        sampling_mask_delta=RolloutSamplingMask.from_rows(rows),
        export_metadata_patch={"marker": list(logprobs)},
        **candidates,
    )


@pytest.mark.parametrize(
    "incoming",
    [
        {"tokens": (1, 7)},
        {"rows": ((3, 2), (2, 7, 0))},
        {"rows": ((1, 3), (2, 7, 0))},
        {"rows": ((3, 1, 2), (7, 0))},
        {"logprobs": (-0.5, -0.8)},
        {"logprobs": (-0.500006, -1.0)},
    ],
    ids=["tokens", "support_ids", "support_order", "support_boundaries", "behavior_logprob", "relative_tolerance"],
)
def test_sampling_replay_agentic_canonical_response_conflict_rejects_without_mutation(incoming):
    forest, prompt = forest_with_prompt()
    first = canonical_response(forest, prompt)
    leaves = forest.export_leaf_hashes()
    with pytest.raises(ValueError, match="sampling replay metadata conflict for canonical response"):
        canonical_response(forest, prompt, **incoming)
    assert forest.nodes_by_hash[first.state_hash] is first
    assert forest.export_leaf_hashes() == leaves
    assert len(forest.nodes_by_hash) == 3
    sample = forest.build_sample(leaf_state_hash=first.state_hash, tokenizer=Tokenizer())
    assert sample.tokens == [6, 0, 3, 7]
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 1, 2, 7, 0]
    assert sample.rollout_sampling_mask.offsets.tolist() == [0, 2, 5]
    assert sample.rollout_log_probs == [-0.5, -1.0]
    assert sample.metadata["marker"] == [-0.5, -1.0]


@pytest.mark.parametrize("logprobs", [(-0.5, -1.0), (-0.500004, -1.000008), (-0.500004, -0.999992)])
def test_sampling_replay_agentic_canonical_response_reuses_equal_or_close_logprobs(logprobs):
    forest, prompt = forest_with_prompt()
    first = canonical_response(forest, prompt)
    reused = canonical_response(forest, prompt, logprobs=logprobs)
    assert reused is first
    assert len(forest.nodes_by_hash) == 3
    assert forest.export_leaf_hashes() == [first.state_hash]
    sample = forest.build_sample(leaf_state_hash=reused.state_hash, tokenizer=Tokenizer())
    assert sample.rollout_log_probs == [-0.5, -1.0]
    assert sample.metadata["marker"] == [-0.5, -1.0]
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 1, 2, 7, 0]


@pytest.mark.parametrize("logprobs,accepted", [((-2e-7, -1.0), True), ((-2e-6, -1.0), False)])
def test_sampling_replay_agentic_canonical_response_logprob_absolute_tolerance(logprobs, accepted):
    forest, prompt = forest_with_prompt()
    first = canonical_response(forest, prompt, logprobs=(-1e-7, -1.0))
    if accepted:
        assert canonical_response(forest, prompt, logprobs=logprobs) is first
    else:
        with pytest.raises(ValueError, match="sampling replay metadata conflict"):
            canonical_response(forest, prompt, logprobs=logprobs)
    assert first.logprob_delta == [-1e-7, -1.0]


def test_sampling_replay_agentic_canonical_response_off_preserves_legacy_reuse():
    forest, prompt = forest_with_prompt(replay=False)
    first = canonical_response(forest, prompt)
    reused = canonical_response(forest, prompt, tokens=(1, 7), rows=((1, 2), (7,)), logprobs=(-3.0, -2.0))
    assert reused is first
    sample = forest.build_sample(leaf_state_hash=reused.state_hash, tokenizer=Tokenizer())
    assert sample.tokens == [6, 0, 3, 7]
    assert sample.rollout_log_probs == [-0.5, -1.0]
    assert sample.rollout_sampling_mask is None


class ChatTokenizer(Tokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        text = "".join(f"{message['role']}:{message['content']};" for message in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def encode(self, text, **_kwargs):
        return [ord(character) % 8 for character in text]


async def test_sampling_replay_agentic_observation_compiler_keeps_template_tokens():
    from concurrent.futures import ThreadPoolExecutor

    args = replay_args(rollout_max_context_len=None)
    forest = SessionForest.create_empty(session_id="compiled", use_rollout_sampling_replay=True)
    shard, session = shard_and_session(args, forest)
    session.rollout_id = 0
    session.session_id = "compiled"
    session.group = SimpleNamespace(rollout_mode="train")
    with ThreadPoolExecutor(max_workers=1) as executor:
        shard._generation_backend.compiler = runtime.SGLangMessageCompiler(
            tokenizer=ChatTokenizer(), processor=None, cpu_executor=executor
        )
        prompt, _ = await shard._append_observation(
            session=session,
            forest=forest,
            parent=forest.nodes_by_hash[forest.root_state_hash],
            messages_delta=[{"role": "user", "content": "prompt"}],
            tools=[],
            chat_template_kwargs={},
        )
        first = forest.append_resp(
            parent_state_hash=prompt.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": "DH"}],
            token_delta=[3, 7],
            logprob_delta=[-0.5, 0.0],
            sampling_mask_delta=RolloutSamplingMask.from_rows([[3, 1], [7]]),
        )
        observation, _ = await shard._append_observation(
            session=session,
            forest=forest,
            parent=first,
            messages_delta=[{"role": "tool", "tool_call_id": "call", "content": "result"}],
            tools=[],
            chat_template_kwargs={},
        )
    expected_obs = [ord(character) % 8 for character in "tool:result;assistant:"]
    assert observation.train_token_delta == observation.rollout_token_delta == expected_obs
    leaf = forest.append_resp(
        parent_state_hash=observation.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "C"}],
        token_delta=[2],
        logprob_delta=[-0.7],
        sampling_mask_delta=RolloutSamplingMask.from_rows([[2, 5]]),
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=Tokenizer())
    suffix = [3, 7, *expected_obs, 2]
    assert sample.tokens[-len(suffix) :] == suffix
    assert sample.loss_mask == [1, 1, *([0] * len(expected_obs)), 1]
    assert sample.rollout_log_probs == [-0.5, 0.0, *([0.0] * len(expected_obs)), -0.7]
    assert sample.rollout_sampling_mask.ids.tolist() == [3, 1, 7, *expected_obs, 2, 5]
    assert sample.response_length == len(sample.rollout_sampling_mask) == len(suffix)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("change", ["probability", "ids", "coverage", "missing", "close"])
def test_forest_candidate_identity_is_checked_before_reuse(replay, change):
    forest, prompt = forest_with_prompt(replay=replay)
    candidates = dict(
        topk_token_ids_delta=np.array([[3, 1, 2]], dtype=np.int32),
        topk_logprob_delta=np.log([[0.5, 0.3, 0.2]]),
        topk_support_complete_delta=np.array([True]),
    )
    event = dict(tokens=(3,), rows=((3, 1, 2),), logprobs=(math.log(0.5),))
    first = canonical_response(forest, prompt, **event, **candidates)
    incoming = {key: value.copy() for key, value in candidates.items()}
    if change == "close":
        incoming["topk_logprob_delta"][0, 1] += 1e-7
        assert canonical_response(forest, prompt, **event, **incoming) is first
        return
    if change == "probability":
        incoming["topk_logprob_delta"] = np.log([[0.5, 0.35, 0.15]])
    elif change == "ids":
        incoming["topk_token_ids_delta"][0] = [3, 2, 1]
    elif change == "coverage":
        incoming["topk_support_complete_delta"][0] = False
    else:
        incoming = {}
    leaves = forest.export_leaf_hashes()
    with pytest.raises(ValueError, match="metadata conflict"):
        canonical_response(forest, prompt, **event, **incoming)
    assert forest.export_leaf_hashes() == leaves
    assert forest.nodes_by_hash[first.state_hash] is first
    sample = forest.build_sample(leaf_state_hash=first.state_hash, tokenizer=Tokenizer())
    np.testing.assert_allclose(sample.rollout_topk_log_probs, candidates["topk_logprob_delta"])
