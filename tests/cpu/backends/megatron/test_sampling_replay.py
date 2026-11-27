# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real scoring/autograd; isolate only distributed runtime and model
execution."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron import data, loss
from relax.utils.sampling_mask import SAMPLING_MASK_FIELDS, RolloutSamplingMask


ROWS = [[3, 1], [7, 0, 4], [2], [5, 1, 6]]
TOKENS = torch.tensor([6, 0, 3, 7, 2, 5])


def scoring_args(**overrides):
    return Namespace(
        **{
            "qkv_format": "thd",
            "rollout_temperature": 0.7,
            "allgather_cp": False,
            "advantage_estimator": "grpo",
            "loss_type": "policy_loss",
            "entropy_coef": 0.0,
            "log_probs_chunk_size": 2,
            "use_rollout_sampling_replay": True,
            "opd_token_selection": "student_sampled",
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def parallel_runtime(monkeypatch):
    for name, value in {
        "get_context_parallel_world_size": 1,
        "get_context_parallel_rank": 0,
        "get_tensor_model_parallel_world_size": 1,
        "get_tensor_model_parallel_rank": 0,
        "get_tensor_model_parallel_group": None,
        "is_pipeline_last_stage": True,
    }.items():
        monkeypatch.setattr(loss.mpu, name, lambda value=value: value)
    monkeypatch.setattr(data, "get_args", lambda: scoring_args())


def score(logits, args=None, rows=ROWS, **overrides):
    mask = RolloutSamplingMask.from_rows(rows)
    kwargs = {
        "args": args or scoring_args(),
        "unconcat_tokens": [TOKENS],
        "total_lengths": [6],
        "response_lengths": [4],
        "use_rollout_sampling_replay": True,
        "rollout_sampling_mask_ids": [mask.ids],
        "rollout_sampling_mask_offsets": [mask.offsets],
        **overrides,
    }
    return loss.get_log_probs_and_entropy(logits, **kwargs)[1]


def independent_reference(logits, rows, temperature):
    # Target t is predicted at prompt_length - 1 + t. Normalize only saved IDs.
    return torch.stack(
        [
            (logits[0, 1 + t, support].float() / temperature).log_softmax(0)[support.index(int(TOKENS[2 + t]))]
            for t, support in enumerate(rows)
        ]
    )


@pytest.mark.parametrize("entropy_coef", [0.0, 0.2])
def test_sampling_replay_logprob_and_gradient_match_reference(entropy_coef):
    args = scoring_args(entropy_coef=entropy_coef)
    logits = torch.arange(48).reshape(1, 6, 8).float().sin().requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    result = score(logits, args, with_entropy=True)
    expected = independent_reference(reference, ROWS, args.rollout_temperature)
    logp = (reference[0, 1:5] / args.rollout_temperature).log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(result["log_probs"][0], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["entropy"][0], entropy, atol=1e-6, rtol=1e-6)
    weights = torch.tensor([1.0, -0.5, 2.0, 0.25])
    objective = (result["log_probs"][0] * weights).sum()
    expected_objective = (expected * weights).sum()
    if entropy_coef:
        objective = objective + entropy_coef * result["entropy"][0].sum()
        expected_objective = expected_objective + entropy_coef * entropy.sum()
    objective.backward()
    expected_objective.backward()
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-6, rtol=1e-6)
    if not entropy_coef:
        for t, support in enumerate(ROWS):
            outside = [i for i in range(8) if i not in support]
            assert torch.count_nonzero(logits.grad[0, 1 + t, outside]) == 0


def test_sampling_replay_full_support_and_disabled_match_original():
    inputs = torch.arange(48).reshape(1, 6, 8).float().cos()
    original = score(inputs, use_rollout_sampling_replay=False)
    full = score(inputs, rows=[list(range(8))] * 4)
    disabled = score(
        inputs, rollout_sampling_mask_ids=None, rollout_sampling_mask_offsets=None, use_rollout_sampling_replay=False
    )
    torch.testing.assert_close(full["log_probs"][0], original["log_probs"][0], atol=1e-6, rtol=1e-6)
    assert torch.equal(disabled["log_probs"][0], original["log_probs"][0])


def test_sampling_replay_rejects_absent_target_before_primitive_can_retain_it():
    with pytest.raises(RuntimeError, match="absent from rollout sampling support"):
        score(torch.zeros(1, 6, 8), rows=[[0], *ROWS[1:]])


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_sampling_mask_ids": None},
        {"rollout_sampling_mask_offsets": []},
        {"rollout_sampling_mask_offsets": [torch.tensor([0, 2])]},
    ],
)
def test_sampling_replay_rejects_missing_or_misaligned_microbatch(overrides):
    with pytest.raises(ValueError, match="sampling replay"):
        score(torch.zeros(1, 6, 8), **overrides)


def test_sampling_replay_microbatch_permutation_preserves_cpu_csr():
    a, b = RolloutSamplingMask.from_rows(ROWS), RolloutSamplingMask.from_rows([[6, 0]])
    batch = {
        "tokens": [TOKENS, torch.tensor([1, 6])],
        "total_lengths": [6, 2],
        "response_lengths": [4, 1],
        "loss_masks": [torch.ones(4), torch.ones(1)],
        SAMPLING_MASK_FIELDS[0]: [a.ids, b.ids],
        SAMPLING_MASK_FIELDS[1]: [a.offsets, b.offsets],
    }
    iterator = data.DataIterator(batch, micro_batch_indices=[[1, 0]])
    packed = data.get_batch(iterator, list(batch), pad_multiplier=1, pack_device=torch.device("cpu"))
    assert packed["response_lengths"] == [1, 4]
    assert packed[SAMPLING_MASK_FIELDS[0]][0].tolist() == [6, 0]
    assert packed[SAMPLING_MASK_FIELDS[1]][1].tolist() == [0, 2, 5, 6, 9]
    assert all(t.device.type == "cpu" for field in SAMPLING_MASK_FIELDS for t in packed[field])
    logits = torch.arange(64).reshape(1, 8, 8).float().sin()
    _, result = loss.get_log_probs_and_entropy(logits, args=scoring_args(), use_rollout_sampling_replay=True, **packed)
    expected = (logits[0, 0, [6, 0]] / 0.7).log_softmax(0)[0]
    torch.testing.assert_close(result["log_probs"][0][0], expected)
    torch.testing.assert_close(result["log_probs"][1], independent_reference(logits[:, 2:], ROWS, 0.7))


@pytest.mark.parametrize("prefix", ["", "ref_", "teacher_"])
def test_sampling_replay_forward_only_actor_gate_and_microbatch_wiring(monkeypatch, prefix):
    from relax.backends.megatron import actor, model

    logits = torch.arange(48).reshape(1, 6, 8).float().sin()

    class TinyModel(torch.nn.Module):
        def forward(self, **kwargs):
            return logits

    # Isolate only model execution and Megatron pipeline scheduling. Run real
    # get_batch, forward_only callback binding, CSR expansion, and scoring.
    def schedule(**kwargs):
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        return [callback(output)[1]]

    monkeypatch.setattr(model, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(model, "get_model_config", lambda _: SimpleNamespace(timers=None))
    monkeypatch.setattr(model, "straggler_timers", lambda *args: None)
    args = scoring_args(
        data_pad_size_multiplier=1,
        micro_batch_size=1,
        seq_length=6,
        use_rollout_entropy=True,
        custom_megatron_before_log_prob_hook_path=None,
    )
    mask = RolloutSamplingMask.from_rows(ROWS)
    batch = {
        "tokens": [TOKENS],
        "total_lengths": [6],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        SAMPLING_MASK_FIELDS[0]: [mask.ids],
        SAMPLING_MASK_FIELDS[1]: [mask.offsets],
    }
    iterator = data.DataIterator(batch, micro_batch_size=1)
    instance = SimpleNamespace(args=args, model=[TinyModel()])
    result = actor.MegatronTrainRayActor.compute_log_prob(instance, [iterator], [1], store_prefix=prefix)
    if prefix:
        expected = (logits[0, 1:5] / 0.7).log_softmax(-1).gather(1, TOKENS[2:, None]).squeeze(1)
    else:
        expected = independent_reference(logits, ROWS, 0.7)
    torch.testing.assert_close(result[prefix + "log_probs"][0], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampling_replay_conditions_score_centering_candidates_and_preserves_entropy(dtype):
    args = scoring_args(advantage_estimator="score_centering", score_centering_vocab_size=8)
    logits = torch.arange(48).reshape(1, 6, 8).float().sin().to(dtype).requires_grad_()
    candidates = [torch.tensor([[3, 1], [7, 4], [2, -1], [5, 1]])]
    baseline = score(
        logits, args, use_rollout_sampling_replay=False, gather_topk_token_ids=candidates, with_entropy=True
    )
    replay = score(logits, args, gather_topk_token_ids=candidates, with_entropy=True)
    expected_head = torch.cat(
        [
            (logits[0, 1 + t, row].float() / args.rollout_temperature).log_softmax(0)[
                [row.index(i) for i in head if i >= 0]
            ]
            for t, (row, head) in enumerate(zip(ROWS, candidates[0].tolist(), strict=True))
        ]
    )
    head = replay["topk_log_probs"][0][candidates[0] >= 0]
    torch.testing.assert_close(head, expected_head, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(replay["entropy"][0], baseline["entropy"][0], rtol=0, atol=0)
    expected = independent_reference(logits, ROWS, 0.7)
    torch.testing.assert_close(replay["log_probs"][0], expected, atol=1e-6, rtol=1e-6)
    actual_grad = torch.autograd.grad(head.sum() + replay["log_probs"][0].sum(), logits)[0]
    expected_grad = torch.autograd.grad(expected_head.sum() + expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.01 if dtype == torch.bfloat16 else 1e-6, atol=1e-6)


def test_sampling_replay_policy_loss_uses_support_gradient():
    args = scoring_args(
        true_on_policy_mode=True,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_kl_loss=False,
        use_opd=False,
        use_rollout_logprobs=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )
    logits = torch.arange(48).reshape(1, 6, 8).float().sin().requires_grad_()
    ref = logits.detach().clone().requires_grad_()
    mask = RolloutSamplingMask.from_rows(ROWS)
    advantages = torch.tensor([1.0, -0.5, 2.0, 0.25])
    batch = {
        "advantages": [advantages],
        "unconcat_tokens": [TOKENS],
        "total_lengths": [6],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        SAMPLING_MASK_FIELDS[0]: [mask.ids],
        SAMPLING_MASK_FIELDS[1]: [mask.offsets],
    }
    objective, _ = loss.policy_loss_function(args, batch, logits, torch.mean)
    objective.backward()
    # At ratio=1 the PPO gradient is -A * grad(log pi), independently of clipping.
    expected = -(advantages * independent_reference(ref, ROWS, 0.7)).mean()
    expected.backward()
    torch.testing.assert_close(logits.grad, ref.grad, atol=1e-6, rtol=1e-6)


def test_sampling_replay_dynamic_cp_one_skips_static_allgather(monkeypatch):
    monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: 2)
    result = score(
        torch.arange(48).reshape(1, 6, 8).float().sin(),
        scoring_args(allgather_cp=True),
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )
    expected = independent_reference(torch.arange(48).reshape(1, 6, 8).float().sin(), ROWS, 0.7)
    torch.testing.assert_close(result["log_probs"][0], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_sampling_replay_bshd_zigzag_padding_and_empty_shard(monkeypatch, cp_rank):
    monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(loss.mpu, "get_context_parallel_rank", lambda: cp_rank)
    a = torch.arange(48).reshape(6, 8).float().sin()
    b = torch.arange(16).reshape(2, 8).float().cos()
    positions = [0, 1, 2, 9, 10, 11] if cp_rank == 0 else [3, 4, 5, 6, 7, 8]
    packed = torch.stack(
        [torch.stack([sample[p] if p < len(sample) else torch.ones(8) * 19 for p in positions]) for sample in (a, b)]
    )
    masks = [RolloutSamplingMask.from_rows(ROWS), RolloutSamplingMask.from_rows([[6, 0]])]
    _, result = loss.get_log_probs_and_entropy(
        packed,
        args=scoring_args(qkv_format="bshd"),
        unconcat_tokens=[TOKENS, torch.tensor([1, 6])],
        total_lengths=[6, 2],
        response_lengths=[4, 1],
        max_seq_lens=[12, 12],
        use_rollout_sampling_replay=True,
        rollout_sampling_mask_ids=[m.ids for m in masks],
        rollout_sampling_mask_offsets=[m.offsets for m in masks],
    )
    rows = [0, 1] if cp_rank == 0 else [2, 3]
    torch.testing.assert_close(
        result["log_probs"][0], independent_reference(a.unsqueeze(0), ROWS, 0.7)[rows], rtol=1e-6, atol=1e-6
    )
    if cp_rank == 0:
        torch.testing.assert_close(result["log_probs"][1], (b[0, [6, 0]] / 0.7).log_softmax(0)[:1])
    else:
        assert result["log_probs"][1].numel() == 0


def test_sampling_replay_rollout_metrics_exclude_csr(monkeypatch):
    captured = {}
    monkeypatch.setattr(data.mpu, "get_data_parallel_group", lambda **kwargs: None)
    monkeypatch.setattr(data.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data.dist, "all_reduce", lambda *args, **kwargs: None)

    def emit(name, args, rollout_id, values, *extra):
        captured.update(values)

    monkeypatch.setattr(data, "gather_log_data", emit)
    mask = RolloutSamplingMask.from_rows(ROWS)
    batch = {
        "tokens": [TOKENS],
        "total_lengths": [6],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        "rewards": [2.0],
        SAMPLING_MASK_FIELDS[0]: [mask.ids],
        SAMPLING_MASK_FIELDS[1]: [mask.offsets],
    }
    data.log_rollout_data(
        0,
        scoring_args(rollout_batch_size=1, n_samples_per_prompt=1, use_opd=False, ci_test=False, log_multi_turn=False),
        batch,
    )
    assert captured["rewards"] == 2.0
    assert not set(SAMPLING_MASK_FIELDS) & captured.keys()


@pytest.mark.parametrize("model_vocab,bad_id", [(8, 8), (6, 6)])
def test_replay_rejects_out_of_vocab_non_sampled_candidate(model_vocab, bad_id):
    args = scoring_args(sampling_replay_vocab_size=model_vocab)
    logits = torch.zeros(1, 2, 8)
    batch = dict(unconcat_tokens=[torch.tensor([0, 3])], total_lengths=[2], response_lengths=[1])
    valid = score(logits, args, rows=[[3, model_vocab - 1]], **batch)
    torch.testing.assert_close(valid["log_probs"][0], torch.tensor([0.5]).log())
    with pytest.raises(ValueError, match="support.*vocabulary"):
        score(logits, args, rows=[[3, bad_id]], **batch)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize(
    "row,head,q",
    [
        ([0], [0], [1.0]),
        ([0, 2], [0, 2], [0.4, 0.6]),
        ([0, 1, 2], [2], [0.3, 0.3, 0.4]),
    ],
)
def test_sc_replay_final_policy_gradient_matches_restricted_model(mode, row, head, q):
    # Train a reference SC policy whose entire vocabulary is exactly the saved support.
    args = scoring_args(
        advantage_estimator="score_centering",
        rollout_temperature=1.0,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_kl_loss=False,
        use_opd=False,
        use_rollout_logprobs=True,
        score_centering_is=mode,
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
    )
    logits = torch.tensor([[[1.0, 0.0, -1.0, 2.0], [0.0] * 4]], requires_grad=True)
    ref = logits.detach().clone().requires_grad_()
    mask = RolloutSamplingMask.from_rows([row])
    batch = {
        "advantages": [torch.ones(1)],
        "total_lengths": [2],
        "response_lengths": [1],
        "unconcat_tokens": [torch.tensor([0, 0])],
        "loss_masks": [torch.ones(1)],
        "rollout_sampling_mask_ids": [mask.ids],
        "rollout_sampling_mask_offsets": [mask.offsets],
        "rollout_log_probs": [torch.tensor([q[0]]).log()],
        "rollout_topk_token_ids": [torch.tensor([head])],
        "rollout_topk_log_probs": [torch.tensor([[q[row.index(i)] for i in head]]).log()],
        "rollout_topk_support_complete": [torch.tensor([len(row) == len(head)])],
    }
    args.score_centering_vocab_size = logits.size(-1)
    actual, _ = loss.policy_loss_function(args, batch, logits, torch.mean)
    args.use_rollout_sampling_replay = False
    args.score_centering_vocab_size = len(row)
    expected, _ = loss.policy_loss_function(
        args,
        {**batch, "rollout_topk_token_ids": [torch.tensor([[row.index(i) for i in head]])]},
        ref[..., row].contiguous(),
        torch.mean,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, ref.grad, atol=1e-6, rtol=1e-6)
    if len(row) == 1:
        assert logits.grad.count_nonzero() == 0
