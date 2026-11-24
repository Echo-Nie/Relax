# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise production argument normalization and late YAML validation
together."""

import pytest

from relax.utils.arguments import apply_custom_config_overrides, relax_validate_args
from tests.cpu.utils.test_arguments_rloo import _rloo_args


SGLANG_ROLLOUT = "relax.engine.rollout.sglang_rollout.generate_rollout"
AGENTIC_ROLLOUT = "relax.agentic.rollout.generate_rollout"


def replay_args(tmp_path, *, agentic=True, replay=True, **overrides):
    return _rloo_args(
        **{
            "loss_type": "policy_loss",
            "advantage_estimator": "grpo",
            "use_opd": False,
            "use_rollout_sampling_replay": replay,
            "use_agentic_rollout": agentic,
            "rollout_function_path": SGLANG_ROLLOUT,
            "sglang_sampling_backend": "pytorch",
            "rollout_temperature": 0.7,
            "rollout_top_k": 4,
            "rollout_top_p": 0.8,
            "agent_command": "python agent.py",
            "agent_cwd": str(tmp_path),
            "agent_timeout": 60,
            "agent_env": [],
            "agentic_concurrency": None,
            "agentic_eval_concurrency": None,
            "agentic_program_admission": False,
            "agentic_session_lifecycle": False,
            **overrides,
        }
    )


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("enable_from_yaml", [False, True])
def test_sampling_replay_arguments_accept_standard_rollout_after_normalization(tmp_path, agentic, enable_from_yaml):
    args = replay_args(tmp_path, agentic=agentic, replay=not enable_from_yaml)
    # Run the real entry point: Agentic rewrites the path before replay validates it.
    relax_validate_args(args)
    if enable_from_yaml:
        config = tmp_path / "replay.yaml"
        config.write_text("use_rollout_sampling_replay: true\n")
        args.custom_config_path = str(config)
        apply_custom_config_overrides(args)
    assert args.use_rollout_sampling_replay is True
    assert args.rollout_function_path == (AGENTIC_ROLLOUT if agentic else SGLANG_ROLLOUT)
    if agentic:
        assert args.eval_function_path == AGENTIC_ROLLOUT
        assert args.apply_chat_template is False


@pytest.mark.parametrize(
    "agentic,override_path",
    [(True, SGLANG_ROLLOUT), (False, AGENTIC_ROLLOUT), (True, "custom.rollout"), (False, "custom.rollout")],
)
def test_sampling_replay_arguments_reject_unverified_effective_rollout(tmp_path, agentic, override_path):
    args = replay_args(tmp_path, agentic=agentic)
    relax_validate_args(args)
    config = tmp_path / "custom.yaml"
    config.write_text(f"rollout_function_path: {override_path}\n")
    args.custom_config_path = str(config)
    with pytest.raises(ValueError, match="sampling replay requires the standard"):
        apply_custom_config_overrides(args)


def test_sampling_replay_arguments_agentic_still_rejects_unsupported_sampler(tmp_path):
    args = replay_args(tmp_path, sglang_sampling_backend="flashinfer")
    with pytest.raises(ValueError, match="sampling replay requires --sglang-sampling-backend pytorch"):
        relax_validate_args(args)
