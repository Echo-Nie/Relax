from types import SimpleNamespace

import pytest

from relax.distributed.ray.rollout import _compute_spec_metrics
from relax.utils.types import Sample


def _args():
    return SimpleNamespace(sglang_speculative_algorithm="EAGLE")


def _generation(
    request_id: str,
    *,
    accepted: int | None = None,
    proposed: int | None = None,
    verify: int | None = None,
    completion: int | None = None,
    resp_state_hash: str = "state",
) -> dict:
    generation = {
        "request_id": request_id,
        "resp_state_hash": resp_state_hash,
    }
    if accepted is not None:
        generation["spec_accept_token_num"] = accepted
    if proposed is not None:
        generation["spec_draft_token_num"] = proposed
    if verify is not None:
        generation["spec_verify_ct"] = verify
    if completion is not None:
        generation["completion_token_num"] = completion
    return generation


def _agentic_sample(session_id: str, generations: list[dict]) -> Sample:
    return Sample(
        session_id=session_id,
        metadata={
            "agentic_trace": {
                "session_id": session_id,
                "spec_generations": generations,
            }
        },
    )


def test_spec_metrics_weight_raw_counters_before_division() -> None:
    samples = [
        _agentic_sample(
            "session",
            [_generation("r1", accepted=1, proposed=2, verify=2, completion=3)],
        ),
        _agentic_sample(
            "session",
            [_generation("r2", accepted=9, proposed=10, verify=4, completion=9)],
        ),
    ]

    metrics = _compute_spec_metrics(_args(), samples)

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(12 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_deduplicate_shared_generation_within_session() -> None:
    shared = _generation("A", accepted=1, proposed=2, verify=1, completion=2)
    sample_ab = _agentic_sample(
        "session",
        [
            shared,
            _generation("B", accepted=2, proposed=4, verify=2, completion=3),
        ],
    )
    sample_ac = _agentic_sample(
        "session",
        [
            shared,
            _generation("C", accepted=9, proposed=10, verify=3, completion=6),
        ],
    )

    metrics = _compute_spec_metrics(_args(), [sample_ab, sample_ac])

    assert metrics["spec_accept_rate"] == pytest.approx(12 / 16)
    assert metrics["spec_accept_length"] == pytest.approx(11 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_scope_request_id_by_session() -> None:
    first = _agentic_sample(
        "session-1",
        [_generation("same-request", accepted=1, proposed=2)],
    )
    second = _agentic_sample(
        "session-2",
        [_generation("same-request", accepted=9, proposed=10)],
    )

    metrics = _compute_spec_metrics(_args(), [first, second])

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_rate_coverage"] == 1.0


def test_spec_metrics_report_missing_and_explicit_zero_coverage() -> None:
    sample = _agentic_sample(
        "session",
        [
            _generation("explicit-zero", accepted=0, proposed=10),
            _generation("missing"),
            _generation("zero-denominator", accepted=0, proposed=0),
        ],
    )

    metrics = _compute_spec_metrics(_args(), [sample])

    assert metrics["spec_accept_rate"] == 0.0
    assert metrics["spec_accept_rate_coverage"] == pytest.approx(2 / 3)
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_length_coverage"] == 0.0


def test_spec_metrics_do_not_fabricate_rate_for_zero_denominator() -> None:
    sample = _agentic_sample(
        "session",
        [_generation("zero-denominator", accepted=0, proposed=0, verify=0, completion=0)],
    )

    metrics = _compute_spec_metrics(_args(), [sample])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_empty_input_has_only_zero_coverage() -> None:
    metrics = _compute_spec_metrics(_args(), [])

    assert metrics == {
        "spec_accept_rate_coverage": 0.0,
        "spec_accept_length_coverage": 0.0,
    }


def test_spec_metrics_legacy_samples_use_weighted_fallback() -> None:
    first = Sample()
    first.spec_info = Sample.SpecInfo(
        spec_accept_token_num=1,
        spec_draft_token_num=2,
        spec_verify_ct=2,
        completion_token_num=3,
    )
    second = Sample()
    second.spec_info = Sample.SpecInfo(
        spec_accept_token_num=9,
        spec_draft_token_num=10,
        spec_verify_ct=4,
        completion_token_num=9,
    )

    metrics = _compute_spec_metrics(_args(), [first, second])

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(12 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_legacy_zero_defaults_are_treated_as_uncovered() -> None:
    serialized = Sample().to_dict()
    serialized["spec_info"] = {
        "spec_accept_token_num": 0,
        "spec_draft_token_num": 0,
        "spec_verify_ct": 0,
        "completion_token_num": 0,
    }
    legacy = Sample.from_dict(serialized)

    metrics = _compute_spec_metrics(_args(), [legacy])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_rate_coverage"] == 0.0
    assert metrics["spec_accept_length_coverage"] == 0.0


def test_spec_metrics_disabled_returns_no_metrics() -> None:
    args = SimpleNamespace(sglang_speculative_algorithm=None)

    assert _compute_spec_metrics(args, []) == {}
