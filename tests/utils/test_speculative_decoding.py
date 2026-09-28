from relax.utils.types import get_spec_token_counts


def test_get_spec_token_counts_preserves_missing_vs_zero() -> None:
    assert get_spec_token_counts({}) is None

    assert get_spec_token_counts(
        {
            "spec_num_correct_drafts": 0,
            "spec_num_proposed_drafts": 10,
        }
    ) == (0, 10)

    assert get_spec_token_counts(
        {
            "spec_num_correct_drafts": 0,
            "spec_num_proposed_drafts": 0,
        }
    ) == (0, 0)
