"""Tests for rr.scoring (Phase 0). Skipped until implemented."""

import pytest


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_format_mc_prompt_ends_with_answer_cue() -> None:
    """Target-style blocks end with the "Answer:" cue and list every option label;
    with include_answer=True the correct label follows the cue instead."""


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_score_mc_prediction_is_argmax_of_option_logprobs() -> None:
    """With a tiny randomly initialised causal LM (no downloads), the returned
    prediction equals argmax(option_logprobs), is_correct matches the comparison
    with correct_index, and correct_logprob equals option_logprobs[correct_index]."""


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_score_mc_is_deterministic() -> None:
    """Scoring the same prompt twice gives identical log-probs (no sampling)."""
