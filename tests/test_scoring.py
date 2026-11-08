"""Tests for rr.scoring (Phase 0). Skipped until implemented."""

import pytest

from rr.data import MCQuestion
from rr.scoring import format_mc_prompt


def test_format_mc_prompt_ends_with_answer_cue() -> None:
    question = MCQuestion("toy", "What is 2 + 2?", ("3", "4", "5", "6"), 1)

    block = format_mc_prompt(question)

    assert block == "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"


def test_format_mc_prompt_with_answer_appends_correct_label() -> None:
    question = MCQuestion("toy", "What is 2 + 2?", ("3", "4", "5", "6"), 1)

    assert format_mc_prompt(question, include_answer=True).endswith("\nAnswer: B")


def test_format_mc_prompt_rejects_label_count_mismatch() -> None:
    question = MCQuestion("toy", "Q?", ("x", "y"), 0)

    with pytest.raises(ValueError, match="labels"):
        format_mc_prompt(question)


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_score_mc_prediction_is_argmax_of_option_logprobs() -> None:
    """With a tiny randomly initialised causal LM (no downloads), the returned
    prediction equals argmax(option_logprobs), is_correct matches the comparison
    with correct_index, and correct_logprob equals option_logprobs[correct_index]."""


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_score_mc_is_deterministic() -> None:
    """Scoring the same prompt twice gives identical log-probs (no sampling)."""
