"""Tests for rr.routing.align (Phase 2). Skipped until implemented."""

import pytest


@pytest.mark.skip(reason="Phase 2: not implemented yet")
def test_aligned_positions_have_identical_token_ids() -> None:
    """For a zero-shot and a few-shot prompt of the same target question, the
    token ids at positions_a in prompt A equal those at positions_b in prompt B."""


@pytest.mark.skip(reason="Phase 2: not implemented yet")
def test_alignment_covers_the_whole_target_block_including_answer_cue() -> None:
    """The aligned positions run from the first token of the question through the
    final "Answer:" token, with no gaps."""


@pytest.mark.skip(reason="Phase 2: not implemented yet")
def test_alignment_handles_token_merging_at_example_boundary() -> None:
    """When the last example and the target question could merge into one token,
    alignment either stays correct or raises ValueError, never silently misaligns."""


@pytest.mark.skip(reason="Phase 2: not implemented yet")
def test_alignment_rejects_prompts_with_different_targets() -> None:
    """align_target_tokens raises ValueError for prompts built from different questions."""
