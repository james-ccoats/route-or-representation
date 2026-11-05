"""Sanity checks for rr.routing.force (Phase 3). Skipped until implemented."""

import pytest


@pytest.mark.skip(reason="Phase 3: not implemented yet")
def test_forcing_own_routing_leaves_logits_unchanged() -> None:
    """Record routing on the zero-shot prompt, then force that same routing back
    onto the same prompt. Final-position logits match the unforced run within
    numerical tolerance."""


@pytest.mark.skip(reason="Phase 3: not implemented yet")
def test_forcing_random_experts_changes_outputs() -> None:
    """Forcing random experts on the target tokens changes the logits, and over
    a small question set lowers accuracy relative to unforced routing."""


@pytest.mark.skip(reason="Phase 3: not implemented yet")
def test_forcing_touches_only_specified_positions() -> None:
    """Routing recorded during a forced run equals the forced routing at
    spec.token_positions and equals the unforced routing elsewhere."""


@pytest.mark.skip(reason="Phase 0: not implemented yet")
def test_recorder_and_forcer_remove_hooks_on_error() -> None:
    """After a with-block that raises, no forward hooks remain on the router modules."""
