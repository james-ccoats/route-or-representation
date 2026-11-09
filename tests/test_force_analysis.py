"""Tests for rr.force_analysis: gain fractions with bootstrap intervals, and sanity checks."""

from __future__ import annotations

import pytest

from rr.force_analysis import (
    IDENTITY_TOLERANCE,
    GainFraction,
    format_force_summary,
    gain_fractions,
    sanity_reports,
)
from rr.prompts import PromptCondition
from rr.results import ResultRecord

ZERO, CROSS = PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN
OPTIONS = (-0.5, -1.5, -2.5, -3.5)
N = 10


def rec(index, condition, intervention, correct, run="r", option_logprobs=OPTIONS):
    return ResultRecord(
        run=run,
        question_index=index,
        subject="s",
        condition=condition,
        option_logprobs=option_logprobs,
        prediction=0,
        is_correct=correct,
        correct_logprob=option_logprobs[0] if correct else option_logprobs[1],
        num_examples=0 if condition is ZERO else 2,
        example_tokens=0 if condition is ZERO else 10,
        jaccard_per_layer=None,
        intervention=intervention,
    )


def series(condition, intervention, correct_count, run="r", n=N):
    """``n`` records where the first ``correct_count`` questions are answered correctly."""
    return [rec(i, condition, intervention, i < correct_count, run) for i in range(n)]


def decomposition_records(run="r", zero=2, few=8, forced_zero_routing=5, forced_few_routing=6):
    """Z=0.2, F=0.8, few-shot with zero-shot routing=0.5, zero-shot with few-shot routing=0.6."""
    return (
        series(ZERO, "none", zero, run)
        + series(CROSS, "none", few, run)
        + series(CROSS, "force_zero_shot_routing", forced_zero_routing, run)
        + series(ZERO, "force_cross_domain_routing", forced_few_routing, run)
    )


def pick(fractions, kind, metric="accuracy"):
    return next(f for f in fractions if f.kind == kind and f.metric == metric)


# --- gain fractions -------------------------------------------------------------------------


def test_retained_and_recovered_fractions_use_aggregate_scores() -> None:
    fractions = gain_fractions(decomposition_records())

    assert pick(fractions, "retained").estimate == pytest.approx((0.5 - 0.2) / (0.8 - 0.2))
    assert pick(fractions, "recovered").estimate == pytest.approx((0.6 - 0.2) / (0.8 - 0.2))
    assert pick(fractions, "retained").gain == pytest.approx(0.6)


def test_logprob_fractions_are_reported_alongside_accuracy() -> None:
    fractions = gain_fractions(decomposition_records())

    # correct_logprob moves by exactly 1.0 between wrong and right, so it tracks accuracy.
    assert pick(fractions, "retained", "correct_logprob").estimate == pytest.approx(0.5)


def test_fractions_carry_identifying_metadata() -> None:
    fraction = pick(gain_fractions(decomposition_records(run="bio_med_a2b")), "retained")

    assert (fraction.run, fraction.condition, fraction.n) == ("bio_med_a2b", CROSS, N)


def test_bootstrap_interval_brackets_the_estimate() -> None:
    fraction = pick(gain_fractions(decomposition_records()), "retained")

    assert fraction.low <= fraction.estimate <= fraction.high
    assert fraction.low < fraction.high
    assert 0.0 <= fraction.undefined_fraction <= 1.0


def test_deterministic_data_gives_a_degenerate_interval() -> None:
    records = decomposition_records(zero=0, few=10, forced_zero_routing=10, forced_few_routing=0)

    retained = pick(gain_fractions(records), "retained")
    recovered = pick(gain_fractions(records), "recovered")

    assert (retained.estimate, retained.low, retained.high) == (1.0, 1.0, 1.0)
    assert (recovered.estimate, recovered.low, recovered.high) == (0.0, 0.0, 0.0)


def test_fractions_are_reproducible_for_a_seed() -> None:
    assert gain_fractions(decomposition_records(), seed=4) == gain_fractions(
        decomposition_records(), seed=4
    )


def test_no_few_shot_gain_leaves_the_fraction_undefined() -> None:
    records = decomposition_records(zero=5, few=5, forced_zero_routing=5, forced_few_routing=5)

    retained = pick(gain_fractions(records), "retained")

    assert retained.estimate is None
    assert retained.low is None
    assert retained.high is None
    assert retained.gain == 0.0


def test_a_kind_is_skipped_when_its_forced_records_are_absent() -> None:
    records = (
        series(ZERO, "none", 2)
        + series(CROSS, "none", 8)
        + series(CROSS, "force_zero_shot_routing", 5)
    )

    kinds = {f.kind for f in gain_fractions(records)}

    assert kinds == {"retained"}


def test_runs_are_analysed_separately() -> None:
    records = decomposition_records("a", forced_zero_routing=8) + decomposition_records("b")

    fractions = gain_fractions(records)

    retained = {
        f.run: f.estimate for f in fractions if f.kind == "retained" and f.metric == "accuracy"
    }
    assert retained["a"] == pytest.approx(1.0)
    assert retained["b"] == pytest.approx(0.5)


def test_unpaired_questions_are_rejected() -> None:
    records = [
        r
        for r in decomposition_records()
        if not (r.intervention == "force_zero_shot_routing" and r.question_index == 9)
    ]

    with pytest.raises(ValueError, match="paired"):
        gain_fractions(records)


# --- sanity checks ---------------------------------------------------------------------------


def sanity_records(identity_shift=0.0, random_correct=3):
    shifted = tuple(v + identity_shift for v in OPTIONS)
    identity = [
        rec(i, ZERO, "force_zero_shot_routing", i < 4, option_logprobs=shifted) for i in range(N)
    ]
    return (
        series(ZERO, "none", 4)
        + series(CROSS, "none", 8)
        + identity
        + series(CROSS, "force_random_routing", random_correct)
    )


def test_identity_check_passes_when_forcing_own_routing_changes_nothing() -> None:
    (report,) = sanity_reports(sanity_records())

    assert report.identity_max_abs_logprob_diff == 0.0
    assert report.identity_ok is True


def test_identity_check_fails_when_logprobs_move() -> None:
    (report,) = sanity_reports(sanity_records(identity_shift=10 * IDENTITY_TOLERANCE))

    assert report.identity_max_abs_logprob_diff == pytest.approx(10 * IDENTITY_TOLERANCE)
    assert report.identity_ok is False


def test_random_forcing_is_reported_as_hurting_when_accuracy_drops() -> None:
    (report,) = sanity_reports(sanity_records(random_correct=3))

    assert report.random_accuracy_change == {CROSS: pytest.approx(0.3 - 0.8)}
    assert report.random_hurts is True


def test_random_forcing_that_does_not_hurt_is_flagged() -> None:
    (report,) = sanity_reports(sanity_records(random_correct=9))

    assert report.random_hurts is False


def test_missing_controls_are_reported_as_not_run() -> None:
    (report,) = sanity_reports(series(ZERO, "none", 4) + series(CROSS, "none", 8))

    assert report.identity_ok is None
    assert report.random_hurts is None
    assert report.random_accuracy_change == {}


# --- formatting -----------------------------------------------------------------------------------


def test_summary_mentions_fractions_and_sanity_outcomes() -> None:
    text = format_force_summary(decomposition_records() + sanity_records()[2 * N :])

    for expected in ("retained", "recovered", "cross_domain", "accuracy", "identity", "random"):
        assert expected in text


def test_summary_marks_undefined_fractions() -> None:
    text = format_force_summary(
        decomposition_records(zero=5, few=5, forced_zero_routing=5, forced_few_routing=5)
    )

    assert "undefined" in text


def test_gain_fraction_is_a_value_object() -> None:
    fraction = pick(gain_fractions(decomposition_records()), "retained")

    assert isinstance(fraction, GainFraction)
    with pytest.raises(AttributeError):
        fraction.estimate = 0.0  # type: ignore[misc]
