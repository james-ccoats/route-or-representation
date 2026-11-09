"""Tests for rr.shot_curves: gain against k, with spread across example-set replicates."""

from __future__ import annotations

import pytest

from rr.force_analysis import GainFraction
from rr.prompts import PromptCondition
from rr.results import ResultRecord
from rr.shot_curves import (
    format_shot_force_summary,
    format_shot_gain_table,
    fractions_by_shots,
    gain_by_shots,
    mean_by_shots,
)

ZERO, CROSS, IRR = (
    PromptCondition.ZERO_SHOT,
    PromptCondition.CROSS_DOMAIN,
    PromptCondition.IRRELEVANT,
)
OPTIONS = (-0.5, -1.5, -2.5, -3.5)
N = 10


def rec(run, index, condition, correct, intervention="none", jaccard=None, logprob=None):
    options = OPTIONS if logprob is None else (logprob, -9.0, -9.5, -10.0)
    return ResultRecord(
        run=run,
        question_index=index,
        subject="s",
        condition=condition,
        option_logprobs=options,
        prediction=0,
        is_correct=correct,
        correct_logprob=options[0] if correct else options[1],
        num_examples=0 if condition is ZERO else 2,
        example_tokens=0 if condition is ZERO else 10,
        jaccard_per_layer=jaccard,
        intervention=intervention,
    )


def cell(run, zero_correct, cond_correct, condition=CROSS, n=N):
    """A run with zero-shot and one few-shot condition; the first q questions are correct."""
    return [rec(run, i, ZERO, i < zero_correct) for i in range(n)] + [
        rec(run, i, condition, i < cond_correct, jaccard=(0.5, 0.7)) for i in range(n)
    ]


def test_gain_by_shots_reports_each_k_with_the_mean_over_replicates() -> None:
    records = (
        cell("d_a2b_r0_k1", 2, 4)
        + cell("d_a2b_r1_k1", 2, 6)
        + cell("d_a2b_r0_k2", 2, 7)
        + cell("d_a2b_r1_k2", 2, 9)
    )

    gains = gain_by_shots(records, "accuracy", n_boot=200)

    by_k = {g.shots: g for g in gains}
    assert sorted(by_k) == [1, 2]
    assert by_k[1].mean_gain == pytest.approx(0.3)  # replicates +0.2 and +0.4
    assert by_k[2].mean_gain == pytest.approx(0.6)  # replicates +0.5 and +0.7
    assert by_k[1].replicate_gains == pytest.approx((0.2, 0.4))
    assert by_k[1].n_replicates == 2 and by_k[1].n_questions == N


def test_gain_by_shots_interval_contains_the_estimate() -> None:
    records = cell("d_a2b_r0_k1", 2, 6) + cell("d_a2b_r1_k1", 3, 7)

    (gain,) = gain_by_shots(records, "accuracy", n_boot=500)

    assert gain.low <= gain.mean_gain <= gain.high


def test_gain_by_shots_separates_directions_and_conditions() -> None:
    records = (
        cell("x_a2b_r0_k1", 2, 6)
        + cell("y_a2b_r0_k1", 2, 3)
        + cell("x_a2b_r0_k1", 2, 5, condition=IRR)[N:]
    )

    gains = gain_by_shots(records, "accuracy", n_boot=100)

    keys = {(g.direction, g.condition) for g in gains}
    assert keys == {("x_a2b", CROSS), ("y_a2b", CROSS), ("x_a2b", IRR)}


def test_gain_by_shots_uses_the_requested_metric() -> None:
    records = [rec("d_a2b_r0_k1", i, ZERO, True, logprob=-3.0) for i in range(N)] + [
        rec("d_a2b_r0_k1", i, CROSS, True, logprob=-1.0) for i in range(N)
    ]

    (gain,) = gain_by_shots(records, "correct_logprob", n_boot=100)

    assert gain.mean_gain == pytest.approx(2.0)


def test_gain_by_shots_skips_runs_that_are_not_part_of_the_shot_grid() -> None:
    assert gain_by_shots(cell("plain_run", 2, 6), "accuracy", n_boot=100) == []


def test_gain_by_shots_ignores_forced_records() -> None:
    forced = [
        rec("d_a2b_r0_k1", i, CROSS, True, intervention="force_zero_shot_routing") for i in range(N)
    ]

    (gain,) = gain_by_shots(cell("d_a2b_r0_k1", 2, 6) + forced, "accuracy", n_boot=100)

    assert gain.mean_gain == pytest.approx(0.4)


def test_gain_by_shots_rejects_replicates_over_different_questions() -> None:
    records = cell("d_a2b_r0_k1", 2, 6) + cell("d_a2b_r1_k1", 2, 6, n=N - 1)

    with pytest.raises(ValueError, match="questions"):
        gain_by_shots(records, "accuracy", n_boot=100)


def test_gain_by_shots_rejects_a_few_shot_run_missing_its_zero_shot() -> None:
    records = [r for r in cell("d_a2b_r0_k1", 2, 6) if r.condition is not ZERO]

    with pytest.raises(ValueError, match="zero"):
        gain_by_shots(records, "accuracy", n_boot=100)


def test_mean_by_shots_averages_a_per_record_metric_over_questions_then_replicates() -> None:
    records = cell("d_a2b_r0_k1", 2, 4) + cell("d_a2b_r1_k1", 2, 4)

    (summary,) = mean_by_shots(records, "jaccard")

    assert summary.condition is CROSS
    assert summary.mean == pytest.approx(0.6)
    assert summary.replicate_means == pytest.approx((0.6, 0.6))


def test_mean_by_shots_skips_zero_shot_and_runs_where_the_metric_is_absent() -> None:
    records = cell("d_a2b_r0_k1", 2, 4)

    assert mean_by_shots(records, "hidden_shift") == []


def _fraction(run, kind, estimate, condition=CROSS, metric="accuracy"):
    return GainFraction(
        run=run,
        condition=condition,
        kind=kind,
        metric=metric,
        estimate=estimate,
        low=None if estimate is None else estimate - 0.1,
        high=None if estimate is None else estimate + 0.1,
        gain=0.3,
        n=N,
        undefined_fraction=0.0 if estimate is not None else 1.0,
    )


def test_fractions_by_shots_averages_defined_replicates_and_reports_the_range() -> None:
    fractions = [
        _fraction("d_a2b_r0_k3", "retained", 0.4),
        _fraction("d_a2b_r1_k3", "retained", 0.8),
        _fraction("d_a2b_r2_k3", "retained", None),
        _fraction("d_a2b_r0_k3", "recovered", 0.2),
    ]

    result = {(f.kind, f.shots): f for f in fractions_by_shots(fractions)}

    retained = result[("retained", 3)]
    assert retained.mean == pytest.approx(0.6)
    assert (retained.minimum, retained.maximum) == (0.4, 0.8)
    assert (retained.n_defined, retained.n_replicates) == (2, 3)
    assert result[("recovered", 3)].mean == pytest.approx(0.2)


def test_fractions_by_shots_marks_a_group_with_no_defined_estimate() -> None:
    fractions = [_fraction("d_a2b_r0_k3", "retained", None)]

    (result,) = fractions_by_shots(fractions)

    assert result.mean is None and result.n_defined == 0


def test_gain_table_lists_each_direction_condition_and_k() -> None:
    records = cell("d_a2b_r0_k1", 2, 4) + cell("d_a2b_r0_k2", 2, 6)

    table = format_shot_gain_table(records, n_boot=100)

    assert "d_a2b" in table and "cross_domain" in table
    assert "k=1" in table and "k=2" in table


def test_force_summary_aggregates_across_replicates_and_reports_the_controls() -> None:
    def run_records(run, forced):
        records = cell(run, 2, 8)
        records += [rec(run, i, CROSS, i < forced, "force_zero_shot_routing") for i in range(N)]
        records += [rec(run, i, ZERO, i < 2, "force_zero_shot_routing") for i in range(N)]
        records += [rec(run, i, CROSS, False, "force_random_routing") for i in range(N)]
        return records

    summary = format_shot_force_summary(
        run_records("d_a2b_r0_k2", 5) + run_records("d_a2b_r1_k2", 5), n_boot=100
    )

    assert "k=2" in summary and "retained" in summary
    assert "identity" in summary and "random" in summary


def test_summaries_use_the_shot_tables_only_for_shot_grid_runs() -> None:
    from rr.shot_curves import summarize_force_results, summarize_pilot_results

    grid = cell("d_a2b_r0_k1", 2, 6)
    plain = cell("plain_run", 2, 6)

    assert "k=1" in summarize_pilot_results(grid, n_boot=100)
    assert "k=1" not in summarize_pilot_results(plain, n_boot=100)
    assert "Sanity controls over all runs" in summarize_force_results(grid, n_boot=100)
    assert "Sanity controls:" in summarize_force_results(plain, n_boot=100)
