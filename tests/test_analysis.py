"""Tests for the gain-fraction arithmetic in rr.analysis."""

import pytest

from rr.analysis import fraction_of_gain_recovered, fraction_of_gain_retained


def test_retained_is_one_when_forcing_changes_nothing() -> None:
    assert fraction_of_gain_retained(forced=0.7, zero_shot=0.5, few_shot=0.7) == pytest.approx(1.0)


def test_retained_is_zero_when_forcing_erases_gain() -> None:
    assert fraction_of_gain_retained(forced=0.5, zero_shot=0.5, few_shot=0.7) == pytest.approx(0.0)


def test_retained_partial_gain() -> None:
    assert fraction_of_gain_retained(forced=0.6, zero_shot=0.5, few_shot=0.7) == pytest.approx(0.5)


def test_retained_can_be_negative_when_forcing_hurts_below_zero_shot() -> None:
    assert fraction_of_gain_retained(forced=0.4, zero_shot=0.5, few_shot=0.7) < 0


def test_recovered_matches_formula() -> None:
    assert fraction_of_gain_recovered(forced=0.55, zero_shot=0.5, few_shot=0.7) == pytest.approx(
        0.25
    )


def test_negative_gain_is_handled_consistently() -> None:
    assert fraction_of_gain_retained(forced=0.55, zero_shot=0.6, few_shot=0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("fn", [fraction_of_gain_retained, fraction_of_gain_recovered])
def test_zero_gain_raises_value_error(fn) -> None:
    with pytest.raises(ValueError, match="undefined"):
        fn(forced=0.5, zero_shot=0.5, few_shot=0.5)


# --- pilot summaries ----------------------------------------------------------------------

import numpy as np  # noqa: E402

from rr.analysis import (  # noqa: E402
    BootstrapResult,
    accuracy_by_condition,
    format_pilot_summary,
    load_results,
    mean_jaccard_by_condition,
    paired_bootstrap_difference,
    pilot_contrasts,
)
from rr.prompts import PromptCondition  # noqa: E402
from rr.results import ResultRecord, append_records  # noqa: E402

ZERO, CROSS, SAME, IRRELEVANT = (
    PromptCondition.ZERO_SHOT,
    PromptCondition.CROSS_DOMAIN,
    PromptCondition.SAME_DOMAIN,
    PromptCondition.IRRELEVANT,
)
CORRECT_LOGPROB, WRONG_LOGPROB = -0.5, -1.5


def rec(run: str, index: int, condition: PromptCondition, correct: bool, jaccard=None):
    if condition is ZERO:
        jaccard = None
    elif jaccard is None:
        jaccard = (0.5, 0.5)
    return ResultRecord(
        run=run,
        question_index=index,
        subject="s",
        condition=condition,
        option_logprobs=(CORRECT_LOGPROB, WRONG_LOGPROB, -2.5, -3.5),
        prediction=0,
        is_correct=correct,
        correct_logprob=CORRECT_LOGPROB if correct else WRONG_LOGPROB,
        num_examples=0 if condition is ZERO else 2,
        example_tokens=0 if condition is ZERO else 10,
        jaccard_per_layer=jaccard,
    )


def test_accuracy_by_condition_averages_per_run_and_condition() -> None:
    records = [
        rec("r1", 0, ZERO, False),
        rec("r1", 1, ZERO, True),
        rec("r1", 0, CROSS, True),
        rec("r1", 1, CROSS, True),
        rec("r2", 0, ZERO, False),
    ]

    result = accuracy_by_condition(records)

    assert result == {("r1", ZERO): 0.5, ("r1", CROSS): 1.0, ("r2", ZERO): 0.0}


def test_mean_jaccard_by_condition_averages_layers_and_questions_and_skips_zero_shot() -> None:
    records = [
        rec("r1", 0, ZERO, True),
        rec("r1", 0, CROSS, True, jaccard=(0.2, 0.4)),
        rec("r1", 1, CROSS, True, jaccard=(0.6, 0.8)),
    ]

    result = mean_jaccard_by_condition(records)

    assert result == {("r1", CROSS): pytest.approx(0.5)}


def test_bootstrap_of_identical_samples_is_zero() -> None:
    result = paired_bootstrap_difference([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])

    assert result == BootstrapResult(mean_diff=0.0, low=0.0, high=0.0, n=3)


def test_bootstrap_of_a_constant_difference_has_a_degenerate_interval() -> None:
    result = paired_bootstrap_difference([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])

    assert (result.mean_diff, result.low, result.high) == (1.0, 1.0, 1.0)


def test_bootstrap_interval_contains_the_mean_and_is_seed_reproducible() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 50)

    first = paired_bootstrap_difference(a, b, seed=3)
    second = paired_bootstrap_difference(a, b, seed=3)

    assert first == second
    assert first.low < first.mean_diff < first.high
    assert first.mean_diff == pytest.approx(float(np.mean(a - b)))


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "match"),
    [
        ([1.0], [1.0, 2.0], {}, "same length"),
        ([], [], {}, "empty"),
        ([1.0], [1.0], {"confidence": 1.0}, "confidence"),
        ([1.0], [1.0], {"n_boot": 0}, "n_boot"),
    ],
)
def test_bootstrap_rejects_invalid_input(a, b, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        paired_bootstrap_difference(a, b, **kwargs)


def _pilot_records():
    records = []
    for index in range(4):
        records += [
            rec("r1", index, ZERO, False),
            rec("r1", index, CROSS, True, jaccard=(0.4, 0.6)),
            rec("r1", index, SAME, True, jaccard=(0.3, 0.3)),
            rec("r1", index, IRRELEVANT, False, jaccard=(0.9, 0.9)),
        ]
    return records


def _contrast(contrasts, name):
    return next(c for c in contrasts if c.name == name)


def test_pilot_contrasts_report_paired_differences() -> None:
    contrasts = pilot_contrasts(_pilot_records())

    cross_vs_zero = _contrast(contrasts, "accuracy: cross_domain - zero_shot")
    shift = _contrast(contrasts, "jaccard: irrelevant - cross_domain")
    assert (cross_vs_zero.mean_diff, cross_vs_zero.low, cross_vs_zero.high) == (1.0, 1.0, 1.0)
    assert cross_vs_zero.n == 4
    assert shift.mean_diff == pytest.approx(0.4)  # irrelevant examples move routing less
    assert {c.run for c in contrasts} == {"r1"}


def test_pilot_contrasts_skip_conditions_a_run_does_not_have() -> None:
    records = [r for r in _pilot_records() if r.condition is not SAME]

    names = {c.name for c in pilot_contrasts(records)}

    assert "accuracy: cross_domain - zero_shot" in names
    assert not any("same_domain" in name for name in names)


def test_pilot_contrasts_reject_unpaired_questions() -> None:
    records = [r for r in _pilot_records() if not (r.condition is CROSS and r.question_index == 3)]

    with pytest.raises(ValueError, match="paired"):
        pilot_contrasts(records)


def test_pilot_contrasts_are_reproducible_for_a_seed() -> None:
    assert pilot_contrasts(_pilot_records(), seed=5) == pilot_contrasts(_pilot_records(), seed=5)


def test_load_results_reads_a_results_directory(tmp_path) -> None:
    records = [rec("r1", 0, ZERO, True), rec("r1", 0, CROSS, False)]
    append_records(tmp_path / "r1.jsonl", records)

    assert load_results(tmp_path) == records


def test_format_pilot_summary_mentions_runs_conditions_and_contrasts() -> None:
    text = format_pilot_summary(_pilot_records())

    for expected in ("r1", "zero_shot", "cross_domain", "accuracy: cross_domain - zero_shot"):
        assert expected in text
