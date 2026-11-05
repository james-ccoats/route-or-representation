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
