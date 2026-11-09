"""Tests for rr.hidden: per-layer hidden-state shift statistics."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import TINY_NUM_LAYERS, TINY_ROUTER_NAMES, build_tiny_moe
from rr.hidden import HiddenShift, HiddenShiftMeter

E2_H4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]  # router sees only the first two dims


def meter(*weights) -> HiddenShiftMeter:
    return HiddenShiftMeter([torch.tensor(w, dtype=torch.float32) for w in weights])


def states(*tokens) -> torch.Tensor:
    """One layer: ``[1, num_tokens, hidden]``."""
    return torch.tensor([list(tokens)], dtype=torch.float32)


ONES = [1.0, 1.0, 1.0, 1.0]


def plus(vector, delta):
    return [v + d for v, d in zip(vector, delta, strict=True)]


# --- the router-visible share ---------------------------------------------------------------


def test_a_shift_inside_the_router_row_space_is_fully_visible() -> None:
    shift = meter(E2_H4).measure(states(ONES), states(plus(ONES, [3, 4, 0, 0])))

    assert shift.router_visible_share == pytest.approx((1.0,))
    assert shift.router_visible_chance == pytest.approx((0.5,))  # rank 2 of 4 dimensions


def test_a_shift_in_the_null_space_is_invisible_to_the_router() -> None:
    shift = meter(E2_H4).measure(states(ONES), states(plus(ONES, [0, 0, 3, 4])))

    assert shift.router_visible_share == pytest.approx((0.0,))


def test_a_mixed_shift_splits_its_energy_between_the_two_spaces() -> None:
    shift = meter(E2_H4).measure(states(ONES), states(plus(ONES, [3, 0, 4, 0])))

    assert shift.router_visible_share == pytest.approx((9 / 25,))


def test_the_visible_share_pools_energy_over_tokens_instead_of_averaging_shares() -> None:
    before = states(ONES, ONES)
    after = states(plus(ONES, [1, 0, 0, 0]), plus(ONES, [0, 0, math.sqrt(3), 0]))

    shift = meter(E2_H4).measure(before, after)

    # energy 1 is visible and energy 3 is not: 1 / 4, not the per-token mean of 1.0 and 0.0
    assert shift.router_visible_share == pytest.approx((0.25,))


def test_a_rank_deficient_router_has_a_smaller_visible_subspace() -> None:
    duplicated = [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]  # rank 1

    shift = meter(duplicated).measure(states(ONES), states(plus(ONES, [1, 0, 0, 0])))

    assert shift.router_visible_chance == pytest.approx((0.25,))
    assert shift.router_visible_share == pytest.approx((1.0,))


# --- size of the shift ------------------------------------------------------------------------


def test_relative_shift_is_the_mean_of_per_token_ratios() -> None:
    before = states([2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    after = states([4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0])  # ratios 1 and 3

    shift = meter(E2_H4).measure(before, after)

    assert shift.relative_l2 == pytest.approx((2.0,))


def test_relative_shift_is_the_shift_norm_over_the_original_norm() -> None:
    shift = meter(E2_H4).measure(states(ONES), states(plus(ONES, [3, 4, 0, 0])))

    assert shift.relative_l2 == pytest.approx((5 / 2,))  # |delta| = 5, |h| = 2


def test_cosine_distance_runs_from_zero_for_same_direction_to_two_for_opposite() -> None:
    base = [1.0, 0.0, 0.0, 0.0]
    m = meter(E2_H4)

    assert m.measure(states(base), states([5, 0, 0, 0])).cosine_distance == pytest.approx((0.0,))
    assert m.measure(states(base), states([0, 1, 0, 0])).cosine_distance == pytest.approx((1.0,))
    assert m.measure(states(base), states([-1, 0, 0, 0])).cosine_distance == pytest.approx((2.0,))


def test_no_shift_gives_zero_everywhere_and_a_zero_visible_share() -> None:
    shift = meter(E2_H4).measure(states(ONES), states(ONES))

    assert shift.relative_l2 == (0.0,)
    assert shift.cosine_distance == pytest.approx((0.0,), abs=1e-7)
    assert shift.router_visible_share == (0.0,)  # undefined without a shift; defined as 0


def test_each_layer_uses_its_own_router() -> None:
    e1 = [[1.0, 0.0, 0.0, 0.0]]
    e3 = [[0.0, 0.0, 1.0, 0.0]]
    before = torch.tensor([[ONES], [ONES]], dtype=torch.float32)  # two layers, one token
    after = torch.tensor([[plus(ONES, [1, 0, 0, 0])], [plus(ONES, [1, 0, 0, 0])]])

    shift = meter(e1, e3).measure(before, after)

    assert shift.router_visible_share == pytest.approx((1.0, 0.0))
    assert len(shift.relative_l2) == 2


def test_the_inputs_are_not_modified() -> None:
    before, after = states(ONES), states(plus(ONES, [3, 4, 0, 0]))
    before_copy, after_copy = before.clone(), after.clone()

    meter(E2_H4).measure(before, after)

    assert torch.equal(before, before_copy)
    assert torch.equal(after, after_copy)


# --- validation ---------------------------------------------------------------------------------


def test_measure_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        meter(E2_H4).measure(states(ONES), states(ONES, ONES))


def test_measure_rejects_a_hidden_size_that_does_not_match_the_router() -> None:
    with pytest.raises(ValueError, match="hidden"):
        meter(E2_H4).measure(states([1.0, 2.0, 3.0]), states([1.0, 2.0, 3.0]))


def test_measure_rejects_the_wrong_number_of_layers() -> None:
    two_layers = torch.zeros(2, 1, 4)

    with pytest.raises(ValueError, match="layers"):
        meter(E2_H4).measure(two_layers, two_layers)


def test_measure_rejects_input_that_is_not_layers_by_tokens_by_hidden() -> None:
    with pytest.raises(ValueError, match="3-D"):
        meter(E2_H4).measure(torch.zeros(1, 4), torch.zeros(1, 4))


def test_the_meter_needs_two_dimensional_router_weights() -> None:
    with pytest.raises(ValueError, match="2-D"):
        HiddenShiftMeter([torch.zeros(4)])
    with pytest.raises(ValueError, match="at least one"):
        HiddenShiftMeter([])


def test_the_meter_can_be_built_from_a_models_routers() -> None:
    model = build_tiny_moe()

    built = HiddenShiftMeter.from_model(model, TINY_ROUTER_NAMES)

    hidden = model.config.hidden_size
    shift = built.measure(
        torch.ones(TINY_NUM_LAYERS, 3, hidden), torch.ones(TINY_NUM_LAYERS, 3, hidden) * 2
    )
    rank = torch.linalg.matrix_rank(
        model.get_submodule(TINY_ROUTER_NAMES[0]).weight.detach()
    ).item()
    assert len(shift.relative_l2) == TINY_NUM_LAYERS
    assert shift.router_visible_chance[0] == pytest.approx(rank / hidden)


# --- the HiddenShift value ----------------------------------------------------------------------


def make_shift(**overrides) -> HiddenShift:
    values = {
        "relative_l2": (0.5, 0.7),
        "cosine_distance": (0.1, 0.2),
        "router_visible_share": (0.05, 0.1),
        "router_visible_chance": (0.03, 0.03),
    }
    return HiddenShift(**{**values, **overrides})


def test_hidden_shift_round_trips_through_a_dict() -> None:
    shift = make_shift()

    assert HiddenShift.from_dict(shift.to_dict()) == shift


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"relative_l2": (0.5,)}, "length"),
        ({"relative_l2": (-0.1, 0.7)}, "relative_l2"),
        ({"cosine_distance": (0.1, 2.5)}, "cosine_distance"),
        ({"router_visible_share": (0.05, 1.5)}, "router_visible_share"),
        ({"router_visible_chance": (0.03, -0.1)}, "router_visible_chance"),
        ({"relative_l2": (0.5, float("nan"))}, "relative_l2"),
        ({"relative_l2": (), "cosine_distance": (), "router_visible_share": (),
          "router_visible_chance": ()}, "empty"),
    ],
)  # fmt: skip
def test_invalid_hidden_shift_values_are_rejected(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_shift(**overrides)


def test_from_dict_names_missing_fields() -> None:
    data = make_shift().to_dict()
    del data["cosine_distance"]

    with pytest.raises(ValueError, match="cosine_distance"):
        HiddenShift.from_dict(data)
