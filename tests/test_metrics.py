"""Tests for rr.routing.metrics: top-k expert-set overlap."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rr.routing.metrics import per_layer_jaccard, token_jaccard


def _topk(*rows: list[int]) -> torch.Tensor:
    """Build a [1, num_tokens, top_k] tensor from one list of expert ids per token."""
    return torch.tensor([list(rows)], dtype=torch.long)


def test_identical_routing_has_jaccard_one() -> None:
    topk = _topk([1, 2, 3, 4], [5, 6, 7, 8])

    assert torch.equal(token_jaccard(topk, topk), torch.ones(1, 2))


def test_disjoint_routing_has_jaccard_zero() -> None:
    result = token_jaccard(_topk([1, 2, 3, 4]), _topk([5, 6, 7, 8]))

    assert torch.equal(result, torch.zeros(1, 1))


def test_partial_overlap_is_intersection_over_union() -> None:
    # {1,2,3,4} vs {3,4,5,6}: |inter| = 2, |union| = 6.
    result = token_jaccard(_topk([1, 2, 3, 4]), _topk([3, 4, 5, 6]))

    assert result.item() == pytest.approx(2 / 6)


def test_order_within_top_k_is_ignored() -> None:
    result = token_jaccard(_topk([1, 2, 3, 4]), _topk([4, 3, 2, 1]))

    assert result.item() == pytest.approx(1.0)


def test_output_shape_is_layers_by_tokens() -> None:
    topk = torch.randint(0, 64, (3, 5, 8))

    assert token_jaccard(topk, topk).shape == (3, 5)


def test_values_stay_in_unit_interval_for_random_routing() -> None:
    generator = torch.Generator().manual_seed(0)
    a = torch.stack([torch.randperm(16, generator=generator)[:4] for _ in range(60)]).view(3, 20, 4)
    b = torch.stack([torch.randperm(16, generator=generator)[:4] for _ in range(60)]).view(3, 20, 4)

    result = token_jaccard(a, b)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_token_jaccard_does_not_modify_inputs() -> None:
    a, b = _topk([1, 2, 3, 4]), _topk([3, 4, 5, 6])
    a_before, b_before = a.clone(), b.clone()

    token_jaccard(a, b)

    assert torch.equal(a, a_before)
    assert torch.equal(b, b_before)


def test_mismatched_shapes_raise_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        token_jaccard(
            torch.zeros(2, 3, 4, dtype=torch.long), torch.zeros(2, 3, 5, dtype=torch.long)
        )


def test_non_three_dimensional_input_raises_value_error() -> None:
    flat = torch.zeros(3, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="3-D"):
        token_jaccard(flat, flat)


def test_per_layer_jaccard_averages_over_tokens() -> None:
    a = torch.tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    b = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [1, 2]]])

    result = per_layer_jaccard(a, b)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result == pytest.approx([0.5, 1.0])
