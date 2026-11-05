"""Routing-comparison metrics.

Role in the project: quantifies how much a prompt changes which experts the
router picks. Phase 2 uses it to test whether relevant examples shift routing
on target-question tokens more than irrelevant ones; Phase 4 uses it for
layer-by-layer analysis.

Inputs are routing tensors already restricted to aligned target-question
positions (see :mod:`rr.routing.align` and ``RoutingRecord.select_positions``),
shape ``[num_layers, num_tokens, top_k]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


def token_jaccard(topk_a: torch.Tensor, topk_b: torch.Tensor) -> torch.Tensor:
    """Jaccard overlap of top-k expert *sets* for every (layer, token).

    Args:
        topk_a: ``[num_layers, num_tokens, top_k]`` expert ids for run A.
        topk_b: Same shape for run B, token-aligned with A.

    Returns:
        ``[num_layers, num_tokens]`` float tensor in ``[0, 1]``;
        ``|A ∩ B| / |A ∪ B|``. Order within the top-k is ignored.

    Raises:
        ValueError: If shapes differ.
    """
    # TODO(Phase 2): vectorized set intersection (e.g. one-hot over experts, or
    # pairwise equality counts), union = 2k - intersection.
    raise NotImplementedError("TODO(Phase 2): per-token Jaccard")


def per_layer_jaccard(topk_a: torch.Tensor, topk_b: torch.Tensor) -> np.ndarray:
    """Mean Jaccard over tokens, per layer.

    Returns:
        ``[num_layers]`` array. Lower values mean more routing change.
    """
    # TODO(Phase 2): token_jaccard(...).mean(dim=1) as numpy.
    raise NotImplementedError("TODO(Phase 2): per-layer mean Jaccard")
