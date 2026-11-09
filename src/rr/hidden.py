"""Per-layer summary of how far the hidden state the router sees has shifted.

Role in the project: the forcing experiment holds *routing* fixed and asks how much of
the few-shot gain survives. That infers a hidden-state ("representation") effect as the
residual; it never measures one. This module measures it directly, as a supporting
observation: for the aligned target-question tokens, how different is the state each
router receives in a few-shot prompt from the one it receives in the zero-shot prompt.

What is compared: the router's input at each MoE layer, which in OLMoE is
``post_attention_layernorm(residual stream after attention)``. It is the RMS-normalised
residual stream at that layer, i.e. exactly the vector the router multiplies by its
weight matrix ``W`` (``logits = W h``). Magnitudes are therefore those of the normalised
state, not of the raw residual.

Per layer, with ``h0`` the zero-shot state, ``h1`` the few-shot state and
``delta = h1 - h0`` over the aligned target tokens:

* ``relative_l2``: mean over tokens of ``|delta| / |h0|``.
* ``cosine_distance``: mean over tokens of ``1 - cos(h0, h1)``.
* ``router_visible_share``: the fraction of the shift's squared norm that lies in the
  row space of ``W`` (the directions that can change the routing logits), pooled over
  tokens as ``sum |P delta|^2 / sum |delta|^2``. The rest lies in the null space of
  ``W``: it cannot change which experts are chosen, only what they compute.
* ``router_visible_chance``: ``rank(W) / hidden_size``, the share a shift in a random
  direction would have in the row space. The visible share is only informative
  relative to this: ``W`` has at most one row per expert (64) against thousands of
  hidden dimensions, so a small visible share is the default, not a finding.

These are descriptive statistics. A large shift does not show that it caused the gain,
and any extra context moves the state somewhat; compare against the irrelevant-example
control. Only summaries are kept, never the raw vectors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional

from rr.routing._hooks import resolve_routers

if TYPE_CHECKING:
    from transformers import PreTrainedModel

NORM_FLOOR = 1e-12  # keeps ratios finite for an all-zero state
COSINE_DISTANCE_MAX = 2.0
FIELDS = ("relative_l2", "cosine_distance", "router_visible_share", "router_visible_chance")


@dataclass(frozen=True)
class HiddenShift:
    """Per-layer shift of the router's input between two prompts (see module docstring).

    Each attribute is a tuple with one value per MoE layer, in layer order.
    """

    relative_l2: tuple[float, ...]
    cosine_distance: tuple[float, ...]
    router_visible_share: tuple[float, ...]
    router_visible_chance: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {len(getattr(self, name)) for name in FIELDS}
        if len(lengths) != 1:
            raise ValueError(f"the per-layer tuples must have the same length, got {lengths}")
        if lengths == {0}:
            raise ValueError("a hidden shift needs at least one layer; got an empty one")
        bounds = {
            "relative_l2": (0.0, math.inf),
            "cosine_distance": (0.0, COSINE_DISTANCE_MAX),
            "router_visible_share": (0.0, 1.0),
            "router_visible_chance": (0.0, 1.0),
        }
        for name, (low, high) in bounds.items():
            values = getattr(self, name)
            if not all(_is_number(v) and low <= v <= high for v in values):
                raise ValueError(f"{name}: values must be finite numbers in [{low}, {high}]")

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-serializable form."""
        return {name: list(getattr(self, name)) for name in FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HiddenShift:
        """Parse and validate (external data is never trusted).

        Raises:
            ValueError: Naming the offending field.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"hidden_shift: must be an object, got {type(data).__name__}")
        values: dict[str, tuple[float, ...]] = {}
        for name in FIELDS:
            if name not in data:
                raise ValueError(f"hidden_shift: missing field {name!r}")
            raw = data[name]
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"hidden_shift: {name} must be a list")
            values[name] = tuple(raw)
        return cls(**values)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class HiddenShiftMeter:
    """Computes :class:`HiddenShift` from router inputs, using each layer's router weights.

    The row-space basis of each router matrix is computed once at construction.
    """

    def __init__(self, router_weights: Sequence[torch.Tensor]) -> None:
        """
        Args:
            router_weights: One ``[num_experts, hidden_size]`` matrix per layer, in layer order.

        Raises:
            ValueError: If there are none, a matrix is not 2-D or all zero, or the
                hidden sizes disagree.
        """
        if len(router_weights) == 0:
            raise ValueError("need at least one router weight matrix")
        bases: list[torch.Tensor] = []
        for weight in router_weights:
            if weight.dim() != 2:
                raise ValueError(
                    f"router weights must be 2-D [experts, hidden], got {weight.dim()}-D"
                )
            bases.append(_row_space_basis(weight))
        hidden_sizes = {basis.shape[1] for basis in bases}
        if len(hidden_sizes) != 1:
            raise ValueError(f"routers disagree on the hidden size: {sorted(hidden_sizes)}")
        self._bases = tuple(bases)
        self._hidden_size = hidden_sizes.pop()
        self._chance = tuple(basis.shape[0] / self._hidden_size for basis in bases)

    @classmethod
    def from_model(
        cls, model: PreTrainedModel, router_module_names: Sequence[str]
    ) -> HiddenShiftMeter:
        """Build from the router modules of a loaded model (weights read once, on the CPU)."""
        routers = resolve_routers(model, router_module_names)
        return cls([router.weight.detach().float().cpu() for router in routers])

    def measure(self, before: torch.Tensor, after: torch.Tensor) -> HiddenShift:
        """Shift of the router input from ``before`` to ``after`` on aligned tokens.

        Args:
            before: ``[num_layers, num_tokens, hidden_size]`` router inputs, e.g. zero-shot.
            after: Same shape, token-aligned with ``before``, e.g. few-shot.

        Raises:
            ValueError: On shapes that are not 3-D, differ, or do not fit the routers.
        """
        if before.dim() != 3:
            raise ValueError(f"expected 3-D [layers, tokens, hidden] states, got {before.dim()}-D")
        if before.shape != after.shape:
            raise ValueError(f"shape mismatch: {tuple(before.shape)} vs {tuple(after.shape)}")
        if before.shape[0] != len(self._bases):
            raise ValueError(
                f"got states for {before.shape[0]} layers but {len(self._bases)} routers"
            )
        if before.shape[2] != self._hidden_size:
            raise ValueError(
                f"hidden size {before.shape[2]} does not match the routers' {self._hidden_size}"
            )
        h0, h1 = before.detach().float(), after.detach().float()
        delta = h1 - h0
        relative = (delta.norm(dim=-1) / h0.norm(dim=-1).clamp_min(NORM_FLOOR)).mean(dim=1)
        cosine = functional.cosine_similarity(h0, h1, dim=-1, eps=NORM_FLOOR)
        cosine_distance = (1.0 - cosine).clamp(0.0, COSINE_DISTANCE_MAX).mean(dim=1)
        total = (delta**2).sum(dim=(1, 2))
        visible = torch.stack(
            [((delta[i] @ basis.T) ** 2).sum() for i, basis in enumerate(self._bases)]
        )
        share = torch.where(
            total > 0, visible / total.clamp_min(NORM_FLOOR), torch.zeros_like(total)
        )
        return HiddenShift(
            relative_l2=tuple(relative.tolist()),
            cosine_distance=tuple(cosine_distance.tolist()),
            router_visible_share=tuple(share.clamp(0.0, 1.0).tolist()),
            router_visible_chance=self._chance,
        )


def _row_space_basis(weight: torch.Tensor) -> torch.Tensor:
    """Orthonormal rows spanning the row space of ``weight``, as ``[rank, hidden]``."""
    matrix = weight.detach().float().cpu()
    _, singular_values, right_vectors = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = singular_values.max() * max(matrix.shape) * torch.finfo(torch.float32).eps
    rank = int((singular_values > tolerance).sum())
    if rank == 0:
        raise ValueError("a router weight matrix is all zero")
    return right_vectors[:rank]
