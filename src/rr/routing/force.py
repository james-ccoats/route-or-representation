"""Force MoE routing on selected token positions.

Role in the project: the intervention half of the routing toolkit. Phase 3
runs a few-shot prompt with zero-shot routing forced on the target-question
tokens (and the reverse); Phase 4 restricts forcing to subsets of layers.

Methodological notes:

* Forcing applies **only at the specified token positions**; every other
  position (e.g. the in-context examples) routes naturally. Default scope is
  the target-question tokens.
* Selection vs. mixing weights is an open decision
  (docs/design_decisions.md). ``ForcingSpec.topk_weights = None`` means
  "force which experts, let the router's own probabilities weight them".
* Implementation choice to resolve in Phase 0/3: modify router logits so the
  desired experts win top-k, or patch the top-k selection inside the MoE block.
  The former is simpler but couples selection and weights.
* Forcing at layer l changes the hidden states fed to layer l+1, whose own
  routing is *also* forced on those positions. Later layers still see
  downstream effects through attention on non-forced positions.
* Sanity checks (tests/test_force_identity.py): forcing a run's own routing
  back onto itself must not change outputs; forcing random experts must hurt.
* Hooks must be removed on exit even on error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

    from rr.routing.record import RoutingRecord


@dataclass(frozen=True)
class ForcingSpec:
    """What to force, where.

    Attributes:
        token_positions: ``[N]`` positions *in the prompt being run* to override.
        topk_indices: ``[num_layers, N, top_k]`` expert ids to force.
        topk_weights: Optional ``[num_layers, N, top_k]`` mixing weights; ``None``
            keeps the router's own weights for the forced experts.
        layers: Layers to force; ``None`` means all.
    """

    token_positions: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor | None = None
    layers: tuple[int, ...] | None = None


def spec_from_record(
    record: RoutingRecord,
    source_positions: Sequence[int],
    target_positions: Sequence[int],
    *,
    force_weights: bool = False,
    layers: Sequence[int] | None = None,
) -> ForcingSpec:
    """Build a spec that transplants routing from one run into another.

    Args:
        record: Routing recorded from the *source* run.
        source_positions: Aligned token positions in the source run.
        target_positions: Corresponding positions in the run being forced.
        force_weights: Also force the recorded router probabilities.
        layers: Restrict forcing to these layers (``None`` = all).

    Returns:
        A :class:`ForcingSpec` for :class:`RoutingForcer`.
    """
    # TODO(Phase 3): select source positions from `record`, pair with target positions.
    raise NotImplementedError("TODO(Phase 3): build spec from recorded routing")


def random_spec(
    num_layers: int,
    num_experts: int,
    top_k: int,
    target_positions: Sequence[int],
    seed: int,
    layers: Sequence[int] | None = None,
) -> ForcingSpec:
    """Build a spec forcing uniformly random distinct experts (sanity control)."""
    # TODO(Phase 3): seeded sampling of `top_k` distinct experts per (layer, token).
    raise NotImplementedError("TODO(Phase 3): random routing spec")


class RoutingForcer:
    """Context manager that overrides expert selection at chosen positions.

    Example:
        >>> with RoutingForcer(model, router_names, spec):
        ...     score = score_mc(model, tokenizer, prompt, correct_index)
    """

    def __init__(
        self, model: PreTrainedModel, router_module_names: Sequence[str], spec: ForcingSpec
    ) -> None:
        """
        Args:
            model: The loaded MoE model.
            router_module_names: One router module name per layer, in layer order.
            spec: What to force.
        """
        self._model = model
        self._router_module_names = tuple(router_module_names)
        self._spec = spec
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> RoutingForcer:
        """Attach hooks that override routing at ``spec.token_positions``."""
        # TODO(Phase 3): register hooks only for spec.layers; hooks must leave
        # all other token positions untouched and handle batch size 1.
        raise NotImplementedError("TODO(Phase 3): attach forcing hooks")

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all hooks, even if the ``with`` body raised. Do not suppress errors."""
        # TODO(Phase 3): remove handles in try/finally.
        raise NotImplementedError("TODO(Phase 3): remove forcing hooks")
