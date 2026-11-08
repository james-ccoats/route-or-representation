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

import torch

from rr.routing._hooks import (
    attach_batch_size_guard,
    normalizes_topk_probs,
    remove_handles,
    resolve_routers,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from rr.routing.record import RoutingRecord


@dataclass(frozen=True)
class ForcingSpec:
    """What to force, where.

    Attributes:
        token_positions: ``[N]`` positions *in the prompt being run* to override.
        topk_indices: ``[num_layers, N, top_k]`` expert ids to force. Dim 0 always
            spans *every* router layer of the model; ``layers`` picks which are applied.
        topk_weights: Optional ``[num_layers, N, top_k]`` mixing weights; ``None``
            keeps the router's own weights for the forced experts (recomputed from
            the router's logits, renormalized if the router normalizes).
        layers: Layers to force; ``None`` means all.
    """

    token_positions: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor | None = None
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        positions = self.token_positions
        for name, tensor in (
            ("token_positions", positions),
            ("topk_indices", self.topk_indices),
        ):
            if not _is_integer(tensor):
                raise ValueError(f"{name} must be an integer tensor, got dtype {tensor.dtype}")
        if positions.dim() != 1:
            raise ValueError(f"token_positions must be 1-D, got {positions.dim()}-D")
        if positions.numel() and positions.min() < 0:
            raise ValueError("token_positions must be non-negative")
        if positions.unique().numel() != positions.numel():
            raise ValueError("token_positions must be unique")
        if self.topk_indices.dim() != 3:
            raise ValueError(f"topk_indices must be 3-D, got {self.topk_indices.dim()}-D")
        num_layers, num_tokens, _ = self.topk_indices.shape
        if num_tokens != positions.numel():
            raise ValueError(
                f"topk_indices shape {tuple(self.topk_indices.shape)} does not match "
                f"{positions.numel()} token positions"
            )
        if self.topk_weights is not None and self.topk_weights.shape != self.topk_indices.shape:
            raise ValueError(
                f"topk_weights shape {tuple(self.topk_weights.shape)} must equal "
                f"topk_indices shape {tuple(self.topk_indices.shape)}"
            )
        if self.layers is not None and any(not 0 <= i < num_layers for i in self.layers):
            raise ValueError(f"layers {self.layers} out of range for {num_layers} layers")


def _is_integer(tensor: torch.Tensor) -> bool:
    return not (tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool)


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
    if len(source_positions) != len(target_positions):
        raise ValueError(
            f"source positions ({len(source_positions)}) and target positions "
            f"({len(target_positions)}) must pair up one to one"
        )
    source = record.select_positions(source_positions)
    return ForcingSpec(
        token_positions=torch.tensor(list(target_positions), dtype=torch.long),
        topk_indices=source.topk_indices,
        topk_weights=source.router_probs if force_weights else None,
        layers=None if layers is None else tuple(layers),
    )


def random_spec(
    num_layers: int,
    num_experts: int,
    top_k: int,
    target_positions: Sequence[int],
    seed: int,
    layers: Sequence[int] | None = None,
) -> ForcingSpec:
    """Build a spec forcing uniformly random distinct experts (sanity control)."""
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(num_layers, len(target_positions), num_experts, generator=generator)
    return ForcingSpec(
        token_positions=torch.tensor(list(target_positions), dtype=torch.long),
        topk_indices=noise.topk(top_k, dim=-1).indices,  # top-k of noise = random distinct experts
        layers=None if layers is None else tuple(layers),
    )


class RoutingForcer:
    """Context manager that overrides expert selection at chosen positions (batch size 1 only).

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
        spec, names = self._spec, self._router_module_names
        if spec.topk_indices.shape[0] != len(names):
            raise ValueError(
                f"spec covers {spec.topk_indices.shape[0]} layers but {len(names)} "
                "router modules were given"
            )
        routers = resolve_routers(self._model, names)
        layers = range(len(names)) if spec.layers is None else spec.layers
        hooks = [(layer, self._make_hook(layer, routers[layer])) for layer in layers]
        try:
            self._handles.append(attach_batch_size_guard(self._model))
            for layer, hook in hooks:
                self._handles.append(routers[layer].register_forward_hook(hook))
        except BaseException:
            self._detach()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all hooks, even if the ``with`` body raised. Do not suppress errors."""
        self._detach()

    def _make_hook(self, layer: int, router: torch.nn.Module):
        spec = self._spec
        renormalize = normalizes_topk_probs(router) if spec.topk_weights is None else False

        def hook(module, args, output):
            logits, scores, indices = output
            positions = spec.token_positions.to(indices.device)
            num_rows = indices.shape[0]
            if positions.numel() and int(positions.max()) >= num_rows:
                raise ValueError(
                    f"forcing position {int(positions.max())} is out of range for "
                    f"{num_rows} tokens in layer {layer}"
                )
            forced = spec.topk_indices[layer].to(device=indices.device, dtype=indices.dtype)
            if forced.shape[-1] != indices.shape[-1]:
                raise ValueError(
                    f"top_k mismatch in layer {layer}: spec forces {forced.shape[-1]} experts "
                    f"but the router selects {indices.shape[-1]}"
                )
            if forced.numel() and (int(forced.min()) < 0 or int(forced.max()) >= logits.shape[-1]):
                raise ValueError(
                    f"forced expert ids must lie in [0, {logits.shape[-1]}) in layer {layer}"
                )
            if spec.topk_weights is not None:
                weights = spec.topk_weights[layer].to(device=scores.device, dtype=scores.dtype)
            else:
                weights = self._router_weights(logits[positions], forced, renormalize, scores.dtype)
            new_indices = indices.clone()
            new_indices[positions] = forced
            new_scores = scores.clone()
            new_scores[positions] = weights
            return logits, new_scores, new_indices

        return hook

    @staticmethod
    def _router_weights(
        logits: torch.Tensor, experts: torch.Tensor, renormalize: bool, dtype: torch.dtype
    ) -> torch.Tensor:
        """The router's own mixing weights for the forced experts, computed as it does."""
        weights = logits.float().softmax(dim=-1).gather(-1, experts)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(dtype)

    def _detach(self) -> None:
        handles, self._handles = self._handles, []
        remove_handles(handles)
