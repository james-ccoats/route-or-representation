"""Record per-token top-k expert routing with forward hooks.

Role in the project: the measurement half of the routing toolkit. Phase 0
validates that recording works; Phase 2 uses recorded routing to compute how
much examples shift routing; Phase 3 uses recorded routing as the source of
the routing that is forced into another run.

Shape convention (batch size 1 throughout; the batch dimension is dropped):
``[num_layers, num_tokens, top_k]``.

Methodological notes:

* Hook the router modules (named in the model config, verified in Phase 0).
  The router output is logits over experts; top-k indices and probabilities
  must be derived exactly as the model does (softmax-then-topk vs
  topk-then-softmax, optional renormalization). Verify against the installed
  ``transformers`` source and test that recorded top-k matches the experts the
  model actually used.
* Hooks must be removed on exit even if the body raises (try/finally).
* Recording must not change model outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


@dataclass(frozen=True)
class RoutingRecord:
    """Routing decisions for one forward pass.

    Attributes:
        topk_indices: ``[num_layers, num_tokens, top_k]`` long tensor of the
            selected expert ids.
        router_probs: ``[num_layers, num_tokens, top_k]`` float tensor of the
            router probability (mixing weight) of each selected expert, in the
            same order as ``topk_indices``.
    """

    topk_indices: torch.Tensor
    router_probs: torch.Tensor

    def select_positions(self, positions: Sequence[int]) -> RoutingRecord:
        """Return a new record restricted to the given token positions."""
        # TODO(Phase 0): index dim 1 with `positions` and return a new RoutingRecord.
        raise NotImplementedError("TODO(Phase 0): slice record by token positions")


class RoutingRecorder:
    """Context manager that records routing during forward passes.

    Example:
        >>> with RoutingRecorder(model, router_names, top_k=8) as recorder:
        ...     model(**inputs)
        >>> record = recorder.get_record()
    """

    def __init__(
        self, model: PreTrainedModel, router_module_names: Sequence[str], top_k: int
    ) -> None:
        """
        Args:
            model: The loaded MoE model.
            router_module_names: One router module name per layer, in layer order.
            top_k: Number of experts selected per token.
        """
        self._model = model
        self._router_module_names = tuple(router_module_names)
        self._top_k = top_k
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> RoutingRecorder:
        """Attach a forward hook to every router module."""
        # TODO(Phase 0): register hooks via module.register_forward_hook and
        # store the handles in self._handles; buffers are per-layer lists.
        raise NotImplementedError("TODO(Phase 0): attach hooks")

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all hooks, even if the ``with`` body raised. Do not suppress errors."""
        # TODO(Phase 0): for handle in self._handles: handle.remove() inside a
        # try/finally so partial failures still clear the list; return None.
        raise NotImplementedError("TODO(Phase 0): remove hooks")

    def get_record(self) -> RoutingRecord:
        """Assemble buffered routing into a :class:`RoutingRecord`.

        Raises:
            RuntimeError: If nothing was recorded (no forward pass ran).
        """
        # TODO(Phase 0): stack per-layer buffers into [L, T, K] tensors.
        raise NotImplementedError("TODO(Phase 0): build RoutingRecord")
