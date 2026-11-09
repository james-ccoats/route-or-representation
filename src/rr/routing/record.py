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

import torch

from rr.routing._hooks import attach_batch_size_guard, remove_handles, resolve_routers

if TYPE_CHECKING:
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
        router_inputs: ``[num_layers, num_tokens, hidden_size]`` float32 CPU tensor of the
            hidden state each router received (the vector it multiplies by its weights),
            or ``None`` if the recorder was not asked to keep it.
    """

    topk_indices: torch.Tensor
    router_probs: torch.Tensor
    router_inputs: torch.Tensor | None = None

    def select_positions(self, positions: Sequence[int]) -> RoutingRecord:
        """Return a new record restricted to the given token positions."""
        if len(positions) == 0:
            raise ValueError("cannot select an empty set of positions")
        num_tokens = self.topk_indices.shape[1]
        if any(not 0 <= p < num_tokens for p in positions):
            raise IndexError(f"positions {list(positions)} out of range for {num_tokens} tokens")
        index = torch.tensor(list(positions), dtype=torch.long)
        return RoutingRecord(
            topk_indices=self.topk_indices.index_select(1, index),
            router_probs=self.router_probs.index_select(1, index),
            router_inputs=(
                None if self.router_inputs is None else self.router_inputs.index_select(1, index)
            ),
        )


class RoutingRecorder:
    """Context manager that records routing during forward passes (batch size 1 only).

    Example:
        >>> with RoutingRecorder(model, router_names, top_k=8) as recorder:
        ...     model(**inputs)
        >>> record = recorder.get_record()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        router_module_names: Sequence[str],
        top_k: int,
        *,
        record_inputs: bool = False,
    ) -> None:
        """
        Args:
            model: The loaded MoE model.
            router_module_names: One router module name per layer, in layer order.
            top_k: Number of experts selected per token.
            record_inputs: Also keep each router's input hidden state (float32, on the
                CPU). This costs ``layers * tokens * hidden_size * 4`` bytes per pass,
                so enable it only when the hidden-state shift is being measured.
        """
        self._model = model
        self._router_module_names = tuple(router_module_names)
        self._top_k = top_k
        self._record_inputs = record_inputs
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._indices: list[list[torch.Tensor]] = []
        self._probs: list[list[torch.Tensor]] = []
        self._inputs: list[list[torch.Tensor]] = []

    def __enter__(self) -> RoutingRecorder:
        """Attach a forward hook to every router module."""
        routers = resolve_routers(self._model, self._router_module_names)
        self._indices = [[] for _ in routers]
        self._probs = [[] for _ in routers]
        self._inputs = [[] for _ in routers]
        try:
            self._handles.append(attach_batch_size_guard(self._model))
            for layer, router in enumerate(routers):
                self._handles.append(router.register_forward_hook(self._make_hook(layer)))
        except BaseException:
            self._detach()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all hooks, even if the ``with`` body raised. Do not suppress errors."""
        self._detach()

    def get_record(self) -> RoutingRecord:
        """Assemble buffered routing into a :class:`RoutingRecord`.

        Raises:
            RuntimeError: If nothing was recorded (no forward pass ran), or more
                than one forward pass was recorded.
        """
        counts = {len(buffer) for buffer in self._indices}
        if counts == {0} or not counts:
            raise RuntimeError("nothing was recorded: no forward pass ran inside the with-block")
        if counts != {1}:
            raise RuntimeError(
                "recorded more than one forward pass (or an uneven number across layers); "
                "use one forward pass per RoutingRecorder"
            )
        return RoutingRecord(
            topk_indices=torch.stack([buffer[0] for buffer in self._indices]),
            router_probs=torch.stack([buffer[0] for buffer in self._probs]),
            router_inputs=(
                torch.stack([buffer[0] for buffer in self._inputs]) if self._record_inputs else None
            ),
        )

    def _make_hook(self, layer: int):
        name = self._router_module_names[layer]

        def hook(module, args, output) -> None:
            _, scores, indices = output
            if indices.shape[-1] != self._top_k:
                raise ValueError(
                    f"top_k mismatch: recorder expects {self._top_k} but router {name!r} "
                    f"selected {indices.shape[-1]} experts per token"
                )
            self._indices[layer].append(indices.detach().cpu().clone())
            self._probs[layer].append(scores.detach().float().cpu())
            if self._record_inputs:
                if not args:
                    raise RuntimeError(
                        f"cannot read the input of router {name!r}: it was called with keyword "
                        "arguments only"
                    )
                self._inputs[layer].append(
                    args[0].detach().to(device="cpu", dtype=torch.float32, copy=True)
                )

        return hook

    def _detach(self) -> None:
        handles, self._handles = self._handles, []
        remove_handles(handles)
