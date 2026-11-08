"""Helpers shared by the recorder and the forcer: router lookup and hook cleanup."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedModel

BATCH_INPUT_KEYS = ("input_ids", "inputs_embeds")


def resolve_routers(model: PreTrainedModel, names: Sequence[str]) -> list[nn.Module]:
    """Look up every router module by name before any hook is attached.

    Raises:
        ValueError: If a name does not resolve, listing the offending name.
    """
    modules: list[nn.Module] = []
    for name in names:
        try:
            modules.append(model.get_submodule(name))
        except AttributeError as error:
            raise ValueError(f"router module {name!r} not found in the model") from error
    return modules


def remove_handles(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove every handle, even if one removal raises."""
    first_error: Exception | None = None
    for handle in handles:
        try:
            handle.remove()
        except Exception as error:  # keep removing the rest, re-raise below
            first_error = first_error or error
    if first_error is not None:
        raise first_error


def normalizes_topk_probs(router: nn.Module) -> bool:
    """Whether the router renormalizes its top-k probabilities to sum to 1.

    Raises:
        RuntimeError: If the router does not expose ``norm_topk_prob``; the
            answer must be read off the model, not guessed.
    """
    if not hasattr(router, "norm_topk_prob"):
        raise RuntimeError(
            f"{type(router).__name__} has no 'norm_topk_prob'; cannot tell whether its "
            "top-k weights are renormalized"
        )
    return bool(router.norm_topk_prob)


def attach_batch_size_guard(model: PreTrainedModel) -> torch.utils.hooks.RemovableHandle:
    """Reject forward passes with batch size > 1 while routing hooks are attached.

    Routers see flattened ``[batch * tokens, ...]`` rows, so token positions only
    mean what we think they mean for a single sequence. Batches would otherwise be
    recorded or forced on the wrong rows without any error.

    Returns:
        The handle of the model-level pre-hook, to be removed with the router hooks.
    """

    def guard(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        batch_size = _input_batch_size(args, kwargs)
        if batch_size is not None and batch_size != 1:
            raise ValueError(
                f"routing hooks support batch size 1 only, got batch size {batch_size}"
            )

    return model.register_forward_pre_hook(guard, with_kwargs=True)


def _input_batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Batch size of the model input, or ``None`` if it cannot be determined."""
    candidates = [kwargs.get(key) for key in BATCH_INPUT_KEYS] + list(args[:1])
    for candidate in candidates:
        if isinstance(candidate, torch.Tensor) and candidate.dim() >= 1:
            return candidate.shape[0]
    return None
