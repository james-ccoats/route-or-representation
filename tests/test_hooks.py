"""Tests for rr.routing._hooks: cleanup, router lookup, normalization flag, batch guard."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import TINY_ROUTER_NAMES, TINY_TOP_K, build_tiny_moe
from rr.routing._hooks import normalizes_topk_probs, remove_handles, resolve_routers
from rr.routing.force import RoutingForcer, random_spec
from rr.routing.record import RoutingRecorder


class _Handle:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.removed = False

    def remove(self) -> None:
        self.removed = True
        if self.error:
            raise self.error


def _router_hooks(model) -> list[int]:
    return [len(model.get_submodule(n)._forward_hooks) for n in TINY_ROUTER_NAMES]


def _spec():
    return random_spec(2, 8, TINY_TOP_K, [1, 2], seed=0)


# --- remove_handles ---------------------------------------------------------------


def test_remove_handles_keeps_removing_after_a_failure_and_reraises() -> None:
    handles = [_Handle(), _Handle(RuntimeError("first")), _Handle(), _Handle(ValueError("second"))]

    with pytest.raises(RuntimeError, match="first"):
        remove_handles(handles)

    assert all(h.removed for h in handles)


def test_remove_handles_is_silent_when_everything_succeeds() -> None:
    handles = [_Handle(), _Handle()]

    remove_handles(handles)

    assert all(h.removed for h in handles)


# --- router lookup and normalization flag --------------------------------------------


def test_resolve_routers_returns_modules_in_name_order(tiny_moe) -> None:
    routers = resolve_routers(tiny_moe, TINY_ROUTER_NAMES)

    assert routers == [tiny_moe.get_submodule(n) for n in TINY_ROUTER_NAMES]


def test_normalizes_topk_probs_requires_the_attribute() -> None:
    with pytest.raises(RuntimeError, match="norm_topk_prob"):
        normalizes_topk_probs(nn.Linear(2, 2))


# --- registration failure unwinds ------------------------------------------------------


def _fail_registration(model, layer: int) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("cannot register")

    model.get_submodule(TINY_ROUTER_NAMES[layer]).register_forward_hook = boom


def test_recorder_unwinds_when_registration_fails_midway() -> None:
    model = build_tiny_moe()  # own model: the failing hook is patched onto an instance
    _fail_registration(model, layer=1)

    with pytest.raises(RuntimeError, match="cannot register"):
        RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K).__enter__()

    assert _router_hooks(model)[0] == 0
    assert len(model._forward_pre_hooks) == 0


def test_forcer_unwinds_when_registration_fails_midway() -> None:
    model = build_tiny_moe()
    _fail_registration(model, layer=1)

    with pytest.raises(RuntimeError, match="cannot register"):
        RoutingForcer(model, TINY_ROUTER_NAMES, _spec()).__enter__()

    assert _router_hooks(model)[0] == 0
    assert len(model._forward_pre_hooks) == 0


# --- batch-size guard ---------------------------------------------------------------


@pytest.fixture
def batch_of_two() -> torch.Tensor:
    return torch.tensor([[3, 17, 42, 8], [4, 18, 43, 9]])


def test_recorder_rejects_batches_larger_than_one(batch_of_two) -> None:
    model = build_tiny_moe()

    with pytest.raises(ValueError, match="batch size"):
        with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K):
            model(input_ids=batch_of_two)

    assert _router_hooks(model) == [0, 0]
    assert len(model._forward_pre_hooks) == 0


def test_forcer_rejects_batches_larger_than_one(batch_of_two) -> None:
    model = build_tiny_moe()

    with pytest.raises(ValueError, match="batch size"):
        with RoutingForcer(model, TINY_ROUTER_NAMES, _spec()):
            model(input_ids=batch_of_two)

    assert _router_hooks(model) == [0, 0]
    assert len(model._forward_pre_hooks) == 0


def test_guard_also_covers_inputs_embeds(batch_of_two) -> None:
    model = build_tiny_moe()
    embeds = model.get_input_embeddings()(batch_of_two)

    with pytest.raises(ValueError, match="batch size"):
        with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K):
            model(inputs_embeds=embeds)


def test_batch_size_one_is_accepted(tiny_moe, input_ids) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        tiny_moe(input_ids=input_ids)

    assert recorder.get_record().topk_indices.shape[1] == input_ids.shape[1]
