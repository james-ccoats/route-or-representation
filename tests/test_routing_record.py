"""Tests for rr.routing.record: RoutingRecord slicing and RoutingRecorder hooks."""

from __future__ import annotations

import pytest
import torch

from conftest import (
    TINY_NUM_EXPERTS,
    TINY_NUM_LAYERS,
    TINY_ROUTER_NAMES,
    TINY_TOP_K,
    build_tiny_moe,
)
from rr.routing.record import RoutingRecord, RoutingRecorder


def _record() -> RoutingRecord:
    num_layers, num_tokens, top_k = 2, 5, 3
    indices = torch.arange(num_layers * num_tokens * top_k).view(num_layers, num_tokens, top_k)
    return RoutingRecord(topk_indices=indices, router_probs=indices.float() / 100)


def _hook_counts(model, names) -> list[int]:
    return [len(model.get_submodule(name)._forward_hooks) for name in names]


# --- RoutingRecord.select_positions -----------------------------------------------


def test_select_positions_keeps_only_requested_tokens_in_order() -> None:
    record = _record()

    selected = record.select_positions([3, 1])

    assert selected.topk_indices.shape == (2, 2, 3)
    assert torch.equal(selected.topk_indices[:, 0], record.topk_indices[:, 3])
    assert torch.equal(selected.router_probs[:, 1], record.router_probs[:, 1])


def test_select_positions_returns_a_new_record_and_leaves_the_original() -> None:
    record = _record()
    before = record.topk_indices.clone()

    selected = record.select_positions([0, 1])

    assert selected is not record
    assert torch.equal(record.topk_indices, before)


def test_select_positions_rejects_out_of_range_positions() -> None:
    with pytest.raises(IndexError):
        _record().select_positions([5])


def test_select_positions_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="empty"):
        _record().select_positions([])


# --- RoutingRecorder --------------------------------------------------------------


def test_recorded_shapes_are_layers_by_tokens_by_top_k(tiny_moe, input_ids) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        tiny_moe(input_ids=input_ids)

    record = recorder.get_record()

    expected = (TINY_NUM_LAYERS, input_ids.shape[1], TINY_TOP_K)
    assert record.topk_indices.shape == expected
    assert record.router_probs.shape == expected


def test_recorded_top_k_matches_the_experts_the_model_actually_selects(input_ids) -> None:
    # Own model: transformers leaves a permanent capture hook on routers when
    # output_router_logits=True, which would pollute the shared fixture's hook counts.
    model = build_tiny_moe()
    with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        output = model(input_ids=input_ids, output_router_logits=True)

    record = recorder.get_record()

    for layer, router_logits in enumerate(output.router_logits):
        probs = router_logits.float().softmax(dim=-1)
        top_values, top_indices = probs.topk(TINY_TOP_K, dim=-1)
        assert torch.equal(record.topk_indices[layer], top_indices)
        assert torch.allclose(record.router_probs[layer], top_values, atol=1e-6)


def test_recorded_experts_are_valid_and_distinct_per_token(tiny_moe, input_ids) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        tiny_moe(input_ids=input_ids)

    indices = recorder.get_record().topk_indices

    assert indices.min() >= 0
    assert indices.max() < TINY_NUM_EXPERTS
    assert all(len(set(row.tolist())) == TINY_TOP_K for row in indices.view(-1, TINY_TOP_K))


def test_recording_does_not_change_model_outputs(tiny_moe, input_ids) -> None:
    with torch.inference_mode():
        baseline = tiny_moe(input_ids=input_ids).logits
        with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K):
            recorded = tiny_moe(input_ids=input_ids).logits

    assert torch.equal(baseline, recorded)


def test_recorder_removes_hooks_on_normal_exit(tiny_moe, input_ids) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K):
        assert _hook_counts(tiny_moe, TINY_ROUTER_NAMES) == [1, 1]
        tiny_moe(input_ids=input_ids)

    assert _hook_counts(tiny_moe, TINY_ROUTER_NAMES) == [0, 0]


def test_recorder_removes_hooks_when_the_body_raises(tiny_moe) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K):
            raise RuntimeError("boom")

    assert _hook_counts(tiny_moe, TINY_ROUTER_NAMES) == [0, 0]


def test_recorder_with_unknown_router_name_raises_and_leaves_no_hooks(tiny_moe) -> None:
    names = (TINY_ROUTER_NAMES[0], "model.layers.9.mlp.gate")

    with pytest.raises(ValueError, match="model.layers.9.mlp.gate"):
        with RoutingRecorder(tiny_moe, names, TINY_TOP_K):
            pass

    assert _hook_counts(tiny_moe, TINY_ROUTER_NAMES) == [0, 0]


def test_get_record_before_any_forward_pass_raises(tiny_moe) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        pass

    with pytest.raises(RuntimeError, match="nothing"):
        recorder.get_record()


def test_get_record_after_two_forward_passes_raises(tiny_moe, input_ids) -> None:
    with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        tiny_moe(input_ids=input_ids)
        tiny_moe(input_ids=input_ids)

    with pytest.raises(RuntimeError, match="one forward pass"):
        recorder.get_record()


def test_recorder_top_k_mismatch_with_model_raises(input_ids) -> None:
    model = build_tiny_moe()

    with pytest.raises(ValueError, match="top_k"):
        with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K + 1):
            model(input_ids=input_ids)


def test_recorded_probs_are_renormalized_when_the_model_normalizes(input_ids) -> None:
    model = build_tiny_moe(norm_topk_prob=True)

    with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        model(input_ids=input_ids)

    sums = recorder.get_record().router_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
