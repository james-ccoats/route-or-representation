"""Sanity checks for rr.routing.force: identity, scope, error safety, spec builders."""

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
from rr.routing.force import ForcingSpec, RoutingForcer, random_spec, spec_from_record
from rr.routing.record import RoutingRecorder

TARGET_POSITIONS = (3, 4, 5)


def _record_routing(model, input_ids):
    with torch.inference_mode(), RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
        logits = model(input_ids=input_ids).logits
    return recorder.get_record(), logits


def _forced_run(model, input_ids, spec):
    """Run under the forcer, recording the routing that was actually used downstream."""
    with torch.inference_mode(), RoutingForcer(model, TINY_ROUTER_NAMES, spec):
        with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
            logits = model(input_ids=input_ids).logits
    return recorder.get_record(), logits


def _hook_counts(model) -> list[int]:
    return [len(model.get_submodule(n)._forward_hooks) for n in TINY_ROUTER_NAMES]


def _own_routing_spec(record, positions=TARGET_POSITIONS, **kwargs) -> ForcingSpec:
    return spec_from_record(record, positions, positions, **kwargs)


# --- identity ---------------------------------------------------------------------


@pytest.mark.parametrize("force_weights", [False, True])
def test_forcing_own_routing_leaves_logits_unchanged(tiny_moe, input_ids, force_weights) -> None:
    record, baseline = _record_routing(tiny_moe, input_ids)
    spec = _own_routing_spec(record, force_weights=force_weights)

    _, forced = _forced_run(tiny_moe, input_ids, spec)

    assert torch.allclose(baseline, forced, atol=1e-6)


def test_forcing_own_routing_on_every_position_leaves_logits_unchanged(tiny_moe, input_ids) -> None:
    record, baseline = _record_routing(tiny_moe, input_ids)
    everything = tuple(range(input_ids.shape[1]))

    _, forced = _forced_run(tiny_moe, input_ids, _own_routing_spec(record, everything))

    assert torch.allclose(baseline, forced, atol=1e-6)


def test_forcing_own_routing_with_normalized_topk_leaves_logits_unchanged(input_ids) -> None:
    model = build_tiny_moe(norm_topk_prob=True)
    record, baseline = _record_routing(model, input_ids)

    _, forced = _forced_run(model, input_ids, _own_routing_spec(record))

    assert torch.allclose(baseline, forced, atol=1e-6)


# --- the intervention does something ------------------------------------------------


def test_forcing_random_experts_changes_outputs(tiny_moe, input_ids) -> None:
    _, baseline = _record_routing(tiny_moe, input_ids)
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=1)

    _, forced = _forced_run(tiny_moe, input_ids, spec)

    assert not torch.allclose(baseline, forced, atol=1e-4)


def test_forced_routing_is_what_the_model_actually_uses(tiny_moe, input_ids) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=2)

    used, _ = _forced_run(tiny_moe, input_ids, spec)

    for layer in range(TINY_NUM_LAYERS):
        assert torch.equal(
            used.topk_indices[layer, list(TARGET_POSITIONS)], spec.topk_indices[layer]
        )


def test_forced_weights_are_used_when_provided(tiny_moe, input_ids) -> None:
    record, _ = _record_routing(tiny_moe, input_ids)
    uniform = torch.full((TINY_NUM_LAYERS, len(TARGET_POSITIONS), TINY_TOP_K), 1 / TINY_TOP_K)
    spec = ForcingSpec(
        token_positions=torch.tensor(TARGET_POSITIONS),
        topk_indices=record.topk_indices[:, list(TARGET_POSITIONS)],
        topk_weights=uniform,
    )

    used, _ = _forced_run(tiny_moe, input_ids, spec)

    assert torch.allclose(used.router_probs[:, list(TARGET_POSITIONS)], uniform)


# --- scope ------------------------------------------------------------------------


def test_forcing_leaves_earlier_positions_and_layer_zero_untouched(tiny_moe, input_ids) -> None:
    baseline, _ = _record_routing(tiny_moe, input_ids)
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=3)
    others = [p for p in range(input_ids.shape[1]) if p not in TARGET_POSITIONS]
    earlier = [p for p in others if p < min(TARGET_POSITIONS)]

    used, _ = _forced_run(tiny_moe, input_ids, spec)

    # Layer 0 sees the same hidden states at un-forced positions, so it routes identically.
    assert torch.equal(used.topk_indices[0, others], baseline.topk_indices[0, others])
    # Causal attention: nothing before the first forced position can change in any layer.
    assert torch.equal(used.topk_indices[:, earlier], baseline.topk_indices[:, earlier])


def test_forcing_only_selected_layers_leaves_other_layers_natural(tiny_moe, input_ids) -> None:
    baseline, _ = _record_routing(tiny_moe, input_ids)
    spec = random_spec(
        TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=4, layers=[1]
    )

    used, _ = _forced_run(tiny_moe, input_ids, spec)

    assert torch.equal(used.topk_indices[0], baseline.topk_indices[0])
    assert torch.equal(used.topk_indices[1, list(TARGET_POSITIONS)], spec.topk_indices[1])


def test_routing_transplants_between_prompts_at_aligned_positions(tiny_moe) -> None:
    source = torch.tensor([[11, 12, 13, 8, 99, 5, 61, 23]])
    target = torch.tensor([[7, 70, 8, 99, 5, 61, 23, 1]])  # same tail tokens, shifted by one
    source_record, _ = _record_routing(tiny_moe, source)
    spec = spec_from_record(source_record, [4, 5, 6], [3, 4, 5])

    used, _ = _forced_run(tiny_moe, target, spec)

    assert torch.equal(used.topk_indices[:, [3, 4, 5]], source_record.topk_indices[:, [4, 5, 6]])


# --- hook safety ------------------------------------------------------------------


def test_forcer_removes_hooks_on_normal_exit(tiny_moe, input_ids) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=5)

    with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
        assert _hook_counts(tiny_moe) == [1, 1]
        tiny_moe(input_ids=input_ids)

    assert _hook_counts(tiny_moe) == [0, 0]


def test_recorder_and_forcer_remove_hooks_on_error(tiny_moe) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=6)

    with pytest.raises(RuntimeError, match="boom"):
        with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
            with RoutingRecorder(tiny_moe, TINY_ROUTER_NAMES, TINY_TOP_K):
                raise RuntimeError("boom")

    assert _hook_counts(tiny_moe) == [0, 0]


def test_forcer_only_hooks_the_requested_layers(tiny_moe) -> None:
    spec = random_spec(
        TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=7, layers=[1]
    )

    with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
        assert _hook_counts(tiny_moe) == [0, 1]

    assert _hook_counts(tiny_moe) == [0, 0]


def test_forcer_leaves_no_hooks_when_forward_fails_on_bad_positions(tiny_moe, input_ids) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K, (100,), seed=8)

    with pytest.raises(ValueError, match="position"):
        with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
            tiny_moe(input_ids=input_ids)

    assert _hook_counts(tiny_moe) == [0, 0]


def test_forcer_rejects_spec_with_wrong_number_of_layers(tiny_moe) -> None:
    spec = random_spec(TINY_NUM_LAYERS + 1, TINY_NUM_EXPERTS, TINY_TOP_K, TARGET_POSITIONS, seed=9)

    with pytest.raises(ValueError, match="layers"):
        RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec).__enter__()

    assert _hook_counts(tiny_moe) == [0, 0]


def test_forcer_rejects_expert_ids_outside_the_model(tiny_moe, input_ids) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS + 5, TINY_TOP_K, TARGET_POSITIONS, seed=10)
    spec = ForcingSpec(
        token_positions=spec.token_positions,
        topk_indices=spec.topk_indices.clamp(min=TINY_NUM_EXPERTS),
    )

    with pytest.raises(ValueError, match="expert"):
        with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
            tiny_moe(input_ids=input_ids)


def test_forcer_rejects_top_k_mismatch(tiny_moe, input_ids) -> None:
    spec = random_spec(TINY_NUM_LAYERS, TINY_NUM_EXPERTS, TINY_TOP_K + 1, TARGET_POSITIONS, seed=11)

    with pytest.raises(ValueError, match="top_k"):
        with RoutingForcer(tiny_moe, TINY_ROUTER_NAMES, spec):
            tiny_moe(input_ids=input_ids)


# --- spec builders and validation ---------------------------------------------------


def test_random_spec_is_reproducible_for_a_seed() -> None:
    a = random_spec(2, 8, 2, [1, 2], seed=0)
    b = random_spec(2, 8, 2, [1, 2], seed=0)
    c = random_spec(2, 8, 2, [1, 2], seed=1)

    assert torch.equal(a.topk_indices, b.topk_indices)
    assert not torch.equal(a.topk_indices, c.topk_indices)


def test_random_spec_picks_distinct_valid_experts_per_token() -> None:
    spec = random_spec(3, 8, 4, [0, 1, 2, 3, 4], seed=0)

    assert spec.topk_indices.shape == (3, 5, 4)
    assert spec.topk_indices.min() >= 0
    assert spec.topk_indices.max() < 8
    assert all(len(set(row.tolist())) == 4 for row in spec.topk_indices.view(-1, 4))


def test_random_spec_rejects_top_k_above_num_experts() -> None:
    with pytest.raises(ValueError, match="top_k"):
        random_spec(2, 4, 5, [0], seed=0)


def test_spec_from_record_pairs_source_routing_with_target_positions() -> None:
    indices = torch.arange(2 * 6 * 2).view(2, 6, 2)
    probs = indices.float() / 100
    from rr.routing.record import RoutingRecord

    record = RoutingRecord(topk_indices=indices, router_probs=probs)

    spec = spec_from_record(record, [4, 5], [1, 2], force_weights=True, layers=[1])

    assert spec.token_positions.tolist() == [1, 2]
    assert torch.equal(spec.topk_indices, indices[:, [4, 5]])
    assert torch.equal(spec.topk_weights, probs[:, [4, 5]])
    assert spec.layers == (1,)


def test_spec_from_record_keeps_router_weights_by_default() -> None:
    from rr.routing.record import RoutingRecord

    indices = torch.zeros(2, 4, 2, dtype=torch.long)
    record = RoutingRecord(topk_indices=indices, router_probs=torch.ones(2, 4, 2))

    assert spec_from_record(record, [0], [0]).topk_weights is None


def test_spec_from_record_rejects_length_mismatch() -> None:
    from rr.routing.record import RoutingRecord

    indices = torch.zeros(2, 4, 2, dtype=torch.long)
    record = RoutingRecord(topk_indices=indices, router_probs=torch.ones(2, 4, 2))

    with pytest.raises(ValueError, match="positions"):
        spec_from_record(record, [0, 1], [0])


def test_spec_from_record_rejects_unknown_layers() -> None:
    from rr.routing.record import RoutingRecord

    indices = torch.zeros(2, 4, 2, dtype=torch.long)
    record = RoutingRecord(topk_indices=indices, router_probs=torch.ones(2, 4, 2))

    with pytest.raises(ValueError, match="layers"):
        spec_from_record(record, [0], [0], layers=[2])


@pytest.mark.parametrize(
    ("positions", "indices", "match"),
    [
        (torch.tensor([0, 0]), torch.zeros(2, 2, 2, dtype=torch.long), "unique"),
        (torch.tensor([-1]), torch.zeros(2, 1, 2, dtype=torch.long), "non-negative"),
        (torch.tensor([0, 1]), torch.zeros(2, 3, 2, dtype=torch.long), "shape"),
        (torch.tensor([[0, 1]]), torch.zeros(2, 2, 2, dtype=torch.long), "1-D"),
    ],
)
def test_forcing_spec_validates_its_shapes(positions, indices, match) -> None:
    with pytest.raises(ValueError, match=match):
        ForcingSpec(token_positions=positions, topk_indices=indices)


@pytest.mark.parametrize("field", ["token_positions", "topk_indices"])
def test_forcing_spec_requires_integer_tensors(field) -> None:
    kwargs = {
        "token_positions": torch.tensor([0, 1]),
        "topk_indices": torch.zeros(2, 2, 2, dtype=torch.long),
    }
    kwargs[field] = kwargs[field].float()

    with pytest.raises(ValueError, match="integer"):
        ForcingSpec(**kwargs)
