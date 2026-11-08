"""Tests for rr.models: MoE structure discovery and verification (no downloads)."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from conftest import (
    TINY_NUM_EXPERTS,
    TINY_NUM_LAYERS,
    TINY_ROUTER_NAMES,
    TINY_TOP_K,
    build_tiny_moe,
)
from rr.config import ArchitectureConfig
from rr.models import inspect_moe_structure, verify_structure


def _expected(**overrides) -> ArchitectureConfig:
    values = {
        "num_layers": TINY_NUM_LAYERS,
        "num_experts": TINY_NUM_EXPERTS,
        "top_k": TINY_TOP_K,
        "router_module_pattern": "model.layers.{layer}.mlp.gate",
    }
    return ArchitectureConfig(**{**values, **overrides})


def test_inspect_reads_structure_off_the_loaded_model(tiny_moe) -> None:
    structure = inspect_moe_structure(tiny_moe)

    assert structure.num_layers == TINY_NUM_LAYERS
    assert structure.num_experts == TINY_NUM_EXPERTS
    assert structure.top_k == TINY_TOP_K
    assert structure.router_module_names == TINY_ROUTER_NAMES
    assert structure.normalizes_topk_probs is False


def test_inspect_detects_topk_renormalization() -> None:
    assert inspect_moe_structure(build_tiny_moe(norm_topk_prob=True)).normalizes_topk_probs is True


def test_inspect_raises_when_the_model_has_no_routers() -> None:
    dense = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=1, n_head=2))

    with pytest.raises(RuntimeError, match="router"):
        inspect_moe_structure(dense)


def test_verify_structure_accepts_a_matching_config(tiny_moe) -> None:
    verify_structure(inspect_moe_structure(tiny_moe), _expected())


def test_verify_structure_lists_every_mismatch(tiny_moe) -> None:
    structure = inspect_moe_structure(tiny_moe)

    with pytest.raises(ValueError) as error:
        verify_structure(
            structure,
            _expected(num_layers=16, num_experts=64, top_k=8, router_module_pattern="x.{layer}.g"),
        )

    message = str(error.value)
    for field in ("num_layers", "num_experts", "top_k", "router_module"):
        assert field in message


def _router(model, layer: int):
    return model.get_submodule(TINY_ROUTER_NAMES[layer])


def test_inspect_raises_when_routers_disagree_on_top_k() -> None:
    model = build_tiny_moe()
    _router(model, 1).top_k = TINY_TOP_K + 1

    with pytest.raises(RuntimeError, match="disagree"):
        inspect_moe_structure(model)


def test_inspect_raises_when_router_weight_rows_differ_from_num_experts() -> None:
    model = build_tiny_moe()
    router = _router(model, 0)
    router.weight = torch.nn.Parameter(torch.zeros(TINY_NUM_EXPERTS - 1, router.weight.shape[1]))

    with pytest.raises(RuntimeError, match="weight rows"):
        inspect_moe_structure(model)


def test_inspect_raises_when_model_config_disagrees_with_the_modules() -> None:
    model = build_tiny_moe()
    model.config.num_experts = TINY_NUM_EXPERTS * 2

    with pytest.raises(RuntimeError, match=r"model\.config\.num_experts"):
        inspect_moe_structure(model)
