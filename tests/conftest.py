"""Shared test fixtures: small question factories and a download-free tokenizer."""

from __future__ import annotations

import re
from typing import Any

import pytest

from rr.data import MCQuestion

WORD = re.compile(r"\S+")


class WhitespaceTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer: one token per whitespace word."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        matches = list(WORD.finditer(text))
        encoded: dict[str, Any] = {"input_ids": [self._token_id(m.group()) for m in matches]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [m.span() for m in matches]
        return encoded

    @staticmethod
    def _token_id(word: str) -> int:
        return sum(ord(c) for c in word)


class TinyVocabTokenizer(WhitespaceTokenizer):
    """Whitespace tokenizer whose ids fit the tiny MoE's 128-token vocabulary."""

    @staticmethod
    def _token_id(word: str) -> int:
        return sum(ord(c) for c in word) % 120 + 4


def make_question(subject: str, index: int, words: int = 3) -> MCQuestion:
    stem = " ".join(f"{subject}{index}w{i}" for i in range(words))
    return MCQuestion(
        subject=subject,
        question=f"{stem}?",
        choices=tuple(f"{subject}{index}{letter}" for letter in "abcd"),
        answer_index=index % 4,
    )


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


@pytest.fixture
def make_questions():
    def _make(subject: str, count: int, words: int = 3) -> list[MCQuestion]:
        return [make_question(subject, i, words) for i in range(count)]

    return _make


# --- tiny real OLMoE (random weights, no downloads) -----------------------------------

TINY_NUM_LAYERS = 2
TINY_NUM_EXPERTS = 8
TINY_TOP_K = 2
TINY_ROUTER_NAMES = tuple(f"model.layers.{i}.mlp.gate" for i in range(TINY_NUM_LAYERS))


def build_tiny_moe(*, norm_topk_prob: bool = False, seed: int = 0):
    """A real ``OlmoeForCausalLM`` with random weights, small enough to run in milliseconds."""
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=TINY_NUM_EXPERTS,
        num_experts_per_tok=TINY_TOP_K,
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    return OlmoeForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tiny_moe():
    return build_tiny_moe()


@pytest.fixture
def router_names() -> tuple[str, ...]:
    return TINY_ROUTER_NAMES


@pytest.fixture
def input_ids():
    import torch

    return torch.tensor([[3, 17, 42, 8, 99, 5, 61, 23]])


# --- prompt-group builders shared by the pilot and forcing tests ----------------------------

ALL_CONDITIONS = ("zero_shot", "cross_domain", "same_domain", "irrelevant")


def build_group(tokenizer, index: int = 0, conditions=None):
    """All condition prompts for one target question (tiny shots so prompts stay short)."""
    from rr.prompts import PromptCondition, build_condition_prompts

    conditions = conditions or tuple(PromptCondition(c) for c in ALL_CONDITIONS)
    return build_condition_prompts(
        make_question("bio", index, words=3),
        source_examples=[make_question("chem", i) for i in range(2)],
        same_domain_examples=[make_question("bio", 50 + i) for i in range(2)],
        irrelevant_examples=[make_question("hist", i) for i in range(2)],
        conditions=conditions,
        tokenizer=tokenizer,
    )


def write_prompt_file(
    tmp_path, tokenizer, count: int = 3, name: str = "bio_med_a2b", conditions=None
):
    """Write ``count`` target questions' prompts to ``<tmp_path>/<name>.jsonl``."""
    from rr.prompts import save_prompts

    prompts = [p for i in range(count) for p in build_group(tokenizer, i, conditions).values()]
    path = tmp_path / f"{name}.jsonl"
    save_prompts(prompts, path)
    return path
