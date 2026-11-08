"""Tests for rr.routing.align: target-question token positions across prompts."""

from __future__ import annotations

import re

import pytest

from conftest import WhitespaceTokenizer
from rr.prompts import BuiltPrompt, PromptCondition, build_prompt
from rr.routing.align import align_target_tokens, target_token_positions
from rr.scoring import format_mc_prompt


class BosTokenizer(WhitespaceTokenizer):
    """Whitespace tokenizer that prepends a BOS token with an empty offset span."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        encoded = super().__call__(text, add_special_tokens, return_offsets_mapping)
        if add_special_tokens:
            encoded["input_ids"] = [1, *encoded["input_ids"]]
            if return_offsets_mapping:
                encoded["offset_mapping"] = [(0, 0), *encoded["offset_mapping"]]
        return encoded


class BoundaryMergingTokenizer(WhitespaceTokenizer):
    """Merges the example/target separator with 'Question:' into a single token."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        matches = list(re.finditer(r"\n\nQuestion:|\S+", text))
        encoded = {"input_ids": [self._token_id(m.group()) for m in matches]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [m.span() for m in matches]
        return encoded


class FirstTokenQuirkTokenizer(WhitespaceTokenizer):
    """Encodes the first token of a text differently, like some SentencePiece tokenizers."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        encoded = super().__call__(text, add_special_tokens, return_offsets_mapping)
        if encoded["input_ids"]:
            encoded["input_ids"][0] += 1000
        return encoded


def _prompts(make_questions, shots: int = 2) -> tuple[BuiltPrompt, BuiltPrompt]:
    target = make_questions("bio", 1)[0]
    examples = make_questions("chem", shots)
    zero = build_prompt(PromptCondition.ZERO_SHOT, target, [])
    few = build_prompt(PromptCondition.CROSS_DOMAIN, target, examples)
    return zero, few


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def test_aligned_positions_have_identical_token_ids(tokenizer, make_questions) -> None:
    zero, few = _prompts(make_questions)

    alignment = align_target_tokens(tokenizer, zero, few)

    ids_zero = tokenizer(zero.text)["input_ids"]
    ids_few = tokenizer(few.text)["input_ids"]
    assert [ids_zero[i] for i in alignment.positions_a] == [
        ids_few[i] for i in alignment.positions_b
    ]


def test_alignment_covers_the_whole_target_block_including_answer_cue(
    tokenizer, make_questions
) -> None:
    zero, few = _prompts(make_questions)
    target_len = len(_ids(tokenizer, format_mc_prompt(zero.target)))

    positions = target_token_positions(tokenizer, few)

    ids_few = tokenizer(few.text)["input_ids"]
    assert len(positions) == target_len
    assert positions == tuple(range(positions[0], positions[0] + target_len))
    assert positions[-1] == len(ids_few) - 1  # the final "Answer:" token


def test_few_shot_positions_are_offset_by_the_example_prefix(tokenizer, make_questions) -> None:
    zero, few = _prompts(make_questions)

    alignment = align_target_tokens(tokenizer, zero, few)

    prefix_len = len(_ids(tokenizer, few.text[: few.target_char_span[0]]))
    assert alignment.positions_a[0] == 0
    assert alignment.positions_b[0] == prefix_len


def test_special_tokens_shift_positions_and_are_excluded(make_questions) -> None:
    zero, _ = _prompts(make_questions)

    positions = target_token_positions(BosTokenizer(), zero)

    assert positions[0] == 1  # position 0 is BOS, which has an empty span
    assert len(positions) == len(_ids(BosTokenizer(), zero.text))


def test_alignment_rejects_a_token_merged_across_the_example_boundary(make_questions) -> None:
    zero, few = _prompts(make_questions)

    with pytest.raises(ValueError, match="straddles"):
        align_target_tokens(BoundaryMergingTokenizer(), zero, few)


def test_alignment_rejects_prompts_with_different_targets(tokenizer, make_questions) -> None:
    zero, _ = _prompts(make_questions)
    other_target = make_questions("phys", 1)[0]
    other = build_prompt(PromptCondition.ZERO_SHOT, other_target, [])

    with pytest.raises(ValueError, match="target"):
        align_target_tokens(tokenizer, zero, other)


def test_alignment_rejects_differing_token_ids(make_questions) -> None:
    zero, few = _prompts(make_questions)

    with pytest.raises(ValueError, match="token ids differ"):
        align_target_tokens(FirstTokenQuirkTokenizer(), zero, few)
