"""Align target-question token positions across prompts.

Role in the project: the target question is the same text in the zero-shot and
few-shot prompts but sits at different token offsets (few-shot has a long
prefix). To compare or transplant routing we need, for each target-question
token, its position in each prompt. Used in Phase 2 (routing-shift metrics)
and Phase 3 (forcing).

Methodological notes:

* Tokenization can merge tokens across the boundary between examples and the
  target question, so naive "last N tokens" alignment can be off by one. Prefer
  offset mapping (``return_offsets_mapping=True``) from the target character
  span in :class:`rr.prompts.BuiltPrompt`.
* Alignment must be validated: the token ids at aligned positions in both
  prompts must be identical. Fail loudly otherwise.
* Causal attention means the *same* token can have different hidden states in
  the two prompts; that difference is exactly the "representation" effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from rr.prompts import BuiltPrompt


@dataclass(frozen=True)
class TokenAlignment:
    """Aligned target-question positions in two prompts.

    Attributes:
        positions_a: Token positions of the target question in prompt A.
        positions_b: Token positions of the same tokens in prompt B (same length).
    """

    positions_a: tuple[int, ...]
    positions_b: tuple[int, ...]


def target_token_positions(
    tokenizer: PreTrainedTokenizerBase, prompt: BuiltPrompt
) -> tuple[int, ...]:
    """Token positions covering ``prompt.target_char_span``.

    Args:
        tokenizer: Tokenizer used for the model.
        prompt: Prompt with its target character span.

    Returns:
        Sorted token positions (including BOS offset, if any) of the target block.
    """
    positions, _ = _encode_target(tokenizer, prompt)
    return positions


def _encode_target(
    tokenizer: PreTrainedTokenizerBase, prompt: BuiltPrompt
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Positions and token ids of the target block, using the model's default encoding."""
    encoded = tokenizer(prompt.text, return_offsets_mapping=True)
    span_start, span_end = prompt.target_char_span
    positions: list[int] = []
    token_ids: list[int] = []
    for position, (token_id, (start, end)) in enumerate(
        zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
    ):
        if start == end:  # special tokens (e.g. BOS) have an empty span
            continue
        if start < span_start < end or start < span_end < end:
            raise ValueError(
                f"a token straddles the target boundary in the {prompt.condition} prompt "
                f"(token span {start}-{end}, target span {span_start}-{span_end})"
            )
        if start >= span_start and end <= span_end:
            positions.append(position)
            token_ids.append(token_id)
    if not positions:
        raise ValueError(f"no tokens fall inside the target span of the {prompt.condition} prompt")
    return tuple(positions), tuple(token_ids)


def align_target_tokens(
    tokenizer: PreTrainedTokenizerBase, prompt_a: BuiltPrompt, prompt_b: BuiltPrompt
) -> TokenAlignment:
    """Map target-question tokens between two prompts of the same target.

    Args:
        tokenizer: Tokenizer used for the model.
        prompt_a: First prompt (e.g. zero-shot).
        prompt_b: Second prompt (e.g. few-shot).

    Returns:
        A :class:`TokenAlignment`.

    Raises:
        ValueError: If the prompts have different targets or the aligned token
            ids differ.
    """
    if prompt_a.target != prompt_b.target:
        raise ValueError("prompts do not share the same target question")
    positions_a, ids_a = _encode_target(tokenizer, prompt_a)
    positions_b, ids_b = _encode_target(tokenizer, prompt_b)
    if ids_a != ids_b:
        raise ValueError(
            f"target token ids differ between the {prompt_a.condition} and "
            f"{prompt_b.condition} prompts"
        )
    return TokenAlignment(positions_a=positions_a, positions_b=positions_b)
