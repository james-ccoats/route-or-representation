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
    # TODO(Phase 2): tokenize with offsets, select tokens overlapping the span.
    raise NotImplementedError("TODO(Phase 2): char span -> token positions")


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
    # TODO(Phase 2): call target_token_positions for both and assert equal token ids.
    raise NotImplementedError("TODO(Phase 2): align target tokens across prompts")
