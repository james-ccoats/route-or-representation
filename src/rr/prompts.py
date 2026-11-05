"""Prompt construction for the four experimental conditions.

Role in the project: turns a target question plus a pool of candidate examples
into a prompt for each condition, and records where the *target question*
sits in the text so that routing can later be recorded/forced on exactly those
tokens. Used from Phase 1 (building prompts) through Phase 3.

Conditions (per target question):

* ``zero_shot``     -- the target question alone.
* ``cross_domain``  -- k examples from the *source* domain, then the target.
* ``same_domain``   -- k examples from the *target* domain (upper reference).
* ``irrelevant``    -- examples unrelated to the task with token length matched
  to the cross-domain prompt (control for "more context" effects).

Layout of every prompt: ``[examples...][target question block]``. The target
question block must be identical across conditions so that its tokens can be
aligned (see :mod:`rr.routing.align`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from rr.data import MCQuestion

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class PromptCondition(StrEnum):
    """The four prompt conditions."""

    ZERO_SHOT = "zero_shot"
    CROSS_DOMAIN = "cross_domain"
    SAME_DOMAIN = "same_domain"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class BuiltPrompt:
    """A fully rendered prompt plus the metadata needed downstream.

    Attributes:
        condition: Which condition this prompt belongs to.
        text: Full prompt string passed to the tokenizer.
        target: The target question being asked.
        target_char_span: ``(start, end)`` character offsets of the target
            question block within ``text``, from the start of the question
            through the final "Answer:" cue (i.e. every token whose routing
            will be recorded/forced).
        num_examples: Number of in-context examples (0 for zero-shot).
    """

    condition: PromptCondition
    text: str
    target: MCQuestion
    target_char_span: tuple[int, int]
    num_examples: int


def build_prompt(
    condition: PromptCondition, target: MCQuestion, examples: Sequence[MCQuestion]
) -> BuiltPrompt:
    """Render one prompt.

    Args:
        condition: Condition label stored on the result.
        target: The question to be answered.
        examples: In-context examples, in presentation order (empty for zero-shot).

    Returns:
        A :class:`BuiltPrompt` with ``target_char_span`` filled in.

    Methodological note:
        Use :func:`rr.scoring.format_mc_prompt` to render every question so the
        target block is formatted identically in all conditions.
    """
    # TODO(Phase 1): render examples (with answers) then the target (without),
    # record the target block's character span.
    raise NotImplementedError("TODO(Phase 1): render prompt and target_char_span")


def select_irrelevant_examples(
    pool: Sequence[MCQuestion],
    token_budget: int,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
) -> list[MCQuestion]:
    """Pick control examples whose rendered length matches ``token_budget``.

    Args:
        pool: Candidate irrelevant questions (e.g. from unrelated subjects).
        token_budget: Token length of the cross-domain example block to match.
        tokenizer: Tokenizer used to measure length.
        seed: RNG seed.

    Returns:
        Examples whose total rendered token count is as close as possible to
        ``token_budget`` (tolerance to be decided; see docs/design_decisions.md).
    """
    # TODO(Phase 1): seeded selection + length matching (greedy or trim).
    raise NotImplementedError("TODO(Phase 1): token-length-matched irrelevant examples")


def build_condition_prompts(
    target: MCQuestion,
    *,
    source_examples: Sequence[MCQuestion],
    same_domain_examples: Sequence[MCQuestion],
    irrelevant_pool: Sequence[MCQuestion],
    conditions: Sequence[PromptCondition],
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
) -> dict[PromptCondition, BuiltPrompt]:
    """Build the requested condition prompts for one target question.

    Args:
        target: Target question.
        source_examples: Shots drawn from the source domain.
        same_domain_examples: Shots drawn from the target domain.
        irrelevant_pool: Pool for the length-matched control.
        conditions: Which conditions to build.
        tokenizer: Used for length matching of the irrelevant condition.
        seed: RNG seed.

    Returns:
        Mapping from condition to prompt.
    """
    # TODO(Phase 1): call build_prompt per condition; for IRRELEVANT first build
    # the cross-domain prompt and match its example-block token length.
    raise NotImplementedError("TODO(Phase 1): assemble all condition prompts")


def save_prompts(prompts: Sequence[BuiltPrompt], path: Path) -> None:
    """Write prompts to JSONL under ``data/prompts/``."""
    # TODO(Phase 1): one JSON object per line; include condition, text, spans,
    # target question and answer.
    raise NotImplementedError("TODO(Phase 1): JSONL serialization")


def load_prompts(path: Path) -> list[BuiltPrompt]:
    """Read prompts written by :func:`save_prompts`."""
    # TODO(Phase 1): parse and validate each line.
    raise NotImplementedError("TODO(Phase 1): JSONL deserialization")
