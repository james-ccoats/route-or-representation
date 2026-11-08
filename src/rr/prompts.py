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
aligned (see :mod:`rr.routing.align`). Character identity is guaranteed by
construction; token identity depends on the tokenizer and is checked by
:func:`verify_target_tokens`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rr.data import MCQuestion
from rr.scoring import format_mc_prompt

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


BLOCK_SEPARATOR = "\n\n"
# How far (in tokens, either direction) the control's example block may be from the
# cross-domain block. Open question: see D5 in design_decisions.md.
LENGTH_MATCH_TOLERANCE_TOKENS = 16
# Random k-subsets used as starting points, and the cap on swap-improvement sweeps.
LENGTH_MATCH_RESTARTS = 20
MAX_SWAP_SWEEPS = 10


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
        example_tokens: Token count of the rendered example block (0 for zero-shot),
            recorded so the irrelevant-control length match can be audited later.
            Filled in by :func:`build_condition_prompts`.
    """

    condition: PromptCondition
    text: str
    target: MCQuestion
    target_char_span: tuple[int, int]
    num_examples: int
    example_tokens: int = 0


def _render_example_block(examples: Sequence[MCQuestion]) -> str:
    return BLOCK_SEPARATOR.join(format_mc_prompt(e, include_answer=True) for e in examples)


def _render_example_prefix(examples: Sequence[MCQuestion]) -> str:
    """Examples plus the separator that precedes the target block ("" if no examples)."""
    if not examples:
        return ""
    return _render_example_block(examples) + BLOCK_SEPARATOR


def _count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def count_example_tokens(examples: Sequence[MCQuestion], tokenizer: PreTrainedTokenizerBase) -> int:
    """Token length of the rendered example block (0 for no examples)."""
    return _count_tokens(_render_example_block(examples), tokenizer) if examples else 0


def build_prompt(
    condition: PromptCondition, target: MCQuestion, examples: Sequence[MCQuestion]
) -> BuiltPrompt:
    """Render one prompt.

    Args:
        condition: Condition label stored on the result.
        target: The question to be answered.
        examples: In-context examples, in presentation order (empty for zero-shot).

    Returns:
        A :class:`BuiltPrompt` with ``target_char_span`` filled in. ``example_tokens``
        is left at 0; :func:`build_condition_prompts` fills it in.

    Methodological note:
        Uses :func:`rr.scoring.format_mc_prompt` for every question so the
        target block is formatted identically in all conditions.
    """
    prefix = _render_example_prefix(examples)
    target_block = format_mc_prompt(target)
    return BuiltPrompt(
        condition=condition,
        text=prefix + target_block,
        target=target,
        target_char_span=(len(prefix), len(prefix) + len(target_block)),
        num_examples=len(examples),
    )


def _example_lengths(
    pool: Sequence[MCQuestion], tokenizer: PreTrainedTokenizerBase
) -> tuple[list[int], int]:
    """Per-question token lengths and the tokens a block separator adds.

    Lets the search score a candidate set as ``sum(lengths) + (k - 1) * separator``
    without re-tokenizing; the final choice is always re-measured exactly.
    """
    lengths = [count_example_tokens([q], tokenizer) for q in pool]
    if len(pool) < 2:
        return lengths, 0
    separator = count_example_tokens(pool[:2], tokenizer) - lengths[0] - lengths[1]
    return lengths, separator


def _improve_by_swaps(
    chosen: list[int], lengths: Sequence[int], separator: int, token_budget: int
) -> list[int]:
    """Replace one member at a time with the pool item that best closes the gap."""
    chosen = list(chosen)
    fixed = separator * (len(chosen) - 1)
    for _ in range(MAX_SWAP_SWEEPS):
        changed = False
        for position in range(len(chosen)):
            others = sum(lengths[i] for j, i in enumerate(chosen) if j != position)
            taken = set(chosen)
            best = min(
                (i for i in range(len(lengths)) if i not in taken or i == chosen[position]),
                key=lambda i: abs(others + lengths[i] + fixed - token_budget),
            )
            if best != chosen[position]:
                chosen[position] = best
                changed = True
        if not changed:
            break
    return chosen


def select_irrelevant_examples(
    pool: Sequence[MCQuestion],
    num_examples: int,
    token_budget: int,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
) -> list[MCQuestion]:
    """Pick exactly ``num_examples`` control examples matching ``token_budget``.

    Matches both the number of examples and the rendered token length of the
    cross-domain block. The selection depends only on the pool, k, budget and seed,
    so compute it once per direction and share it across all target questions.

    Args:
        pool: Candidate irrelevant questions (large and varied lengths work best).
        num_examples: Number of examples to return (the cross-domain shot count).
        token_budget: Token length of the cross-domain example block to match.
        tokenizer: Tokenizer used to measure length.
        seed: RNG seed.

    Returns:
        ``num_examples`` distinct questions whose rendered block is within
        :data:`LENGTH_MATCH_TOLERANCE_TOKENS` of ``token_budget``: the closest of
        :data:`LENGTH_MATCH_RESTARTS` random starts, each improved by swaps.

    Raises:
        ValueError: If the pool is smaller than ``num_examples`` or no selection
            reaches the budget within tolerance.
    """
    if num_examples < 0 or token_budget < 0:
        raise ValueError("num_examples and token_budget must be non-negative")
    if num_examples == 0:
        if token_budget != 0:
            raise ValueError(f"cannot match a {token_budget}-token budget with zero examples")
        return []
    if len(pool) < num_examples:
        raise ValueError(f"pool has {len(pool)} questions, fewer than the {num_examples} needed")
    lengths, separator = _example_lengths(pool, tokenizer)
    rng = np.random.default_rng(seed)
    best: list[MCQuestion] = []
    best_gap: int | None = None
    for _ in range(LENGTH_MATCH_RESTARTS):
        start = [int(i) for i in rng.choice(len(pool), size=num_examples, replace=False)]
        indices = _improve_by_swaps(start, lengths, separator, token_budget)
        candidate = [pool[i] for i in indices]
        gap = abs(count_example_tokens(candidate, tokenizer) - token_budget)
        if best_gap is None or gap < best_gap:
            best, best_gap = candidate, gap
        if best_gap == 0:
            break
    if best_gap is None or best_gap > LENGTH_MATCH_TOLERANCE_TOKENS:
        raise ValueError(
            f"best of {len(pool)}-question pool is {best_gap} tokens from the "
            f"{token_budget}-token budget with {num_examples} examples "
            f"(tolerance {LENGTH_MATCH_TOLERANCE_TOKENS}); use a larger or more varied pool"
        )
    return best


def _examples_for(
    condition: PromptCondition,
    source_examples: Sequence[MCQuestion],
    same_domain_examples: Sequence[MCQuestion],
    irrelevant_examples: Sequence[MCQuestion],
) -> Sequence[MCQuestion]:
    if condition is PromptCondition.ZERO_SHOT:
        return []
    if condition is PromptCondition.CROSS_DOMAIN:
        return source_examples
    if condition is PromptCondition.SAME_DOMAIN:
        return same_domain_examples
    if condition is PromptCondition.IRRELEVANT:
        return irrelevant_examples
    raise ValueError(f"unhandled prompt condition: {condition!r}")


def build_condition_prompts(
    target: MCQuestion,
    *,
    source_examples: Sequence[MCQuestion],
    same_domain_examples: Sequence[MCQuestion],
    irrelevant_examples: Sequence[MCQuestion],
    conditions: Sequence[PromptCondition],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[PromptCondition, BuiltPrompt]:
    """Build the requested condition prompts for one target question.

    Args:
        target: Target question.
        source_examples: Shots drawn from the source domain.
        same_domain_examples: Shots drawn from the target domain.
        irrelevant_examples: Control examples matching the source shots in both count
            and token length, selected once per direction with
            :func:`select_irrelevant_examples`.
        conditions: Which conditions to build.
        tokenizer: Used to record each prompt's example token count.

    Returns:
        Mapping from condition to prompt.

    Raises:
        ValueError: If the target appears in any example set (leakage), or the
            irrelevant example count differs from the source example count.
    """
    if PromptCondition.IRRELEVANT in conditions and len(irrelevant_examples) != len(
        source_examples
    ):
        raise ValueError(
            f"irrelevant example count ({len(irrelevant_examples)}) must equal "
            f"the source example count ({len(source_examples)})"
        )
    for name, examples in (
        ("source", source_examples),
        ("same-domain", same_domain_examples),
        ("irrelevant", irrelevant_examples),
    ):
        if target in examples:
            raise ValueError(f"target question appears among the {name} examples (leakage)")
    prompts: dict[PromptCondition, BuiltPrompt] = {}
    for condition in conditions:
        examples = _examples_for(
            condition, source_examples, same_domain_examples, irrelevant_examples
        )
        built = build_prompt(condition, target, examples)
        prompts[condition] = replace(
            built, example_tokens=count_example_tokens(examples, tokenizer)
        )
    return prompts


def verify_target_tokens(
    prompts: Sequence[BuiltPrompt], tokenizer: PreTrainedTokenizerBase
) -> None:
    """Check that the target block tokenizes identically in every prompt.

    Character identity does not guarantee token identity: a tokenizer may encode
    the start of a text differently from text after a separator, or merge across
    the example/target boundary. Per-token routing comparisons are only valid when
    the target tokens match, so this must hold for each target's prompts.

    Args:
        prompts: Prompts for the same target question (any conditions).
        tokenizer: A fast tokenizer supporting ``return_offsets_mapping``.

    Raises:
        ValueError: If targets differ, a token straddles the target start, or the
            target token ids differ between prompts.
    """
    if not prompts:
        return
    reference_ids: list[int] | None = None
    reference_condition = prompts[0].condition
    for prompt in prompts:
        if prompt.target != prompts[0].target:
            raise ValueError("prompts do not share the same target question")
        encoded = tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
        start, end = prompt.target_char_span
        ids = []
        for token_id, (token_start, token_end) in zip(
            encoded["input_ids"], encoded["offset_mapping"], strict=True
        ):
            if token_start < start < token_end:
                raise ValueError(
                    f"a token straddles the target start in the {prompt.condition} prompt"
                )
            if token_start >= start and token_end <= end:
                ids.append(token_id)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(
                f"target token ids differ between the {reference_condition} and "
                f"{prompt.condition} prompts"
            )


def save_prompts(prompts: Sequence[BuiltPrompt], path: Path) -> None:
    """Write prompts to JSONL under ``data/prompts/`` (atomically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_prompt_to_dict(p), ensure_ascii=False) + "\n" for p in prompts]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(path)


def load_prompts(path: Path) -> list[BuiltPrompt]:
    """Read prompts written by :func:`save_prompts`.

    Raises:
        ValueError: If any line is malformed or internally inconsistent.
    """
    prompts: list[BuiltPrompt] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            prompts.append(_prompt_from_dict(json.loads(line)))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} line {line_number}: {error!r}") from error
    return prompts


def _prompt_to_dict(prompt: BuiltPrompt) -> dict[str, Any]:
    target = prompt.target
    return {
        "condition": prompt.condition.value,
        "text": prompt.text,
        "target": {
            "subject": target.subject,
            "question": target.question,
            "choices": list(target.choices),
            "answer_index": target.answer_index,
        },
        "target_char_span": list(prompt.target_char_span),
        "num_examples": prompt.num_examples,
        "example_tokens": prompt.example_tokens,
    }


def _require_count(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key!r} must be a non-negative integer, got {value!r}")
    return value


def _prompt_from_dict(data: dict[str, Any]) -> BuiltPrompt:
    text = data["text"]
    start, end = data["target_char_span"]
    if not 0 <= start < end == len(text):
        raise ValueError(f"target_char_span {(start, end)} must end at text length {len(text)}")
    raw_target = data["target"]
    target = MCQuestion(  # validates answer_index
        subject=raw_target["subject"],
        question=raw_target["question"],
        choices=tuple(raw_target["choices"]),
        answer_index=raw_target["answer_index"],
    )
    if text[start:end] != format_mc_prompt(target):
        raise ValueError("target_char_span does not cover the rendered target block")
    condition = PromptCondition(data["condition"])
    num_examples = _require_count(data, "num_examples")
    example_tokens = _require_count(data, "example_tokens")
    has_examples = condition is not PromptCondition.ZERO_SHOT
    if has_examples != (num_examples > 0) or has_examples != (example_tokens > 0):
        raise ValueError(
            f"{condition} prompt has num_examples={num_examples}, example_tokens={example_tokens}"
        )
    return BuiltPrompt(
        condition=condition,
        text=text,
        target=target,
        target_char_span=(start, end),
        num_examples=num_examples,
        example_tokens=example_tokens,
    )
