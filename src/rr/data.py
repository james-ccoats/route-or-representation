"""Dataset loading and question sampling.

Role in the project: provides MMLU multiple-choice questions for each domain
(subject) and the sampling utilities that decide which questions are few-shot
examples and which are evaluation targets. Used from Phase 1 onward.

Methodological notes:

* Shot questions and evaluation questions for a subject must never overlap.
* MMLU's ``dev`` split has only 5 questions per subject. Using more shots than
  that requires drawing shots from another split (see ``shot_split`` and
  ``eval_split`` in ``configs/pairs/*.yaml``) and excluding them from eval.
* All sampling is seeded so prompt sets are reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MCQuestion:
    """A single multiple-choice question.

    Attributes:
        subject: MMLU subject name, e.g. ``"college_biology"``.
        question: Question stem.
        choices: Answer option texts, in presentation order.
        answer_index: Index into ``choices`` of the correct option.
    """

    subject: str
    question: str
    choices: tuple[str, ...]
    answer_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError(
                f"answer_index {self.answer_index} out of range for {len(self.choices)} choices"
            )


def _row_to_question(row: Mapping[str, Any], subject: str) -> MCQuestion:
    """Validate one raw dataset row and convert it (external data is never trusted)."""
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing or empty 'question'")
    if not isinstance(choices, (list, tuple)) or len(choices) < 2:
        raise ValueError("'choices' must be a list of at least two options")
    if not all(isinstance(c, str) for c in choices):
        raise ValueError("every entry of 'choices' must be a string")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
        raise ValueError(f"'answer' must be an index into {len(choices)} choices, got {answer!r}")
    return MCQuestion(
        subject=subject, question=question, choices=tuple(choices), answer_index=answer
    )


def load_mmlu_subject(
    subject: str, split: str, dataset_name: str = "cais/mmlu", revision: str | None = None
) -> list[MCQuestion]:
    """Load one MMLU subject/split as :class:`MCQuestion` objects.

    Args:
        subject: MMLU subject config name.
        split: One of ``"dev"``, ``"validation"``, ``"test"``.
        dataset_name: Hugging Face dataset id.
        revision: Dataset revision (commit hash) to pin; ``None`` uses the latest.
            Sampling indexes into dataset order, so pin this for reproducibility.

    Returns:
        Questions in dataset order.

    Raises:
        ValueError: If a row does not have exactly the expected fields or its
            answer index is out of range (validate external data at the boundary).
    """
    from datasets import load_dataset  # deferred: importing datasets is slow

    rows = load_dataset(dataset_name, subject, split=split, revision=revision)
    questions: list[MCQuestion] = []
    for position, row in enumerate(rows):
        try:
            questions.append(_row_to_question(row, subject))
        except ValueError as error:
            raise ValueError(f"{subject}/{split} row {position}: {error}") from error
    if not questions:
        raise ValueError(f"{subject}/{split} has no rows")
    return questions


def _row_to_text(row: Mapping[str, Any]) -> tuple[str, str]:
    """Validate one raw row of the "all" config and return ``(subject, question + choices)``."""
    subject = row.get("subject")
    question = row.get("question")
    choices = row.get("choices")
    if not isinstance(subject, str) or not subject:
        raise ValueError("missing or empty 'subject'")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing or empty 'question'")
    if not isinstance(choices, (list, tuple)) or not all(isinstance(c, str) for c in choices):
        raise ValueError("'choices' must be a list of strings")
    return subject, " ".join([question, *choices])


def load_mmlu_texts(
    split: str, dataset_name: str = "cais/mmlu", revision: str | None = None
) -> dict[str, list[str]]:
    """Load every subject's question text for one split, grouped by subject.

    Used for the model-free subject distance (:mod:`rr.distance`), whose IDF needs
    all subjects at once. Each text is the question followed by its answer choices.

    Args:
        split: One of ``"dev"``, ``"validation"``, ``"test"``.
        dataset_name: Hugging Face dataset id.
        revision: Dataset revision (commit hash) to pin; ``None`` uses the latest.

    Returns:
        ``{subject: [text, ...]}`` in dataset order.

    Raises:
        ValueError: If a row is malformed or the split is empty.
    """
    from datasets import load_dataset  # deferred: importing datasets is slow

    rows = load_dataset(dataset_name, "all", split=split, revision=revision)
    texts: dict[str, list[str]] = {}
    for position, row in enumerate(rows):
        try:
            subject, text = _row_to_text(row)
        except ValueError as error:
            raise ValueError(f"{split} row {position}: {error}") from error
        texts.setdefault(subject, []).append(text)
    if not texts:
        raise ValueError(f"{split} has no rows")
    return texts


def sample_questions(
    questions: Sequence[MCQuestion],
    n: int,
    seed: int,
    exclude: Sequence[MCQuestion] = (),
) -> list[MCQuestion]:
    """Sample ``n`` questions without replacement, deterministically.

    Args:
        questions: Pool to sample from.
        n: Number of questions to draw.
        seed: RNG seed.
        exclude: Questions that must not be returned (e.g. the shot set when
            sampling eval questions).

    Returns:
        A new list of ``n`` questions.

    Raises:
        ValueError: If fewer than ``n`` questions remain after exclusion.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    excluded = set(exclude)
    candidates = [q for q in questions if q not in excluded]
    if len(candidates) < n:
        raise ValueError(
            f"requested {n} questions but only {len(candidates)} remain after exclusion"
        )
    chosen = np.random.default_rng(seed).choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in chosen]


def load_shot_pool(
    subject: str,
    splits: Sequence[str],
    dataset_name: str = "cais/mmlu",
    revision: str | None = None,
) -> list[MCQuestion]:
    """Load the questions few-shot examples may be drawn from: several splits pooled.

    MMLU's ``dev`` split has only 5 questions per subject, so pooling it with
    ``validation`` gives enough distinct questions to draw several example sets.

    Args:
        subject: MMLU subject config name.
        splits: Splits to pool, in order (never the evaluation split).
        dataset_name: Hugging Face dataset id.
        revision: Dataset revision (commit hash) to pin.

    Returns:
        The pooled questions in split order, with exact duplicates dropped.
    """
    pool: list[MCQuestion] = []
    seen: set[MCQuestion] = set()
    for split in splits:
        for question in load_mmlu_subject(subject, split, dataset_name, revision):
            if question not in seen:
                seen.add(question)
                pool.append(question)
    return pool


def draw_shot_sets(
    pool: Sequence[MCQuestion], shots_per_set: int, num_sets: int, seed: int
) -> list[list[MCQuestion]]:
    """Draw ``num_sets`` ordered example sets of ``shots_per_set`` questions from one pool.

    The pool is shuffled once with the seed and set ``r`` takes the next
    ``shots_per_set`` questions, wrapping around the end. Sets are therefore disjoint
    while the pool lasts; once it runs out, a set reuses only the questions it wraps
    onto, and never repeats a question within itself. The order inside a set is the
    shuffled order, so the first ``k`` questions of a set form a nested k-shot prompt.

    Raises:
        ValueError: On non-positive sizes, or a pool smaller than one set.
    """
    if shots_per_set < 1 or num_sets < 1:
        raise ValueError(
            f"shots_per_set and num_sets must be positive, got {shots_per_set} and {num_sets}"
        )
    unique = list(dict.fromkeys(pool))
    if len(unique) < shots_per_set:
        raise ValueError(
            f"pool has {len(unique)} questions, fewer than the {shots_per_set} needed per set"
        )
    order = np.random.default_rng(seed).permutation(len(unique))
    size = len(unique)
    return [
        [unique[int(order[(r * shots_per_set + j) % size])] for j in range(shots_per_set)]
        for r in range(num_sets)
    ]
