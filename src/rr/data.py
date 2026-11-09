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
