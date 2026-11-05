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

from collections.abc import Sequence
from dataclasses import dataclass


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


def load_mmlu_subject(
    subject: str, split: str, dataset_name: str = "cais/mmlu"
) -> list[MCQuestion]:
    """Load one MMLU subject/split as :class:`MCQuestion` objects.

    Args:
        subject: MMLU subject config name.
        split: One of ``"dev"``, ``"validation"``, ``"test"``.
        dataset_name: Hugging Face dataset id.

    Returns:
        Questions in dataset order.

    Raises:
        ValueError: If a row does not have exactly the expected fields or its
            answer index is out of range (validate external data at the boundary).
    """
    # TODO(Phase 1): datasets.load_dataset(dataset_name, subject, split=split),
    # validate each row, convert to MCQuestion. Do not run until the user
    # approves the dataset download.
    raise NotImplementedError("TODO(Phase 1): load and validate MMLU rows")


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
    # TODO(Phase 1): filter out `exclude`, sample with numpy Generator(seed),
    # return a new list (do not mutate `questions`).
    raise NotImplementedError("TODO(Phase 1): seeded sampling without replacement")
