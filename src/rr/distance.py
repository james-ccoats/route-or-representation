"""A cheap, model-free distance between MMLU subjects.

Role in the project: the pairs config labels each pair near / medium / far, but
those labels are a hypothesis (H5 asks whether the routing share of the few-shot gain
grows with domain distance). This module gives a first *measured* number to set
beside the label, so the ladder of sources in the pairs config is not just an
assumption.

Method: each subject is one "document" (all its question and answer-choice text).
Words are weighted by TF-IDF, with inverse document frequency computed across *all*
subjects, so words that appear in every subject carry no weight. The distance is
``1 - cosine similarity`` of the weighted vectors: 0 for identical vocabulary, 1 for
no distinguishing words in common.

Limits: this is lexical. It sees shared terminology, not shared skills, and it knows
nothing about what the model's router does. Treat it as a sanity check on the labels
and a way to choose a spread of sources, not as the explanatory variable. A
routing-based distance needs recorded routing from the model and is future work.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

from rr.config import VALID_DISTANCES

if TYPE_CHECKING:
    from rr.config import PairsConfig

WORD = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> list[str]:
    """Lowercased alphabetic words of two or more letters (digits and symbols dropped)."""
    return WORD.findall(text.lower())


@dataclass(frozen=True)
class SubjectDistances:
    """Pairwise distances in ``[0, 1]`` between subjects.

    Attributes:
        subjects: Subject names, in matrix order.
        matrix: ``[n, n]`` symmetric array with a zero diagonal (read-only).
    """

    subjects: tuple[str, ...]
    matrix: np.ndarray

    def between(self, subject_a: str, subject_b: str) -> float:
        """Distance between two subjects.

        Raises:
            KeyError: If either subject is not in the matrix.
        """
        return float(self.matrix[self._index(subject_a), self._index(subject_b)])

    def _index(self, subject: str) -> int:
        try:
            return self.subjects.index(subject)
        except ValueError:
            raise KeyError(f"unknown subject {subject!r}") from None


def subject_distances(corpora: Mapping[str, Sequence[str]]) -> SubjectDistances:
    """TF-IDF cosine distances between subjects.

    Args:
        corpora: ``{subject: texts}``. Include *all* subjects you want IDF to be
            computed over, not only the ones you will compare.

    Raises:
        ValueError: With fewer than two subjects, or if a subject has no words.
    """
    if len(corpora) < 2:
        raise ValueError("need at least two subjects to compute distances")
    names = tuple(corpora)
    counts = [Counter(word for text in corpora[name] for word in tokenize(text)) for name in names]
    for name, count in zip(names, counts, strict=True):
        if not count:
            raise ValueError(f"subject {name!r} has an empty corpus (no alphabetic words)")
    vocabulary = sorted(set().union(*counts))
    column = {word: i for i, word in enumerate(vocabulary)}
    document_frequency = Counter(word for count in counts for word in count)
    idf = np.zeros(len(vocabulary))
    for word, frequency in document_frequency.items():
        idf[column[word]] = math.log(len(names) / frequency)
    vectors = np.zeros((len(names), len(vocabulary)))
    for row, count in enumerate(counts):
        for word, term_frequency in count.items():
            vectors[row, column[word]] = (1.0 + math.log(term_frequency)) * idf[column[word]]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    matrix = np.clip(1.0 - unit @ unit.T, 0.0, 1.0)
    np.fill_diagonal(matrix, 0.0)
    matrix.setflags(write=False)
    return SubjectDistances(subjects=names, matrix=matrix)


def nearest_subjects(
    distances: SubjectDistances, subject: str, count: int
) -> list[tuple[str, float]]:
    """The ``count`` subjects closest to ``subject`` (itself excluded), nearest first.

    Ties are broken by name so the result is deterministic.
    """
    others = [(name, distances.between(subject, name)) for name in distances.subjects]
    ranked = sorted((item for item in others if item[0] != subject), key=lambda x: (x[1], x[0]))
    return ranked[:count]


@dataclass(frozen=True)
class DirectionDistance:
    """The expected label and the measured distance for one source -> target run."""

    run: str
    source: str
    target: str
    expected: str
    measured: float


def direction_distances(pairs: PairsConfig, distances: SubjectDistances) -> list[DirectionDistance]:
    """Measured distance for every run the pairs config will make, next to its label."""
    expected = {pair.name: pair.expected_distance for pair in pairs.pairs}
    return [
        DirectionDistance(
            run=run,
            source=source,
            target=target,
            expected=expected[run.rsplit(":", 1)[0]],
            measured=distances.between(source, target),
        )
        for run, source, target in pairs.directions()
    ]


def label_order_violations(rows: Sequence[DirectionDistance]) -> list[str]:
    """Pairs of runs whose labels say one is closer, but the measurement says it is farther.

    Equal labels are never compared. The labels are ordered as in
    :data:`rr.config.VALID_DISTANCES` (``same_field`` < ``near`` < ``medium`` < ``far``).
    """
    rank = {label: i for i, label in enumerate(VALID_DISTANCES)}
    return [
        f"{a.run} is labelled {a.expected} but measures farther ({a.measured:.3f}) "
        f"than {b.run}, labelled {b.expected} ({b.measured:.3f})"
        for a, b in permutations(rows, 2)
        if rank[a.expected] < rank[b.expected] and a.measured > b.measured
    ]


def format_direction_distances(rows: Sequence[DirectionDistance]) -> str:
    """Plain-text table of expected labels beside measured distances, plus the label check."""
    lines = [f"{'run':<26}{'source -> target':<62}{'expected':<12}{'measured':>9}"]
    for r in rows:
        lines.append(
            f"{r.run:<26}{r.source + ' -> ' + r.target:<62}{r.expected:<12}{r.measured:>9.3f}"
        )
    violations = label_order_violations(rows)
    lines.append("")
    if violations:
        lines.append("Expected labels disagree with the measured ordering:")
        lines.extend(f"  {v}" for v in violations)
    else:
        lines.append("Expected labels agree with the measured ordering.")
    return "\n".join(lines)
