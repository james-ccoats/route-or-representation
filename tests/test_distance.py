"""Tests for rr.distance: model-free lexical distance between MMLU subjects."""

from __future__ import annotations

import numpy as np
import pytest

from rr.distance import SubjectDistances, nearest_subjects, subject_distances, tokenize


def corpora(**subjects: list[str]) -> dict[str, list[str]]:
    return dict(subjects)


def test_tokenize_lowercases_and_keeps_alphabetic_words_of_two_or_more_letters() -> None:
    assert tokenize("The DNA of E. coli, 3 cells & a RNA-virus") == [
        "the", "dna", "of", "coli", "cells", "rna", "virus",
    ]  # fmt: skip


def test_identical_subjects_are_at_distance_zero() -> None:
    result = subject_distances(
        corpora(
            a=["cell membrane protein"], b=["cell membrane protein"], c=["algebra matrix group"]
        )
    )

    assert result.between("a", "b") == pytest.approx(0.0)


def test_subjects_with_disjoint_vocabulary_are_at_distance_one() -> None:
    result = subject_distances(
        corpora(a=["cell protein"], b=["algebra matrix"], c=["court statute"])
    )

    assert result.between("a", "b") == pytest.approx(1.0)


def test_shared_rare_words_make_subjects_closer_than_unrelated_ones() -> None:
    result = subject_distances(
        corpora(
            bio=["cell membrane protein enzyme"],
            med=["cell protein enzyme patient"],
            law=["court statute contract patient"],
            math=["algebra matrix theorem group"],
        )
    )

    assert result.between("bio", "med") < result.between("bio", "law")
    assert result.between("bio", "med") < result.between("bio", "math")


def test_a_word_in_every_subject_carries_no_weight() -> None:
    result = subject_distances(corpora(a=["common alpha"], b=["common beta"], c=["common gamma"]))

    assert result.between("a", "b") == pytest.approx(1.0)


def test_distances_are_symmetric_with_a_zero_diagonal() -> None:
    result = subject_distances(
        corpora(a=["cell protein"], b=["cell algebra"], c=["algebra court"], d=["court protein"])
    )

    matrix = result.matrix
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0 + 1e-9


def test_distance_does_not_depend_on_how_text_is_split_across_questions() -> None:
    split = subject_distances(corpora(a=["cell", "protein"], b=["cell protein"], c=["court"]))

    assert split.between("a", "b") == pytest.approx(0.0)


def test_between_rejects_unknown_subjects() -> None:
    result = subject_distances(corpora(a=["cell"], b=["court"]))

    with pytest.raises(KeyError, match="nope"):
        result.between("a", "nope")


def test_at_least_two_subjects_are_required() -> None:
    with pytest.raises(ValueError, match="two"):
        subject_distances(corpora(a=["cell"]))


def test_a_subject_without_any_words_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        subject_distances(corpora(a=["12 34"], b=["cell"]))


def test_nearest_subjects_are_ordered_by_distance_and_exclude_the_subject_itself() -> None:
    result = subject_distances(
        corpora(
            bio=["cell membrane protein enzyme"],
            med=["cell protein enzyme patient"],
            law=["court statute contract patient"],
            math=["algebra matrix theorem group"],
        )
    )

    closest = nearest_subjects(result, "bio", count=1)
    everyone = nearest_subjects(result, "bio", count=10)

    assert [name for name, _ in closest] == ["med"]
    assert [name for name, _ in everyone] == ["med", "law", "math"]  # ties broken by name
    assert [d for _, d in everyone] == sorted(d for _, d in everyone)


def test_result_is_a_value_object() -> None:
    result = subject_distances(corpora(a=["cell"], b=["court"]))

    assert isinstance(result, SubjectDistances)
    assert result.subjects == ("a", "b")


# --- pair-level reporting ---------------------------------------------------------------------

from rr.config import DomainPair, PairsConfig  # noqa: E402
from rr.distance import (  # noqa: E402
    DirectionDistance,
    direction_distances,
    format_direction_distances,
    label_order_violations,
)


def pairs_config(*pairs: DomainPair, both: bool = False) -> PairsConfig:
    return PairsConfig(
        dataset="cais/mmlu",
        shot_split="dev",
        eval_split="test",
        run_both_directions=both,
        pairs=tuple(pairs),
        irrelevant_subjects=(),
    )


FOUR = subject_distances(
    corpora(
        bio=["cell membrane protein enzyme"],
        med=["cell protein enzyme patient"],
        law=["court statute contract patient"],
        math=["algebra matrix theorem group"],
    )
)


def test_direction_distances_follow_the_configs_directions() -> None:
    config = pairs_config(
        DomainPair("bio_med", "bio", "med", "near"),
        DomainPair("bio_law", "bio", "law", "far"),
        both=True,
    )

    rows = direction_distances(config, FOUR)

    assert [(r.run, r.source, r.target, r.expected) for r in rows] == [
        ("bio_med:a2b", "bio", "med", "near"),
        ("bio_med:b2a", "med", "bio", "near"),
        ("bio_law:a2b", "bio", "law", "far"),
        ("bio_law:b2a", "law", "bio", "far"),
    ]
    assert rows[0].measured == pytest.approx(FOUR.between("bio", "med"))
    assert rows[0].measured == pytest.approx(rows[1].measured)  # the distance is symmetric


def test_direction_distances_run_one_way_when_the_config_does() -> None:
    config = pairs_config(DomainPair("bio_med", "bio", "med", "near"))

    assert [r.run for r in direction_distances(config, FOUR)] == ["bio_med:a2b"]


def row(run: str, expected: str, measured: float) -> DirectionDistance:
    return DirectionDistance(run=run, source="s", target="t", expected=expected, measured=measured)


def test_labels_that_agree_with_the_measurement_raise_no_violations() -> None:
    rows = [row("a", "same_field", 0.2), row("b", "near", 0.4), row("c", "far", 0.9)]

    assert label_order_violations(rows) == []


def test_a_closer_label_that_measures_farther_is_reported() -> None:
    rows = [row("close", "near", 0.9), row("distant", "far", 0.5)]

    (violation,) = label_order_violations(rows)

    assert "close" in violation
    assert "distant" in violation


def test_equal_labels_are_never_violations() -> None:
    assert label_order_violations([row("a", "near", 0.9), row("b", "near", 0.1)]) == []


def test_formatting_lists_every_direction_and_the_violations() -> None:
    rows = [row("close", "near", 0.9), row("distant", "far", 0.5)]

    text = format_direction_distances(rows)

    for expected in ("close", "distant", "near", "far", "0.900", "0.500", "disagree"):
        assert expected in text


def test_formatting_without_violations_says_the_labels_agree() -> None:
    text = format_direction_distances([row("a", "near", 0.1), row("b", "far", 0.9)])

    assert "agree" in text
    assert "disagree" not in text
