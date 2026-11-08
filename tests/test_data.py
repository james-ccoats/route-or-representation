"""Tests for rr.data: row validation, MMLU loading (mocked), seeded sampling."""

from __future__ import annotations

import pytest

from rr.data import MCQuestion, load_mmlu_subject, sample_questions


def _row(**overrides):
    row = {
        "question": "Q?",
        "subject": "college_biology",
        "choices": ["a", "b", "c", "d"],
        "answer": 2,
    }
    return {**row, **overrides}


def _patch_dataset(monkeypatch, rows):
    calls = []

    def fake_load_dataset(name, subject, split, revision=None):
        calls.append((name, subject, split))
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


def test_load_mmlu_subject_converts_rows_in_order(monkeypatch) -> None:
    calls = _patch_dataset(
        monkeypatch, [_row(question="one", answer=0), _row(question="two", answer=3)]
    )

    result = load_mmlu_subject("college_biology", "dev")

    assert calls == [("cais/mmlu", "college_biology", "dev")]
    assert result == [
        MCQuestion("college_biology", "one", ("a", "b", "c", "d"), 0),
        MCQuestion("college_biology", "two", ("a", "b", "c", "d"), 3),
    ]


@pytest.mark.parametrize(
    "bad_row",
    [
        _row(answer=4),
        _row(answer=-1),
        _row(answer=True),
        _row(answer="B"),
        _row(choices=["a", "b"]),
        _row(choices=["a", "b", "c", 4]),
        {"question": "Q?", "choices": ["a", "b", "c", "d"]},
        _row(question=""),
    ],
)
def test_load_mmlu_subject_rejects_malformed_rows(monkeypatch, bad_row) -> None:
    _patch_dataset(monkeypatch, [bad_row])

    with pytest.raises(ValueError, match="college_biology"):
        load_mmlu_subject("college_biology", "dev")


def test_load_mmlu_subject_rejects_empty_split(monkeypatch) -> None:
    _patch_dataset(monkeypatch, [])

    with pytest.raises(ValueError, match="no rows"):
        load_mmlu_subject("college_biology", "dev")


def test_sample_questions_is_deterministic_for_a_seed(make_questions) -> None:
    pool = make_questions("bio", 30)

    assert sample_questions(pool, 10, seed=1) == sample_questions(pool, 10, seed=1)


def test_sample_questions_differs_across_seeds(make_questions) -> None:
    pool = make_questions("bio", 30)

    assert sample_questions(pool, 10, seed=1) != sample_questions(pool, 10, seed=2)


def test_sample_questions_has_no_duplicates_and_stays_in_pool(make_questions) -> None:
    pool = make_questions("bio", 30)

    sample = sample_questions(pool, 30, seed=0)

    assert len(sample) == 30
    assert set(sample) == set(pool)


def test_sample_questions_never_returns_excluded(make_questions) -> None:
    pool = make_questions("bio", 20)
    excluded = pool[:15]

    sample = sample_questions(pool, 5, seed=0, exclude=excluded)

    assert set(sample) == set(pool[15:])


def test_sample_questions_does_not_mutate_inputs(make_questions) -> None:
    pool = make_questions("bio", 10)
    snapshot = list(pool)

    sample_questions(pool, 5, seed=0)

    assert pool == snapshot


def test_sample_questions_raises_when_too_few_remain(make_questions) -> None:
    pool = make_questions("bio", 10)

    with pytest.raises(ValueError, match="only 4"):
        sample_questions(pool, 5, seed=0, exclude=pool[:6])


def test_sample_questions_rejects_negative_n(make_questions) -> None:
    with pytest.raises(ValueError):
        sample_questions(make_questions("bio", 3), -1, seed=0)


def test_mcquestion_rejects_out_of_range_answer_index() -> None:
    with pytest.raises(ValueError, match="answer_index"):
        MCQuestion("toy", "Q?", ("a", "b"), 2)


def test_load_mmlu_subject_passes_dataset_revision(monkeypatch) -> None:
    seen = {}

    def fake_load_dataset(name, subject, split, revision=None):
        seen["revision"] = revision
        return [_row()]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    load_mmlu_subject("college_biology", "dev", revision="abc123")

    assert seen["revision"] == "abc123"
