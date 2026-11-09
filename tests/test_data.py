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


# --- load_mmlu_texts -----------------------------------------------------------------------


def _text_row(subject, question="Q", choices=("a", "b", "c", "d")):
    return {"subject": subject, "question": question, "choices": list(choices), "answer": 0}


def test_load_mmlu_texts_groups_question_and_choices_by_subject(monkeypatch) -> None:
    from rr.data import load_mmlu_texts

    calls = _patch_dataset(
        monkeypatch,
        [
            _text_row("bio", "What is a cell?", ("unit", "atom")),
            _text_row("law", "What is a tort?"),
            _text_row("bio", "What is DNA?", ("gene", "rock")),
        ],
    )

    texts = load_mmlu_texts("test")

    assert calls == [("cais/mmlu", "all", "test")]
    assert texts["bio"] == ["What is a cell? unit atom", "What is DNA? gene rock"]
    assert set(texts) == {"bio", "law"}


@pytest.mark.parametrize(
    "bad_row",
    [
        {"subject": "bio", "question": "Q"},
        {"subject": 3, "question": "Q", "choices": ["a"]},
        {"subject": "bio", "question": "", "choices": ["a"]},
        {"subject": "bio", "question": "Q", "choices": "abc"},
    ],
)
def test_load_mmlu_texts_rejects_malformed_rows(monkeypatch, bad_row) -> None:
    from rr.data import load_mmlu_texts

    _patch_dataset(monkeypatch, [bad_row])

    with pytest.raises(ValueError, match="row 0"):
        load_mmlu_texts("test")


def test_load_mmlu_texts_rejects_an_empty_split(monkeypatch) -> None:
    from rr.data import load_mmlu_texts

    _patch_dataset(monkeypatch, [])

    with pytest.raises(ValueError, match="no rows"):
        load_mmlu_texts("test")


# --- shot pools and replicate shot sets ----------------------------------------------------


def _pool(count: int):
    return [MCQuestion("s", f"q{i}?", ("a", "b"), 0) for i in range(count)]


def test_draw_shot_sets_returns_the_requested_number_of_ordered_sets() -> None:
    from rr.data import draw_shot_sets

    sets = draw_shot_sets(_pool(20), shots_per_set=5, num_sets=3, seed=0)

    assert len(sets) == 3
    assert all(len(s) == 5 and len(set(s)) == 5 for s in sets)


def test_draw_shot_sets_are_disjoint_while_the_pool_lasts() -> None:
    from rr.data import draw_shot_sets

    sets = draw_shot_sets(_pool(15), shots_per_set=5, num_sets=3, seed=0)

    flat = [q for s in sets for q in s]
    assert len(set(flat)) == 15


def test_draw_shot_sets_wrap_around_a_small_pool_without_repeating_within_a_set() -> None:
    from rr.data import draw_shot_sets

    sets = draw_shot_sets(_pool(13), shots_per_set=5, num_sets=3, seed=0)

    assert all(len(set(s)) == 5 for s in sets)
    assert len(set(sets[0]) & set(sets[1])) == 0
    assert len(set(sets[2]) & set(sets[0])) == 2  # only the wrapped-around items repeat


def test_draw_shot_sets_is_deterministic_and_seed_dependent() -> None:
    from rr.data import draw_shot_sets

    first = draw_shot_sets(_pool(20), 5, 2, seed=1)

    assert draw_shot_sets(_pool(20), 5, 2, seed=1) == first
    assert draw_shot_sets(_pool(20), 5, 2, seed=2) != first


def test_draw_shot_sets_do_not_depend_on_pool_order_of_other_seeds() -> None:
    from rr.data import draw_shot_sets

    sets = draw_shot_sets(_pool(20), 5, 1, seed=0)

    assert set(sets[0]) <= set(_pool(20))


def test_draw_shot_sets_reject_a_pool_smaller_than_one_set() -> None:
    from rr.data import draw_shot_sets

    with pytest.raises(ValueError, match="pool"):
        draw_shot_sets(_pool(4), shots_per_set=5, num_sets=1, seed=0)


@pytest.mark.parametrize("shots,sets", [(0, 1), (5, 0), (-1, 1)])
def test_draw_shot_sets_reject_non_positive_sizes(shots, sets) -> None:
    from rr.data import draw_shot_sets

    with pytest.raises(ValueError):
        draw_shot_sets(_pool(20), shots, sets, seed=0)


def test_load_shot_pool_concatenates_splits_in_order_and_drops_duplicates(monkeypatch) -> None:
    from rr.data import load_shot_pool

    by_split = {
        "dev": [_row(question="one"), _row(question="two")],
        "validation": [_row(question="two"), _row(question="three")],
    }

    def fake_load_dataset(name, subject, split, revision=None):
        return by_split[split]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    pool = load_shot_pool("college_biology", ("dev", "validation"))

    assert [q.question for q in pool] == ["one", "two", "three"]
