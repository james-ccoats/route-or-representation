"""Tests for rr.results: record schema validation and JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rr.hidden import HiddenShift
from rr.prompts import PromptCondition
from rr.results import (
    ResultRecord,
    append_records,
    read_records,
    read_results_dir,
    repair_torn_tail,
)


def make_record(
    index: int = 0, condition=PromptCondition.CROSS_DOMAIN, **overrides
) -> ResultRecord:
    values = {
        "run": "bio_med_a2b",
        "question_index": index,
        "subject": "college_medicine",
        "condition": condition,
        "option_logprobs": (-1.0, -2.0, -3.0, -4.0),
        "prediction": 0,
        "is_correct": True,
        "correct_logprob": -1.0,
        "num_examples": 5,
        "example_tokens": 300,
        "jaccard_per_layer": (0.5, 0.75),
    }
    return ResultRecord(**{**values, **overrides})


# --- schema ---------------------------------------------------------------------------


def test_record_round_trips_through_a_dict() -> None:
    record = make_record()

    assert ResultRecord.from_dict(record.to_dict()) == record


def test_zero_shot_record_may_have_no_jaccard() -> None:
    record = make_record(
        condition=PromptCondition.ZERO_SHOT,
        num_examples=0,
        example_tokens=0,
        jaccard_per_layer=None,
    )

    assert ResultRecord.from_dict(record.to_dict()).jaccard_per_layer is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run", ""),
        ("question_index", -1),
        ("question_index", True),
        ("prediction", 4),
        ("is_correct", "yes"),
        ("correct_logprob", "high"),
        ("option_logprobs", ()),
        ("jaccard_per_layer", (0.5, 1.5)),
        ("jaccard_per_layer", ()),
    ],
)
def test_invalid_field_values_are_rejected(field, value) -> None:
    data = make_record().to_dict()
    data[field] = list(value) if isinstance(value, tuple) else value

    with pytest.raises(ValueError, match=field):
        ResultRecord.from_dict(data)


def test_missing_field_is_rejected() -> None:
    data = make_record().to_dict()
    del data["condition"]

    with pytest.raises(ValueError, match="condition"):
        ResultRecord.from_dict(data)


def test_unknown_condition_is_rejected() -> None:
    data = make_record().to_dict()
    data["condition"] = "mystery"

    with pytest.raises(ValueError, match="condition"):
        ResultRecord.from_dict(data)


def test_inconsistent_correctness_is_rejected() -> None:
    with pytest.raises(ValueError, match="is_correct"):
        make_record(prediction=1, is_correct=True)


def test_correct_logprob_must_be_one_of_the_option_logprobs() -> None:
    with pytest.raises(ValueError, match="correct_logprob"):
        make_record(correct_logprob=-9.0)


# --- persistence ----------------------------------------------------------------------


def test_append_then_read_returns_the_records_in_order(tmp_path) -> None:
    path = tmp_path / "out" / "run.jsonl"
    first = [make_record(0), make_record(1)]
    second = [make_record(2)]

    append_records(path, first)
    append_records(path, second)

    assert read_records(path) == first + second


def test_read_reports_the_line_of_a_malformed_record(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    append_records(path, [make_record(0)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run": "x"}) + "\n")

    with pytest.raises(ValueError, match=r"run\.jsonl line 2"):
        read_records(path)


def test_repair_torn_tail_drops_a_partial_last_line(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    append_records(path, [make_record(0), make_record(1)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run": "bio_med_a2b", "quest')  # crash mid-write

    assert repair_torn_tail(path) is True

    assert [r.question_index for r in read_records(path)] == [0, 1]


def test_repair_torn_tail_leaves_a_clean_file_alone(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    append_records(path, [make_record(0)])
    before = path.read_bytes()

    assert repair_torn_tail(path) is False
    assert path.read_bytes() == before


def test_repair_torn_tail_ignores_a_missing_file(tmp_path) -> None:
    assert repair_torn_tail(tmp_path / "absent.jsonl") is False


def test_repair_torn_tail_handles_a_file_with_no_complete_line(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"run": "x', encoding="utf-8")

    assert repair_torn_tail(path) is True
    assert path.read_text(encoding="utf-8") == ""


def test_read_results_dir_collects_every_jsonl_file_sorted(tmp_path) -> None:
    append_records(tmp_path / "b_run.jsonl", [replace(make_record(0), run="b_run")])
    append_records(tmp_path / "a_run.jsonl", [replace(make_record(0), run="a_run")])
    (tmp_path / "notes.txt").write_text("ignore me")

    assert [r.run for r in read_results_dir(tmp_path)] == ["a_run", "b_run"]


def test_read_results_dir_fails_loudly_when_empty_or_missing(tmp_path) -> None:
    with pytest.raises(ValueError, match="no result files"):
        read_results_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_results_dir(tmp_path / "missing")


# --- intervention field -----------------------------------------------------------------


def test_intervention_defaults_to_none_and_round_trips() -> None:
    assert make_record().intervention == "none"
    forced = make_record(intervention="force_zero_shot_routing")

    assert ResultRecord.from_dict(forced.to_dict()) == forced


def test_records_written_before_the_intervention_field_still_load() -> None:
    data = make_record().to_dict()
    del data["intervention"]

    assert ResultRecord.from_dict(data).intervention == "none"


def test_empty_intervention_is_rejected() -> None:
    with pytest.raises(ValueError, match="intervention"):
        make_record(intervention="")


# --- hidden-state shift on the record ------------------------------------------------------------


def hidden() -> HiddenShift:
    return HiddenShift(
        relative_l2=(0.5, 0.7),
        cosine_distance=(0.1, 0.2),
        router_visible_share=(0.05, 0.1),
        router_visible_chance=(0.03, 0.03),
    )


def test_hidden_shift_defaults_to_none_and_round_trips() -> None:
    assert make_record().hidden_shift is None
    record = make_record(hidden_shift=hidden())

    assert ResultRecord.from_dict(record.to_dict()) == record
    assert json.loads(json.dumps(record.to_dict()))["hidden_shift"]["relative_l2"] == [0.5, 0.7]


def test_records_without_a_hidden_shift_field_still_load() -> None:
    data = make_record().to_dict()
    del data["hidden_shift"]

    assert ResultRecord.from_dict(data).hidden_shift is None


def test_a_malformed_hidden_shift_in_a_record_is_rejected() -> None:
    data = make_record(hidden_shift=hidden()).to_dict()
    data["hidden_shift"]["router_visible_share"] = [0.1, 7.0]

    with pytest.raises(ValueError, match="router_visible_share"):
        ResultRecord.from_dict(data)


def test_hidden_shift_must_be_a_hiddenshift() -> None:
    with pytest.raises(ValueError, match="hidden_shift"):
        make_record(hidden_shift={"relative_l2": [0.1]})


def test_hidden_shift_and_jaccard_must_cover_the_same_number_of_layers_when_both_present() -> None:
    with pytest.raises(ValueError, match="hidden_shift"):
        make_record(hidden_shift=hidden(), jaccard_per_layer=(0.5, 0.5, 0.5))
