"""Tests for rr.pilot: per-question evaluation and the resumable per-file runner."""

from __future__ import annotations

import shutil

import pytest
import torch

from conftest import (
    TINY_ROUTER_NAMES,
    TINY_TOP_K,
    TinyVocabTokenizer,
    build_group,
    build_tiny_moe,
    make_question,
    write_prompt_file,
)
from rr.manifest import manifest_path_for
from rr.models import inspect_moe_structure
from rr.pilot import evaluate_group, run_prompt_file
from rr.prompts import PromptCondition
from rr.results import append_records, read_records
from rr.routing.align import align_target_tokens
from rr.routing.metrics import per_layer_jaccard
from rr.routing.record import RoutingRecorder
from rr.runner import group_prompts_by_target
from rr.scoring import score_mc

CONDITIONS = (
    PromptCondition.ZERO_SHOT,
    PromptCondition.CROSS_DOMAIN,
    PromptCondition.SAME_DOMAIN,
    PromptCondition.IRRELEVANT,
)


@pytest.fixture(scope="module")
def model():
    return build_tiny_moe()


@pytest.fixture(scope="module")
def structure(model):
    return inspect_moe_structure(model)


@pytest.fixture
def tokenizer() -> TinyVocabTokenizer:
    return TinyVocabTokenizer()


# --- grouping -------------------------------------------------------------------------


def test_group_prompts_by_target_keeps_file_order(tokenizer) -> None:
    prompts = [p for i in range(3) for p in build_group(tokenizer, i).values()]

    groups = group_prompts_by_target(prompts)

    assert [g[PromptCondition.ZERO_SHOT].target.question for g in groups] == [
        make_question("bio", i, words=3).question for i in range(3)
    ]
    assert all(set(g) == set(CONDITIONS) for g in groups)


def test_group_prompts_by_target_rejects_duplicate_conditions(tokenizer) -> None:
    prompts = list(build_group(tokenizer).values()) * 2

    with pytest.raises(ValueError, match="duplicate"):
        group_prompts_by_target(prompts)


# --- evaluate_group -------------------------------------------------------------------


def test_evaluate_group_returns_one_record_per_condition(model, structure, tokenizer) -> None:
    group = build_group(tokenizer)

    records = evaluate_group(model, tokenizer, structure, group, run="r", question_index=7)

    assert [r.condition for r in records] == list(group)
    assert {r.question_index for r in records} == {7}
    assert {r.run for r in records} == {"r"}


def test_zero_shot_has_no_jaccard_and_others_have_one_value_per_layer(
    model, structure, tokenizer
) -> None:
    records = evaluate_group(
        model, tokenizer, structure, build_group(tokenizer), run="r", question_index=0
    )

    by_condition = {r.condition: r for r in records}
    assert by_condition[PromptCondition.ZERO_SHOT].jaccard_per_layer is None
    for condition in CONDITIONS[1:]:
        jaccard = by_condition[condition].jaccard_per_layer
        assert jaccard is not None
        assert len(jaccard) == structure.num_layers
        assert all(0.0 <= value <= 1.0 for value in jaccard)


def test_jaccard_matches_an_independent_computation(model, structure, tokenizer) -> None:
    group = build_group(tokenizer)
    zero, few = group[PromptCondition.ZERO_SHOT], group[PromptCondition.CROSS_DOMAIN]
    recorded = {}
    for key, prompt in (("zero", zero), ("few", few)):
        with RoutingRecorder(model, TINY_ROUTER_NAMES, TINY_TOP_K) as recorder:
            score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
        recorded[key] = recorder.get_record()
    alignment = align_target_tokens(tokenizer, zero, few)
    expected = per_layer_jaccard(
        recorded["zero"].select_positions(alignment.positions_a).topk_indices,
        recorded["few"].select_positions(alignment.positions_b).topk_indices,
    )

    records = evaluate_group(model, tokenizer, structure, group, run="r", question_index=0)

    actual = next(r for r in records if r.condition is PromptCondition.CROSS_DOMAIN)
    assert actual.jaccard_per_layer == pytest.approx(expected.tolist())


def test_scores_match_plain_scoring_without_hooks(model, structure, tokenizer) -> None:
    group = build_group(tokenizer)

    records = evaluate_group(model, tokenizer, structure, group, run="r", question_index=0)

    for record in records:
        prompt = group[record.condition]
        plain = score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
        assert record.option_logprobs == pytest.approx(plain.option_logprobs)
        assert record.prediction == plain.prediction
        assert record.is_correct == plain.is_correct
        assert record.num_examples == prompt.num_examples
        assert record.example_tokens == prompt.example_tokens
        assert record.subject == prompt.target.subject


def test_evaluate_group_requires_the_zero_shot_baseline(model, structure, tokenizer) -> None:
    group = build_group(tokenizer, conditions=CONDITIONS[1:])

    with pytest.raises(ValueError, match="zero_shot"):
        evaluate_group(model, tokenizer, structure, group, run="r", question_index=0)


def test_evaluate_group_leaves_no_hooks_behind(model, structure, tokenizer) -> None:
    evaluate_group(model, tokenizer, structure, build_group(tokenizer), run="r", question_index=0)

    assert all(len(model.get_submodule(n)._forward_hooks) == 0 for n in TINY_ROUTER_NAMES)
    assert len(model._forward_pre_hooks) == 0


# --- run_prompt_file --------------------------------------------------------------------


def test_run_prompt_file_writes_every_question_and_condition(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, count=3)
    output = tmp_path / "results" / "bio_med_a2b.jsonl"

    evaluated = run_prompt_file(model, tokenizer, structure, prompts_path, output)

    records = read_records(output)
    assert evaluated == 3
    assert len(records) == 3 * len(CONDITIONS)
    assert {r.question_index for r in records} == {0, 1, 2}
    assert {r.run for r in records} == {"bio_med_a2b"}


def test_rerunning_a_finished_file_does_no_work(tmp_path, model, structure, tokenizer) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer)
    output = tmp_path / "out.jsonl"
    run_prompt_file(model, tokenizer, structure, prompts_path, output)
    before = output.read_bytes()

    evaluated = run_prompt_file(model, tokenizer, structure, prompts_path, output)

    assert evaluated == 0
    assert output.read_bytes() == before


def test_resuming_after_a_crash_matches_an_uninterrupted_run(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, count=3)
    full = tmp_path / "full.jsonl"
    run_prompt_file(model, tokenizer, structure, prompts_path, full)
    complete = read_records(full)
    # Simulate a crash: keep question 0, half of question 1, and a torn final line.
    keep = len(CONDITIONS) + 2
    resumed = tmp_path / "resumed.jsonl"
    shutil.copy(manifest_path_for(full), manifest_path_for(resumed))  # a crash leaves it
    append_records(resumed, complete[:keep])
    with resumed.open("a", encoding="utf-8") as handle:
        handle.write('{"run": "bio_med_a2b", "quest')

    evaluated = run_prompt_file(model, tokenizer, structure, prompts_path, resumed)

    assert evaluated == 2  # the half-finished question 1 and question 2
    assert read_records(resumed) == complete  # no duplicates, same order


def test_run_prompt_file_is_deterministic(tmp_path, model, structure, tokenizer) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer)
    first, second = tmp_path / "1.jsonl", tmp_path / "2.jsonl"

    run_prompt_file(model, tokenizer, structure, prompts_path, first)
    run_prompt_file(model, tokenizer, structure, prompts_path, second)

    assert read_records(first) == read_records(second)
    assert torch.is_grad_enabled()  # scoring must not leak inference-mode state


# --- hidden-state shift --------------------------------------------------------------------------

from rr.hidden import HiddenShiftMeter  # noqa: E402
from rr.manifest import manifest_path_for as _manifest_path  # noqa: E402


def make_meter(model, structure) -> HiddenShiftMeter:
    return HiddenShiftMeter.from_model(model, structure.router_module_names)


def test_hidden_shift_is_measured_for_every_condition_but_zero_shot(
    model, structure, tokenizer
) -> None:
    records = evaluate_group(
        model,
        tokenizer,
        structure,
        build_group(tokenizer),
        run="r",
        question_index=0,
        meter=make_meter(model, structure),
    )

    by_condition = {r.condition: r for r in records}
    assert by_condition[PromptCondition.ZERO_SHOT].hidden_shift is None
    hidden_size = model.config.hidden_size
    for condition in CONDITIONS[1:]:
        shift = by_condition[condition].hidden_shift
        assert shift is not None
        assert len(shift.relative_l2) == structure.num_layers
        assert all(value > 0 for value in shift.relative_l2)  # extra context moves the state
        assert shift.router_visible_chance[0] == pytest.approx(structure.num_experts / hidden_size)


def test_measuring_the_hidden_shift_leaves_scores_and_routing_unchanged(
    model, structure, tokenizer
) -> None:
    group = build_group(tokenizer)
    plain = evaluate_group(model, tokenizer, structure, group, run="r", question_index=0)
    measured = evaluate_group(
        model,
        tokenizer,
        structure,
        group,
        run="r",
        question_index=0,
        meter=make_meter(model, structure),
    )

    assert all(r.hidden_shift is None for r in plain)
    for a, b in zip(plain, measured, strict=True):
        assert a.option_logprobs == b.option_logprobs
        assert a.jaccard_per_layer == b.jaccard_per_layer


def test_hidden_shift_is_written_by_a_run_when_enabled(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, count=2)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"

    run_prompt_file(model, tokenizer, structure, prompts_path, output, record_hidden_states=True)

    records = read_records(output)
    few_shot = [r for r in records if r.condition is not PromptCondition.ZERO_SHOT]
    assert all(r.hidden_shift is not None for r in few_shot)
    assert all(r.hidden_shift is None for r in records if r.condition is PromptCondition.ZERO_SHOT)


def test_hidden_shift_is_not_recorded_by_default(tmp_path, model, structure, tokenizer) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, count=1)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"

    run_prompt_file(model, tokenizer, structure, prompts_path, output)

    assert all(r.hidden_shift is None for r in read_records(output))


def test_toggling_hidden_shift_recording_between_resumes_is_refused(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, count=1)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    run_prompt_file(model, tokenizer, structure, prompts_path, output)

    with pytest.raises(ValueError, match="record_hidden_states"):
        run_prompt_file(
            model, tokenizer, structure, prompts_path, output, record_hidden_states=True
        )

    assert _manifest_path(output).exists()
