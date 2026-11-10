"""Tests for rr.force_experiment: plans, forced evaluation, resumable runs, option parsing."""

from __future__ import annotations

import shutil

import pytest
import torch

from conftest import (
    TINY_ROUTER_NAMES,
    TinyVocabTokenizer,
    build_group,
    build_tiny_moe,
    write_prompt_file,
)
from rr.force_experiment import (
    ForcingOptions,
    ForcingPlan,
    evaluate_forcing_group,
    parse_forcing_options,
    plans_for_force_routing,
    plans_for_reverse_check,
    run_forcing_file,
)
from rr.manifest import manifest_path_for
from rr.models import inspect_moe_structure
from rr.prompts import PromptCondition
from rr.results import append_records, read_records
from rr.routing.align import align_target_tokens
from rr.routing.force import RoutingForcer, spec_from_record
from rr.routing.record import RoutingRecorder
from rr.scoring import score_mc

ZERO, CROSS = PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN
PLAIN = "none"


@pytest.fixture(scope="module")
def model():
    return build_tiny_moe()


@pytest.fixture(scope="module")
def structure(model):
    return inspect_moe_structure(model)


@pytest.fixture
def tokenizer() -> TinyVocabTokenizer:
    return TinyVocabTokenizer()


def by_key(records):
    return {(r.condition, r.intervention): r for r in records}


def evaluate(model, tokenizer, structure, plans, options=None, index=0, conditions=(ZERO, CROSS)):
    group = build_group(tokenizer, index, conditions)
    return evaluate_forcing_group(
        model,
        tokenizer,
        structure,
        group,
        plans,
        run="r",
        question_index=index,
        options=options or ForcingOptions(),
    )


# --- plans ----------------------------------------------------------------------------------


def test_plan_intervention_names() -> None:
    assert ForcingPlan(CROSS, ZERO).intervention == "force_zero_shot_routing"
    assert ForcingPlan(ZERO, CROSS).intervention == "force_cross_domain_routing"
    assert ForcingPlan(CROSS, None).intervention == "force_random_routing"


def test_force_routing_plans_run_each_few_shot_prompt_with_zero_shot_routing() -> None:
    plans = plans_for_force_routing([ZERO, CROSS], identity_check=False, random_check=False)

    assert plans == (ForcingPlan(CROSS, ZERO),)


def test_force_routing_plans_add_the_sanity_controls() -> None:
    plans = plans_for_force_routing([ZERO, CROSS], identity_check=True, random_check=True)

    assert set(plans) == {
        ForcingPlan(CROSS, ZERO),
        ForcingPlan(ZERO, ZERO),
        ForcingPlan(CROSS, None),
    }


def test_reverse_check_plans_force_each_few_shot_routing_into_the_zero_shot_prompt() -> None:
    plans = plans_for_reverse_check([ZERO, CROSS, PromptCondition.SAME_DOMAIN])

    assert plans == (
        ForcingPlan(ZERO, CROSS),
        ForcingPlan(ZERO, PromptCondition.SAME_DOMAIN),
    )


def test_plans_reject_duplicate_conditions() -> None:
    duplicated = [ZERO, CROSS, CROSS]

    with pytest.raises(ValueError, match="duplicate"):
        plans_for_force_routing(duplicated, identity_check=True, random_check=True)
    with pytest.raises(ValueError, match="duplicate"):
        plans_for_reverse_check(duplicated)


def test_plans_need_at_least_one_few_shot_condition() -> None:
    with pytest.raises(ValueError, match="few-shot"):
        plans_for_force_routing([ZERO], identity_check=True, random_check=True)
    with pytest.raises(ValueError, match="few-shot"):
        plans_for_reverse_check([ZERO])


# --- evaluate_forcing_group -------------------------------------------------------------------


def test_plain_baselines_and_forced_records_are_all_emitted(model, structure, tokenizer) -> None:
    records = evaluate(model, tokenizer, structure, (ForcingPlan(CROSS, ZERO),))

    assert set(by_key(records)) == {
        (ZERO, PLAIN),
        (CROSS, PLAIN),
        (CROSS, "force_zero_shot_routing"),
    }
    assert {r.run for r in records} == {"r"}
    assert all(r.jaccard_per_layer is None for r in records)


def test_plain_baselines_match_ordinary_scoring(model, structure, tokenizer) -> None:
    group = build_group(tokenizer, 0, (ZERO, CROSS))
    records = by_key(evaluate(model, tokenizer, structure, (ForcingPlan(CROSS, ZERO),)))

    for condition in (ZERO, CROSS):
        prompt = group[condition]
        plain = score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
        assert records[(condition, PLAIN)].option_logprobs == pytest.approx(plain.option_logprobs)


def test_forcing_a_prompts_own_routing_is_the_identity(model, structure, tokenizer) -> None:
    records = by_key(evaluate(model, tokenizer, structure, (ForcingPlan(ZERO, ZERO),)))

    assert records[(ZERO, "force_zero_shot_routing")].option_logprobs == pytest.approx(
        records[(ZERO, PLAIN)].option_logprobs, abs=1e-5
    )


@pytest.mark.parametrize("force_weights", [False, True])
def test_forced_scores_match_an_independent_forced_run(
    model, structure, tokenizer, force_weights
) -> None:
    group = build_group(tokenizer, 0, (ZERO, CROSS))
    zero, few = group[ZERO], group[CROSS]
    with RoutingRecorder(model, TINY_ROUTER_NAMES, structure.top_k) as recorder:
        score_mc(model, tokenizer, zero.text, zero.target.answer_index)
    alignment = align_target_tokens(tokenizer, zero, few)
    spec = spec_from_record(
        recorder.get_record(),
        alignment.positions_a,
        alignment.positions_b,
        force_weights=force_weights,
    )
    with RoutingForcer(model, TINY_ROUTER_NAMES, spec):
        expected = score_mc(model, tokenizer, few.text, few.target.answer_index)

    records = by_key(
        evaluate(
            model,
            tokenizer,
            structure,
            (ForcingPlan(CROSS, ZERO),),
            ForcingOptions(force_weights=force_weights),
        )
    )

    forced = records[(CROSS, "force_zero_shot_routing")]
    assert forced.option_logprobs == pytest.approx(expected.option_logprobs, abs=1e-6)


def test_forcing_zero_shot_routing_changes_the_few_shot_scores(model, structure, tokenizer) -> None:
    changed = []
    for index in range(4):
        records = by_key(
            evaluate(model, tokenizer, structure, (ForcingPlan(CROSS, ZERO),), index=index)
        )
        plain = records[(CROSS, PLAIN)].option_logprobs
        forced = records[(CROSS, "force_zero_shot_routing")].option_logprobs
        changed.append(not torch.allclose(torch.tensor(plain), torch.tensor(forced), atol=1e-6))

    assert any(changed)


def test_forcing_no_layers_changes_nothing(model, structure, tokenizer) -> None:
    records = by_key(
        evaluate(
            model,
            tokenizer,
            structure,
            (ForcingPlan(CROSS, ZERO),),
            ForcingOptions(layers=()),
        )
    )

    assert records[(CROSS, "force_zero_shot_routing")].option_logprobs == pytest.approx(
        records[(CROSS, PLAIN)].option_logprobs
    )


def test_random_forcing_is_reproducible_for_a_seed_and_varies_with_it(
    model, structure, tokenizer
) -> None:
    plan = (ForcingPlan(CROSS, None),)

    def forced(seed):
        records = evaluate(model, tokenizer, structure, plan, ForcingOptions(random_seed=seed))
        return by_key(records)[(CROSS, "force_random_routing")].option_logprobs

    assert forced(1) == forced(1)
    assert forced(1) != forced(2)


def test_random_forcing_differs_between_questions(model, structure, tokenizer) -> None:
    plan = (ForcingPlan(CROSS, None),)

    def forced(index):
        records = evaluate(model, tokenizer, structure, plan, index=index)
        return by_key(records)[(CROSS, "force_random_routing")].option_logprobs

    assert forced(0) != forced(1)


def test_a_missing_condition_raises(model, structure, tokenizer) -> None:
    with pytest.raises(ValueError, match="cross_domain"):
        evaluate(
            model,
            tokenizer,
            structure,
            (ForcingPlan(CROSS, ZERO),),
            conditions=(ZERO,),
        )


def test_no_hooks_are_left_behind(model, structure, tokenizer) -> None:
    evaluate(model, tokenizer, structure, (ForcingPlan(CROSS, ZERO), ForcingPlan(CROSS, None)))

    assert all(len(model.get_submodule(n)._forward_hooks) == 0 for n in TINY_ROUTER_NAMES)
    assert len(model._forward_pre_hooks) == 0


# --- run_forcing_file --------------------------------------------------------------------------

PLANS = (ForcingPlan(CROSS, ZERO), ForcingPlan(ZERO, ZERO), ForcingPlan(CROSS, None))
PER_QUESTION = 2 + len(PLANS)  # two plain baselines plus one record per plan


def run(model, structure, tokenizer, prompts_path, output):
    return run_forcing_file(
        model, tokenizer, structure, prompts_path, output, PLANS, ForcingOptions()
    )


def test_run_forcing_file_writes_baselines_and_every_intervention(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, 3, conditions=(ZERO, CROSS))
    output = tmp_path / "out" / "bio_med_a2b.jsonl"

    evaluated = run(model, structure, tokenizer, prompts_path, output)

    records = read_records(output)
    assert evaluated == 3
    assert len(records) == 3 * PER_QUESTION
    assert {r.intervention for r in records} == {
        PLAIN,
        "force_zero_shot_routing",
        "force_random_routing",
    }


def test_rerunning_a_finished_forcing_file_does_no_work(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, 2, conditions=(ZERO, CROSS))
    output = tmp_path / "out.jsonl"
    run(model, structure, tokenizer, prompts_path, output)
    before = output.read_bytes()

    assert run(model, structure, tokenizer, prompts_path, output) == 0
    assert output.read_bytes() == before


def test_forcing_run_resumes_after_a_crash(tmp_path, model, structure, tokenizer) -> None:
    prompts_path = write_prompt_file(tmp_path, tokenizer, 3, conditions=(ZERO, CROSS))
    full = tmp_path / "full.jsonl"
    run(model, structure, tokenizer, prompts_path, full)
    complete = read_records(full)
    resumed = tmp_path / "resumed.jsonl"
    shutil.copy(manifest_path_for(full), manifest_path_for(resumed))  # a crash leaves it
    append_records(resumed, complete[: PER_QUESTION + 3])  # question 0 + part of question 1
    with resumed.open("a", encoding="utf-8") as handle:
        handle.write('{"run": "bio_med_a2b", "quest')

    evaluated = run(model, structure, tokenizer, prompts_path, resumed)

    assert evaluated == 2
    assert read_records(resumed) == complete


# --- parse_forcing_options ------------------------------------------------------------------------


def test_default_options() -> None:
    assert parse_forcing_options({}, seed=7) == ForcingOptions(
        force_weights=True, layers=None, random_seed=7
    )


def test_options_are_read_from_the_config_mapping() -> None:
    options = parse_forcing_options(
        {
            "force_scope": "target_question_tokens",
            "force_mixing_weights": True,
            "layers": [3, 1],
        },
        seed=0,
    )

    assert options == ForcingOptions(force_weights=True, layers=(3, 1), random_seed=0)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"force_scope": "all_tokens"}, "force_scope"),
        ({"force_mixing_weights": "yes"}, "force_mixing_weights"),
        ({"layers": "all"}, "layers"),
        ({"layers": [1, "x"]}, "layers"),
        ({"layers": [-1]}, "layers"),
    ],
)
def test_invalid_options_are_rejected(options, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_forcing_options(options, seed=0)


def test_selection_only_forcing_is_still_available() -> None:
    options = parse_forcing_options({"force_mixing_weights": False}, seed=0)

    assert options.force_weights is False
