"""End to end on a tiny MoE: shot-grid prompts -> pilot / forcing runs -> k-curve summaries."""

from __future__ import annotations

import pytest

from conftest import TinyVocabTokenizer, build_tiny_moe, make_question
from rr.force_experiment import ForcingOptions, plans_for_force_routing, run_forcing_file
from rr.models import inspect_moe_structure
from rr.pilot import run_prompt_file
from rr.prompt_build import build_direction_prompts, write_runs
from rr.prompts import PromptCondition
from rr.results import read_results_dir
from rr.shot_curves import (
    gain_by_shots,
    mean_by_shots,
    summarize_force_results,
    summarize_pilot_results,
)

ALL = tuple(PromptCondition)
TWO = (PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN)


@pytest.fixture(scope="module")
def model():
    return build_tiny_moe()


@pytest.fixture(scope="module")
def structure(model):
    return inspect_moe_structure(model)


def _build(conditions, tmp_path):
    runs = build_direction_prompts(
        direction="chem_bio:a2b",
        source_pool=[make_question("chem", i) for i in range(12)],
        target_pool=[make_question("bio", i) for i in range(12)],
        eval_pool=[make_question("bioeval", i) for i in range(10)],
        irrelevant_pool=[make_question("hist", i, words=4) for i in range(40)],
        tokenizer=TinyVocabTokenizer(),
        conditions=conditions,
        num_shots=3,
        shot_counts=(1, 2, 3),
        num_replicates=2,
        num_questions=3,
        seed=0,
    )
    write_runs(runs, tmp_path / "prompts")
    return tmp_path / "prompts"


def test_pilot_over_the_shot_grid_gives_a_gain_curve_for_every_k(
    tmp_path, model, structure
) -> None:
    prompts_dir = _build(ALL, tmp_path)
    for path in sorted(prompts_dir.glob("*.jsonl")):
        run_prompt_file(model, TinyVocabTokenizer(), structure, path, tmp_path / "out" / path.name)

    records = read_results_dir(tmp_path / "out")

    gains = gain_by_shots(records, "accuracy", n_boot=50)
    assert {(g.condition, g.shots) for g in gains} == {(c, k) for c in ALL[1:] for k in (1, 2, 3)}
    assert all(g.n_replicates == 2 and g.n_questions == 3 for g in gains)
    overlap = mean_by_shots(records, "jaccard")
    assert overlap and all(0.0 <= s.mean <= 1.0 for s in overlap)
    report = summarize_pilot_results(records, n_boot=50)
    assert "k=1" in report and "k=3" in report and "chem_bio_a2b" in report


def test_rerunning_a_finished_shot_grid_run_evaluates_nothing(tmp_path, model, structure) -> None:
    path = next(iter(sorted(_build(TWO, tmp_path).glob("*_r0_k2.jsonl"))))
    output = tmp_path / "out" / path.name
    run_prompt_file(model, TinyVocabTokenizer(), structure, path, output)

    assert run_prompt_file(model, TinyVocabTokenizer(), structure, path, output) == 0


def test_forcing_over_the_shot_grid_summarises_fractions_per_k(tmp_path, model, structure) -> None:
    prompts_dir = _build(TWO, tmp_path)
    plans = plans_for_force_routing(TWO, identity_check=True, random_check=True)
    for path in sorted(prompts_dir.glob("*.jsonl")):
        run_forcing_file(
            model,
            TinyVocabTokenizer(),
            structure,
            path,
            tmp_path / "out" / path.name,
            plans,
            ForcingOptions(),
        )

    records = read_results_dir(tmp_path / "out")

    report = summarize_force_results(records, n_boot=50)
    assert "k=1" in report and "k=3" in report
    assert "identity ok in 6/6 runs" in report
