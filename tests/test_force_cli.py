"""End-to-end tests of the forcing scripts' shared driver, on the tiny model (no downloads)."""

from __future__ import annotations

import pytest
import yaml

import rr.force_cli as force_cli
from conftest import TinyVocabTokenizer, build_tiny_moe, write_prompt_file
from rr.force_cli import force_routing_plans, reverse_check_plans, run_forcing_script
from rr.prompts import PromptCondition
from rr.results import read_results_dir

ZERO, CROSS = PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN
ARCHITECTURE = {
    "num_layers": 2,
    "num_experts": 8,
    "top_k": 2,
    "router_module_pattern": "model.layers.{layer}.mlp.gate",
}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """A config on disk pointing at tiny prompts; ``load_model`` returns the tiny MoE."""
    tokenizer = TinyVocabTokenizer()
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    write_prompt_file(prompts_dir, tokenizer, 3, "bio_med_a2b", conditions=(ZERO, CROSS))
    write_prompt_file(prompts_dir, tokenizer, 2, "chem_phys_a2b", conditions=(ZERO, CROSS))
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "revision": None,
                "dtype": "float32",
                "device": "cpu",
                "architecture": ARCHITECTURE,
            }
        )
    )
    monkeypatch.setattr(force_cli, "load_model", lambda *a, **k: (build_tiny_moe(), tokenizer))

    def write(options=None, name="force_routing", prompts=prompts_dir):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "model_config": str(model_config),
                    "pairs_config": "unused.yaml",
                    "seed": 0,
                    "num_shots": 2,
                    "num_questions_per_target": 3,
                    "conditions": ["zero_shot", "cross_domain"],
                    "prompts_dir": str(prompts),
                    "output_dir": str(tmp_path / "results"),
                    "options": options or {},
                }
            )
        )
        return ["--config", str(path)]

    write.results = tmp_path / "results"
    return write


def test_force_routing_script_writes_results_and_prints_the_summary(experiment, capsys) -> None:
    run_forcing_script("doc", force_routing_plans, experiment())

    records = read_results_dir(experiment.results)
    interventions = {r.intervention for r in records}
    assert interventions == {"none", "force_zero_shot_routing", "force_random_routing"}
    assert {r.run for r in records} == {"bio_med_a2b", "chem_phys_a2b"}
    output = capsys.readouterr().out
    assert "retained" in output
    assert "Sanity controls" in output


def test_reverse_check_script_forces_few_shot_routing_into_the_zero_shot_prompt(
    experiment, capsys
) -> None:
    run_forcing_script("doc", reverse_check_plans, experiment(name="reverse_check"))

    records = read_results_dir(experiment.results)
    assert {r.intervention for r in records} == {"none", "force_cross_domain_routing"}
    assert "recovered" in capsys.readouterr().out


def test_sanity_controls_can_be_switched_off(experiment) -> None:
    options = {"sanity_identity_check": False, "sanity_random_check": False}

    run_forcing_script("doc", force_routing_plans, experiment(options))

    interventions = {r.intervention for r in read_results_dir(experiment.results)}
    assert interventions == {"none", "force_zero_shot_routing"}


def test_rerunning_the_script_resumes_instead_of_repeating(experiment, capsys) -> None:
    args = experiment()
    run_forcing_script("doc", force_routing_plans, args)
    before = [p.read_bytes() for p in sorted(experiment.results.glob("*.jsonl"))]
    capsys.readouterr()

    run_forcing_script("doc", force_routing_plans, args)

    assert [p.read_bytes() for p in sorted(experiment.results.glob("*.jsonl"))] == before
    assert "evaluated 0 questions" in capsys.readouterr().out


def test_missing_prompt_files_point_at_the_build_step(experiment, tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="01_build_prompts"):
        run_forcing_script("doc", force_routing_plans, experiment(prompts=empty))


def test_invalid_options_fail_before_the_model_is_loaded(experiment, monkeypatch) -> None:
    def must_not_load(*args, **kwargs):
        raise AssertionError("model loaded before the options were validated")

    monkeypatch.setattr(force_cli, "load_model", must_not_load)

    with pytest.raises(ValueError, match="force_scope"):
        run_forcing_script("doc", force_routing_plans, experiment({"force_scope": "all_tokens"}))


def test_a_config_that_misdescribes_the_model_fails_loudly(experiment, tmp_path) -> None:
    args = experiment()
    model_yaml = tmp_path / "model.yaml"
    config = yaml.safe_load(model_yaml.read_text())
    config["architecture"]["top_k"] = 8
    model_yaml.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="top_k"):
        run_forcing_script("doc", force_routing_plans, args)
