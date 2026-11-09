"""Tests for rr.manifest and the resume-safety it gives the experiment runners."""

from __future__ import annotations

import json
import shutil

import pytest

from conftest import TinyVocabTokenizer, build_tiny_moe, write_prompt_file
from rr.config import ArchitectureConfig, ModelConfig
from rr.force_experiment import ForcingOptions, ForcingPlan, run_forcing_file
from rr.manifest import build_manifest, ensure_manifest, manifest_path_for
from rr.models import inspect_moe_structure, model_provenance
from rr.pilot import run_prompt_file
from rr.prompts import PromptCondition
from rr.results import read_records, read_results_dir
from rr.runner import run_resumable

ZERO, CROSS = PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN
PLANS = (ForcingPlan(CROSS, ZERO), ForcingPlan(ZERO, ZERO))
PROVENANCE = {"name": "tiny", "revision": "abc123", "dtype": "float32"}


@pytest.fixture(scope="module")
def model():
    return build_tiny_moe()


@pytest.fixture(scope="module")
def structure(model):
    return inspect_moe_structure(model)


@pytest.fixture
def tokenizer() -> TinyVocabTokenizer:
    return TinyVocabTokenizer()


def prompts_file(tmp_path, tokenizer, count=2, name="bio_med_a2b"):
    return write_prompt_file(tmp_path, tokenizer, count, name, conditions=(ZERO, CROSS))


def force(model, structure, tokenizer, prompts_path, output, **overrides):
    kwargs = {"options": ForcingOptions(), "plans": PLANS, "model_provenance": PROVENANCE}
    kwargs.update(overrides)
    return run_forcing_file(
        model,
        tokenizer,
        structure,
        prompts_path,
        output,
        kwargs["plans"],
        kwargs["options"],
        model_provenance=kwargs["model_provenance"],
    )


# --- manifest unit tests ---------------------------------------------------------------


def test_manifest_sits_next_to_the_results_and_is_not_read_as_a_result(tmp_path) -> None:
    assert manifest_path_for(tmp_path / "a" / "run.jsonl") == tmp_path / "a" / "run.manifest.json"


def test_build_manifest_records_run_settings_and_prompt_hash(tmp_path) -> None:
    path = tmp_path / "bio.jsonl"
    path.write_text("one\n")
    first = build_manifest("bio", path, {"layers": (1, 2)})
    path.write_text("two\n")
    second = build_manifest("bio", path, {"layers": (1, 2)})

    assert first["run"] == "bio"
    assert first["settings"] == {"layers": [1, 2]}  # JSON-normalized: tuples become lists
    assert len(first["prompts_sha256"]) == 64
    assert first["prompts_sha256"] != second["prompts_sha256"]


def test_ensure_manifest_writes_it_when_nothing_exists_yet(tmp_path) -> None:
    output = tmp_path / "out" / "run.jsonl"
    manifest = {"run": "run", "settings": {"a": 1}, "prompts_sha256": "x"}

    ensure_manifest(output, manifest)

    assert json.loads(manifest_path_for(output).read_text()) == manifest
    assert list(output.parent.glob("*.tmp")) == []


def test_ensure_manifest_accepts_an_identical_manifest(tmp_path) -> None:
    output = tmp_path / "run.jsonl"
    manifest = {"run": "run", "settings": {"a": [1, 2]}, "prompts_sha256": "x"}
    ensure_manifest(output, manifest)

    ensure_manifest(output, manifest)  # does not raise


def test_ensure_manifest_names_every_difference(tmp_path) -> None:
    output = tmp_path / "run.jsonl"
    ensure_manifest(
        output, {"run": "run", "settings": {"opt": {"w": False}}, "prompts_sha256": "x"}
    )

    with pytest.raises(ValueError) as error:
        ensure_manifest(
            output, {"run": "run", "settings": {"opt": {"w": True}}, "prompts_sha256": "y"}
        )

    message = str(error.value)
    assert "settings.opt.w" in message
    assert "prompts_sha256" in message
    assert "run.manifest.json" in message


def test_results_without_a_manifest_are_not_resumed(tmp_path) -> None:
    output = tmp_path / "run.jsonl"
    output.write_text('{"some": "record"}\n')

    with pytest.raises(ValueError, match="no manifest"):
        ensure_manifest(output, {"run": "run", "settings": {}, "prompts_sha256": "x"})


def test_a_corrupt_manifest_fails_loudly(tmp_path) -> None:
    output = tmp_path / "run.jsonl"
    manifest_path_for(output).write_text("{not json")

    with pytest.raises(ValueError, match="manifest"):
        ensure_manifest(output, {"run": "run", "settings": {}, "prompts_sha256": "x"})


def test_model_provenance_names_the_model_but_not_the_device() -> None:
    architecture = ArchitectureConfig(2, 8, 2, "model.layers.{layer}.mlp.gate")
    config = ModelConfig("org/m", "deadbeef", "bfloat16", "cuda", architecture)

    assert model_provenance(config) == {
        "name": "org/m",
        "revision": "deadbeef",
        "dtype": "bfloat16",
    }


# --- integration through the runners ----------------------------------------------------------


def test_the_manifest_exists_before_the_first_question_is_evaluated(tmp_path, tokenizer) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"

    def crash(group, index):
        raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        run_resumable(
            prompts_path, output, crash, lambda group: {(ZERO, "none")}, settings={"k": 1}
        )

    assert manifest_path_for(output).exists()
    assert not output.exists()


def test_pilot_and_forcing_runs_write_manifests_that_results_loading_ignores(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    out = tmp_path / "out"
    run_prompt_file(
        model,
        tokenizer,
        structure,
        prompts_path,
        out / "bio_med_a2b.jsonl",
        model_provenance=PROVENANCE,
    )

    manifest = json.loads(manifest_path_for(out / "bio_med_a2b.jsonl").read_text())
    assert manifest["run"] == "bio_med_a2b"
    assert manifest["settings"]["model"] == PROVENANCE
    assert {r.run for r in read_results_dir(out)} == {"bio_med_a2b"}


def test_forcing_manifest_records_plans_options_and_model(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    options = ForcingOptions(force_weights=True, layers=(0,), random_seed=3)

    force(model, structure, tokenizer, prompts_path, output, options=options)

    settings = json.loads(manifest_path_for(output).read_text())["settings"]
    assert settings["model"] == PROVENANCE
    assert settings["options"] == {"force_weights": True, "layers": [0], "random_seed": 3}
    assert settings["plans"] == [
        ["cross_domain", "zero_shot"],
        ["zero_shot", "zero_shot"],
    ]


def test_resuming_with_unchanged_settings_is_fine(tmp_path, model, structure, tokenizer) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    force(model, structure, tokenizer, prompts_path, output)

    assert force(model, structure, tokenizer, prompts_path, output) == 0


@pytest.mark.parametrize(
    ("overrides", "expected_key"),
    [
        ({"options": ForcingOptions(force_weights=True)}, "force_weights"),
        ({"options": ForcingOptions(layers=(0,))}, "layers"),
        ({"options": ForcingOptions(random_seed=9)}, "random_seed"),
        ({"plans": (ForcingPlan(CROSS, ZERO),)}, "plans"),
        ({"model_provenance": {**PROVENANCE, "revision": "other"}}, "revision"),
    ],
)
def test_resuming_with_changed_settings_is_refused_and_leaves_results_alone(
    tmp_path, model, structure, tokenizer, overrides, expected_key
) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    force(model, structure, tokenizer, prompts_path, output)
    before = output.read_bytes()

    with pytest.raises(ValueError, match=expected_key):
        force(model, structure, tokenizer, prompts_path, output, **overrides)

    assert output.read_bytes() == before


def test_resuming_after_the_prompt_file_changed_is_refused(
    tmp_path, model, structure, tokenizer
) -> None:
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    force(model, structure, tokenizer, prompts_file(tmp_path, tokenizer, count=2), output)

    with pytest.raises(ValueError, match="prompts_sha256"):
        force(model, structure, tokenizer, prompts_file(tmp_path, tokenizer, count=3), output)


def test_results_from_before_manifests_existed_are_not_silently_resumed(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    output = tmp_path / "out" / "bio_med_a2b.jsonl"
    force(model, structure, tokenizer, prompts_path, output)
    manifest_path_for(output).unlink()

    with pytest.raises(ValueError, match="no manifest"):
        force(model, structure, tokenizer, prompts_path, output)


# --- inner guards still hold when the manifest matches --------------------------------------


def test_results_that_belong_to_another_run_are_refused(
    tmp_path, model, structure, tokenizer
) -> None:
    mine = prompts_file(tmp_path, tokenizer, name="bio_med_a2b")
    other = prompts_file(tmp_path, tokenizer, name="chem_phys_a2b")
    output = tmp_path / "out.jsonl"
    foreign = tmp_path / "foreign.jsonl"
    force(model, structure, tokenizer, mine, output)  # writes a manifest valid for `mine`
    force(model, structure, tokenizer, other, foreign)
    shutil.copy(foreign, output)  # now the file holds another run's records

    with pytest.raises(ValueError, match="holds run"):
        force(model, structure, tokenizer, mine, output)


def test_results_with_more_questions_than_the_prompt_file_are_refused(
    tmp_path, model, structure, tokenizer
) -> None:
    short = prompts_file(tmp_path, tokenizer, count=1)
    long_dir = tmp_path / "long"
    long_dir.mkdir()
    long = prompts_file(long_dir, tokenizer, count=3)
    output = tmp_path / "out.jsonl"
    force(model, structure, tokenizer, short, output)  # manifest valid for `short`
    long_output = tmp_path / "long_out.jsonl"
    force(model, structure, tokenizer, long, long_output)
    shutil.copy(long_output, output)

    with pytest.raises(ValueError, match="does not match"):
        force(model, structure, tokenizer, short, output)

    assert len(read_records(output)) > 0


def test_results_may_not_be_written_over_the_prompt_file(
    tmp_path, model, structure, tokenizer
) -> None:
    prompts_path = prompts_file(tmp_path, tokenizer)
    before = prompts_path.read_bytes()

    with pytest.raises(ValueError, match="prompt file"):
        force(model, structure, tokenizer, prompts_path, prompts_path)

    assert prompts_path.read_bytes() == before
