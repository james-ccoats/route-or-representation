"""Tests for rr.config: shipped configs load, and bad configs fail loudly."""

from pathlib import Path

import pytest
import yaml

from rr.config import (
    VALID_DISTANCES,
    ConfigError,
    load_experiment_config,
    load_model_config,
    load_pairs_config,
    parse_config_arg,
)
from rr.prompts import PromptCondition

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
EXPERIMENT_NAMES = ["pilot", "force_routing", "reverse_check", "ladder_pilot"]


def _write(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _model_dict() -> dict:
    return {
        "name": "org/model",
        "dtype": "bfloat16",
        "device": "cpu",
        "architecture": {
            "num_layers": 2,
            "num_experts": 4,
            "top_k": 2,
            "router_module_pattern": "layers.{layer}.gate",
        },
    }


def test_olmoe_config_loads_and_expands_router_names() -> None:
    cfg = load_model_config(CONFIGS / "models" / "olmoe.yaml")

    names = cfg.architecture.router_module_names()

    assert cfg.name == "allenai/OLMoE-1B-7B-0924"
    assert len(names) == cfg.architecture.num_layers
    assert names[0] != names[1]


def test_pilot_pairs_config_runs_both_directions() -> None:
    cfg = load_pairs_config(CONFIGS / "pairs" / "mmlu_pilot.yaml")

    runs = cfg.directions()

    assert len(runs) == 2 * len(cfg.pairs)
    assert ("bio_med:a2b", "college_biology", "college_medicine") in runs
    assert ("bio_med:b2a", "college_medicine", "college_biology") in runs


def test_ladder_pairs_share_one_target_and_run_source_to_target_only() -> None:
    cfg = load_pairs_config(CONFIGS / "pairs" / "mmlu_ladder.yaml")

    runs = cfg.directions()

    assert len(runs) == len(cfg.pairs) > 1
    assert {target for _, _, target in runs} == {"college_biology"}
    sources = [source for _, source, _ in runs]
    assert len(set(sources)) == len(sources)
    assert "college_biology" not in sources
    assert all(run.endswith(":a2b") for run, _, _ in runs)


def test_ladder_labels_run_from_same_field_to_far() -> None:
    cfg = load_pairs_config(CONFIGS / "pairs" / "mmlu_ladder.yaml")

    labels = [pair.expected_distance for pair in cfg.pairs]

    assert labels[0] == "same_field"
    assert labels[-1] == "far"
    assert labels == sorted(labels, key=VALID_DISTANCES.index)


def test_ladder_experiment_asks_for_no_more_questions_than_the_target_has() -> None:
    cfg = load_experiment_config(CONFIGS / "experiments" / "ladder_pilot.yaml")

    # college_biology has 144 test questions (checked against the dataset on 2026-11-09).
    assert cfg.num_questions_per_target <= 144
    assert cfg.pairs_config == Path("configs/pairs/mmlu_ladder.yaml")


@pytest.mark.parametrize("name", EXPERIMENT_NAMES)
def test_shipped_experiment_configs_load(name: str) -> None:
    cfg = load_experiment_config(CONFIGS / "experiments" / f"{name}.yaml")

    assert cfg.name == name
    assert PromptCondition.ZERO_SHOT in cfg.conditions
    assert (REPO_ROOT / cfg.model_config).is_file()
    assert (REPO_ROOT / cfg.pairs_config).is_file()


def test_experiment_options_are_read_only() -> None:
    cfg = load_experiment_config(CONFIGS / "experiments" / "pilot.yaml")

    with pytest.raises(TypeError):
        cfg.options["record_routing"] = False  # type: ignore[index]


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "nope.yaml")


def test_missing_required_key_raises_config_error(tmp_path: Path) -> None:
    data = _model_dict()
    del data["dtype"]

    with pytest.raises(ConfigError, match="dtype"):
        load_model_config(_write(tmp_path, data))


def test_invalid_dtype_raises_config_error(tmp_path: Path) -> None:
    data = _model_dict()
    data["dtype"] = "int3"

    with pytest.raises(ConfigError, match="dtype"):
        load_model_config(_write(tmp_path, data))


def test_router_pattern_without_layer_placeholder_raises(tmp_path: Path) -> None:
    data = _model_dict()
    data["architecture"]["router_module_pattern"] = "layers.gate"

    with pytest.raises(ConfigError, match="layer"):
        load_model_config(_write(tmp_path, data))


def test_bool_is_not_accepted_as_int(tmp_path: Path) -> None:
    data = _model_dict()
    data["architecture"]["top_k"] = True

    with pytest.raises(ConfigError, match="top_k"):
        load_model_config(_write(tmp_path, data))


def test_unknown_condition_raises_config_error(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "experiments" / "pilot.yaml").read_text())
    data["conditions"] = ["zero_shot", "bogus"]

    with pytest.raises(ConfigError, match="conditions"):
        load_experiment_config(_write(tmp_path, data))


def test_duplicate_conditions_raise_config_error(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "experiments" / "pilot.yaml").read_text())
    data["conditions"] = ["zero_shot", "cross_domain", "cross_domain"]

    with pytest.raises(ConfigError, match="duplicate"):
        load_experiment_config(_write(tmp_path, data))


def test_duplicate_pair_names_raise_config_error(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["pairs"].append(dict(data["pairs"][0]))

    with pytest.raises(ConfigError, match="unique"):
        load_pairs_config(_write(tmp_path, data))


def test_parse_config_arg_returns_path() -> None:
    path = parse_config_arg("desc", ["--config", "configs/models/olmoe.yaml"])

    assert path == Path("configs/models/olmoe.yaml")


def test_shot_and_eval_split_must_differ(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["eval_split"] = data["shot_split"]

    with pytest.raises(ConfigError, match="split"):
        load_pairs_config(_write(tmp_path, data))


@pytest.mark.parametrize("domain_key", ["domain_a", "domain_b"])
def test_irrelevant_subjects_must_not_overlap_pair_domains(tmp_path: Path, domain_key) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["irrelevant_pool"]["subjects"].append(data["pairs"][0][domain_key])

    with pytest.raises(ConfigError, match="irrelevant"):
        load_pairs_config(_write(tmp_path, data))


def test_irrelevant_pool_split_defaults_to_test_and_can_be_overridden(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    del data["irrelevant_pool"]["split"]
    assert load_pairs_config(_write(tmp_path, data)).irrelevant_split == "test"

    data["irrelevant_pool"]["split"] = "validation"
    assert load_pairs_config(_write(tmp_path, data, "other.yaml")).irrelevant_split == "validation"


def test_shipped_pilot_pairs_use_the_test_split_for_the_irrelevant_pool() -> None:
    assert load_pairs_config(CONFIGS / "pairs" / "mmlu_pilot.yaml").irrelevant_split == "test"


# --- shot counts, replicates and pooled shot splits ---------------------------------------


def _experiment_data() -> dict:
    return yaml.safe_load((CONFIGS / "experiments" / "pilot.yaml").read_text())


def test_shot_counts_default_to_num_shots_and_one_replicate(tmp_path: Path) -> None:
    data = _experiment_data()
    data.pop("shot_counts", None)
    data.pop("num_replicates", None)

    cfg = load_experiment_config(_write(tmp_path, data))

    assert cfg.shot_counts == (data["num_shots"],)
    assert cfg.num_replicates == 1


def test_shot_counts_and_replicates_are_read(tmp_path: Path) -> None:
    data = {**_experiment_data(), "num_shots": 5, "shot_counts": [1, 2, 3], "num_replicates": 4}

    cfg = load_experiment_config(_write(tmp_path, data))

    assert cfg.shot_counts == (1, 2, 3)
    assert cfg.num_replicates == 4


@pytest.mark.parametrize(
    "counts",
    [[], [0, 1], [2, 1], [1, 1], [1, 6], [-1], ["1"], [True], "3"],
)
def test_bad_shot_counts_raise_config_error(tmp_path: Path, counts) -> None:
    data = {**_experiment_data(), "num_shots": 5, "shot_counts": counts}

    with pytest.raises(ConfigError, match="shot_counts"):
        load_experiment_config(_write(tmp_path, data))


@pytest.mark.parametrize("replicates", [0, -1, True, "3"])
def test_bad_num_replicates_raise_config_error(tmp_path: Path, replicates) -> None:
    data = {**_experiment_data(), "num_replicates": replicates}

    with pytest.raises(ConfigError, match="num_replicates"):
        load_experiment_config(_write(tmp_path, data))


def test_shot_splits_default_to_the_shot_split_and_accept_extras(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data.pop("extra_shot_splits", None)
    assert load_pairs_config(_write(tmp_path, data)).shot_splits == (data["shot_split"],)

    data["extra_shot_splits"] = ["validation"]
    cfg = load_pairs_config(_write(tmp_path, data, "other.yaml"))
    assert cfg.shot_splits == (data["shot_split"], "validation")


def test_extra_shot_splits_must_not_include_the_eval_split(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["extra_shot_splits"] = [data["eval_split"]]

    with pytest.raises(ConfigError, match="eval_split"):
        load_pairs_config(_write(tmp_path, data))


@pytest.mark.parametrize("extra", ["validation", [1], ["dev"], ["validation", "validation"]])
def test_bad_extra_shot_splits_raise_config_error(tmp_path: Path, extra) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["extra_shot_splits"] = extra

    with pytest.raises(ConfigError, match="extra_shot_splits"):
        load_pairs_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    "name", EXPERIMENT_NAMES + ["ladder_force_routing", "ladder_reverse_check"]
)
def test_shipped_experiments_run_k_1_to_5_over_several_replicates(name: str) -> None:
    cfg = load_experiment_config(CONFIGS / "experiments" / f"{name}.yaml")

    assert cfg.shot_counts == (1, 2, 3, 4, 5)
    assert cfg.num_replicates >= 2


@pytest.mark.parametrize("name", ["mmlu_pilot", "mmlu_ladder"])
def test_shipped_pairs_pool_dev_and_validation_for_shots(name: str) -> None:
    cfg = load_pairs_config(CONFIGS / "pairs" / f"{name}.yaml")

    assert cfg.shot_splits == ("dev", "validation")
    assert cfg.eval_split == "test"


def test_dataset_revision_is_read_from_the_shipped_pairs_configs() -> None:
    for name in ("mmlu_pilot.yaml", "mmlu_ladder.yaml"):
        cfg = load_pairs_config(CONFIGS / "pairs" / name)
        assert cfg.dataset_revision is not None and len(cfg.dataset_revision) == 40


def test_dataset_revision_defaults_to_none(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    del data["dataset_revision"]

    assert load_pairs_config(_write(tmp_path, data)).dataset_revision is None


@pytest.mark.parametrize("bad", [123, "", "  "])
def test_bad_dataset_revision_raises_config_error(tmp_path: Path, bad) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["dataset_revision"] = bad

    with pytest.raises(ConfigError, match="dataset_revision"):
        load_pairs_config(_write(tmp_path, data))
