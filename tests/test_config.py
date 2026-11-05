"""Tests for rr.config: shipped configs load, and bad configs fail loudly."""

from pathlib import Path

import pytest
import yaml

from rr.config import (
    ConfigError,
    load_experiment_config,
    load_model_config,
    load_pairs_config,
    parse_config_arg,
)
from rr.prompts import PromptCondition

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
EXPERIMENT_NAMES = ["pilot", "force_routing", "reverse_check"]


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


def test_duplicate_pair_names_raise_config_error(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "pairs" / "mmlu_pilot.yaml").read_text())
    data["pairs"].append(dict(data["pairs"][0]))

    with pytest.raises(ConfigError, match="unique"):
        load_pairs_config(_write(tmp_path, data))


def test_parse_config_arg_returns_path() -> None:
    path = parse_config_arg("desc", ["--config", "configs/models/olmoe.yaml"])

    assert path == Path("configs/models/olmoe.yaml")
