"""Load and validate YAML configs.

Role in the project: every script takes ``--config <path>``; this module turns
that YAML into frozen, validated dataclasses so errors surface at load time
rather than mid-experiment. Used by all phases. This is the only module that is
fully implemented in the scaffolding.

Three config kinds exist:

* model configs (``configs/models/*.yaml``) -> :class:`ModelConfig`
* domain-pair configs (``configs/pairs/*.yaml``) -> :class:`PairsConfig`
* experiment configs (``configs/experiments/*.yaml``) -> :class:`ExperimentConfig`
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from rr.prompts import PromptCondition

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
# Ordered closest to farthest; the order is used to check labels against measured distance.
VALID_DISTANCES = ("same_field", "near", "medium", "far")


# The irrelevant control draws from subjects that are never eval targets, so it can use
# the large test split without overlapping any evaluated question.
DEFAULT_IRRELEVANT_SPLIT = "test"


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class ArchitectureConfig:
    """Expected MoE architecture. UNVERIFIED until checked against the loaded model."""

    num_layers: int
    num_experts: int
    top_k: int
    router_module_pattern: str

    def router_module_names(self) -> tuple[str, ...]:
        """Expand ``router_module_pattern`` over all layers."""
        return tuple(self.router_module_pattern.format(layer=i) for i in range(self.num_layers))


@dataclass(frozen=True)
class ModelConfig:
    """Which model to load and how."""

    name: str
    revision: str | None
    dtype: str
    device: str
    architecture: ArchitectureConfig


@dataclass(frozen=True)
class DomainPair:
    """Two MMLU subjects; run in both directions when ``run_both_directions``."""

    name: str
    domain_a: str
    domain_b: str
    expected_distance: str


@dataclass(frozen=True)
class PairsConfig:
    """Dataset, splits, and domain pairs."""

    dataset: str
    shot_split: str
    eval_split: str
    run_both_directions: bool
    pairs: tuple[DomainPair, ...]
    irrelevant_subjects: tuple[str, ...]
    irrelevant_split: str = DEFAULT_IRRELEVANT_SPLIT

    def directions(self) -> tuple[tuple[str, str, str], ...]:
        """Return ``(pair_name, source_subject, target_subject)`` for every run.

        With ``run_both_directions`` each pair yields two entries (a->b, b->a).
        """
        runs: list[tuple[str, str, str]] = []
        for pair in self.pairs:
            runs.append((f"{pair.name}:a2b", pair.domain_a, pair.domain_b))
            if self.run_both_directions:
                runs.append((f"{pair.name}:b2a", pair.domain_b, pair.domain_a))
        return tuple(runs)


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by all experiments plus an ``options`` bag for per-experiment keys."""

    name: str
    model_config: Path
    pairs_config: Path
    seed: int
    num_shots: int
    num_questions_per_target: int
    conditions: tuple[PromptCondition, ...]
    prompts_dir: Path
    output_dir: Path
    options: Mapping[str, Any]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _get(data: Mapping[str, Any], key: str, expected: type | tuple[type, ...], where: str) -> Any:
    """Fetch a required key and check its type (bools are not accepted as ints)."""
    if key not in data:
        raise ConfigError(f"{where}: missing required key '{key}'")
    value = data[key]
    if isinstance(value, bool) and bool not in _as_tuple(expected):
        raise ConfigError(f"{where}: '{key}' must be {expected}, got bool")
    if not isinstance(value, expected):
        raise ConfigError(f"{where}: '{key}' must be {expected}, got {type(value).__name__}")
    return value


def _as_tuple(expected: type | tuple[type, ...]) -> tuple[type, ...]:
    return expected if isinstance(expected, tuple) else (expected,)


def _one_of(value: str, allowed: tuple[str, ...], key: str, where: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{where}: '{key}' must be one of {allowed}, got '{value}'")
    return value


def _positive(value: int, key: str, where: str) -> int:
    if value <= 0:
        raise ConfigError(f"{where}: '{key}' must be positive, got {value}")
    return value


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a model config (e.g. ``configs/models/olmoe.yaml``).

    Args:
        path: Path to the YAML file.

    Returns:
        A frozen :class:`ModelConfig`.

    Raises:
        ConfigError: If the file is missing, malformed, or has invalid values.
    """
    where = str(path)
    data = _read_yaml(path)
    arch = _get(data, "architecture", dict, where)
    pattern = _get(arch, "router_module_pattern", str, where)
    if "{layer}" not in pattern:
        raise ConfigError(f"{where}: 'router_module_pattern' must contain '{{layer}}'")
    revision = data.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ConfigError(f"{where}: 'revision' must be a string or null")
    return ModelConfig(
        name=_get(data, "name", str, where),
        revision=revision,
        dtype=_one_of(_get(data, "dtype", str, where), VALID_DTYPES, "dtype", where),
        device=_one_of(_get(data, "device", str, where), VALID_DEVICES, "device", where),
        architecture=ArchitectureConfig(
            num_layers=_positive(_get(arch, "num_layers", int, where), "num_layers", where),
            num_experts=_positive(_get(arch, "num_experts", int, where), "num_experts", where),
            top_k=_positive(_get(arch, "top_k", int, where), "top_k", where),
            router_module_pattern=pattern,
        ),
    )


def load_pairs_config(path: str | Path) -> PairsConfig:
    """Load and validate a domain-pairs config (e.g. ``configs/pairs/mmlu_pilot.yaml``).

    Raises:
        ConfigError: If the file is missing, malformed, or has invalid values.
    """
    where = str(path)
    data = _read_yaml(path)
    raw_pairs = _get(data, "pairs", list, where)
    if not raw_pairs:
        raise ConfigError(f"{where}: 'pairs' must not be empty")
    pairs: list[DomainPair] = []
    for i, raw in enumerate(raw_pairs):
        pair_where = f"{where} pairs[{i}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{pair_where}: must be a mapping")
        distance = _get(raw, "expected_distance", str, pair_where)
        pairs.append(
            DomainPair(
                name=_get(raw, "name", str, pair_where),
                domain_a=_get(raw, "domain_a", str, pair_where),
                domain_b=_get(raw, "domain_b", str, pair_where),
                expected_distance=_one_of(
                    distance, VALID_DISTANCES, "expected_distance", pair_where
                ),
            )
        )
    names = [p.name for p in pairs]
    if len(set(names)) != len(names):
        raise ConfigError(f"{where}: pair names must be unique, got {names}")
    pool_data = data.get("irrelevant_pool", {})
    irrelevant = pool_data.get("subjects", [])
    irrelevant_split = pool_data.get("split", DEFAULT_IRRELEVANT_SPLIT)
    if not isinstance(irrelevant_split, str):
        raise ConfigError(f"{where}: 'irrelevant_pool.split' must be a string")
    if not isinstance(irrelevant, list) or not all(isinstance(s, str) for s in irrelevant):
        raise ConfigError(f"{where}: 'irrelevant_pool.subjects' must be a list of strings")
    pair_domains = {d for p in pairs for d in (p.domain_a, p.domain_b)}
    overlap = sorted(pair_domains.intersection(irrelevant))
    if overlap:
        raise ConfigError(
            f"{where}: irrelevant_pool subjects must not be pair domains, got {overlap}"
        )
    shot_split = _get(data, "shot_split", str, where)
    eval_split = _get(data, "eval_split", str, where)
    if shot_split == eval_split:
        raise ConfigError(
            f"{where}: shot_split and eval_split must differ (both are {shot_split!r}) "
            "so shot questions can never be evaluated"
        )
    return PairsConfig(
        dataset=_get(data, "dataset", str, where),
        shot_split=shot_split,
        eval_split=eval_split,
        run_both_directions=_get(data, "run_both_directions", bool, where),
        pairs=tuple(pairs),
        irrelevant_subjects=tuple(irrelevant),
        irrelevant_split=irrelevant_split,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config (``configs/experiments/*.yaml``).

    Referenced model/pairs config paths are stored as given; load them with
    :func:`load_model_config` and :func:`load_pairs_config`.

    Raises:
        ConfigError: If the file is missing, malformed, or has invalid values.
    """
    where = str(path)
    data = _read_yaml(path)
    num_shots = _get(data, "num_shots", int, where)
    if num_shots < 0:
        raise ConfigError(f"{where}: 'num_shots' must be >= 0, got {num_shots}")
    valid_conditions = {c.value for c in PromptCondition}
    raw_conditions = _get(data, "conditions", list, where)
    unknown = [c for c in raw_conditions if c not in valid_conditions]
    if unknown or not raw_conditions:
        raise ConfigError(
            f"{where}: 'conditions' must be a non-empty list drawn from "
            f"{sorted(valid_conditions)}; got {raw_conditions}"
        )
    if len(set(raw_conditions)) != len(raw_conditions):
        raise ConfigError(f"{where}: 'conditions' has duplicate entries: {raw_conditions}")
    options = data.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"{where}: 'options' must be a mapping")
    return ExperimentConfig(
        name=_get(data, "name", str, where),
        model_config=Path(_get(data, "model_config", str, where)),
        pairs_config=Path(_get(data, "pairs_config", str, where)),
        seed=_get(data, "seed", int, where),
        num_shots=num_shots,
        num_questions_per_target=_positive(
            _get(data, "num_questions_per_target", int, where), "num_questions_per_target", where
        ),
        conditions=tuple(PromptCondition(c) for c in raw_conditions),
        prompts_dir=Path(_get(data, "prompts_dir", str, where)),
        output_dir=Path(_get(data, "output_dir", str, where)),
        options=MappingProxyType(dict(options)),
    )


def parse_config_arg(description: str, argv: list[str] | None = None) -> Path:
    """Parse the ``--config`` argument shared by every script.

    Args:
        description: Script description shown in ``--help``.
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The path passed via ``--config``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file")
    return parser.parse_args(argv).config
