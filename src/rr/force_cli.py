"""Shared driver for ``03_force_routing.py`` and ``04_reverse_check.py``.

Role in the project: the two scripts differ only in which forcing plans they run,
so the config handling, model loading, per-file runs and final report live here
once. Scripts stay one-liners; this module is importable and tested.

Fail-fast order: config and option validation happen *before* the (slow, large)
model is loaded, and the discovered MoE structure is checked against the config's
unverified architecture values before any prompt is run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rr.analysis import load_results
from rr.config import (
    ExperimentConfig,
    load_experiment_config,
    load_model_config,
    parse_config_arg,
)
from rr.force_analysis import format_force_summary
from rr.force_experiment import (
    ForcingPlan,
    parse_bool_option,
    parse_forcing_options,
    plans_for_force_routing,
    plans_for_reverse_check,
    run_forcing_file,
)
from rr.models import inspect_moe_structure, load_model, model_provenance, verify_structure

PlanFactory = Callable[[ExperimentConfig], tuple[ForcingPlan, ...]]


def force_routing_plans(cfg: ExperimentConfig) -> tuple[ForcingPlan, ...]:
    """Plans for ``03``: zero-shot routing forced into each few-shot prompt, plus controls."""
    return plans_for_force_routing(
        cfg.conditions,
        identity_check=parse_bool_option(cfg.options, "sanity_identity_check", True),
        random_check=parse_bool_option(cfg.options, "sanity_random_check", True),
    )


def reverse_check_plans(cfg: ExperimentConfig) -> tuple[ForcingPlan, ...]:
    """Plans for ``04``: each few-shot routing forced into the zero-shot prompt."""
    return plans_for_reverse_check(cfg.conditions)


def run_forcing_script(
    description: str, make_plans: PlanFactory, argv: Sequence[str] | None = None
) -> None:
    """Run a forcing experiment end to end and print its summary.

    Args:
        description: The script's docstring (used for ``--help``).
        make_plans: Chooses the forcing plans from the experiment config.
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Raises:
        FileNotFoundError: If the prompts directory holds no prompt files.
        ValueError: On invalid options, or a config that misdescribes the model.
    """
    cfg = load_experiment_config(parse_config_arg(description, list(argv) if argv else None))
    prompt_files = sorted(cfg.prompts_dir.glob("*.jsonl"))
    if not prompt_files:
        raise FileNotFoundError(
            f"no prompt files in {cfg.prompts_dir}; run scripts/01_build_prompts.py first"
        )
    options = parse_forcing_options(cfg.options, seed=cfg.seed)
    plans = make_plans(cfg)

    model_cfg = load_model_config(cfg.model_config)
    model, tokenizer = load_model(
        model_cfg.name, model_cfg.dtype, model_cfg.device, model_cfg.revision
    )
    structure = inspect_moe_structure(model)
    verify_structure(structure, model_cfg.architecture)

    for path in prompt_files:
        evaluated = run_forcing_file(
            model,
            tokenizer,
            structure,
            path,
            cfg.output_dir / path.name,
            plans,
            options,
            model_provenance=model_provenance(model_cfg),
            show_progress=True,
        )
        print(f"{path.stem}: evaluated {evaluated} questions")
    print(format_force_summary(load_results(cfg.output_dir), seed=cfg.seed))
