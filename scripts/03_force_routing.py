"""Phase 3: few-shot prompt with zero-shot routing forced on target-question tokens.

Usage: uv run python scripts/03_force_routing.py --config configs/experiments/force_routing.yaml
"""

from rr.analysis import fraction_of_gain_retained
from rr.config import load_experiment_config, load_model_config, parse_config_arg
from rr.models import inspect_moe_structure, load_model
from rr.prompts import load_prompts
from rr.routing.align import align_target_tokens
from rr.routing.force import RoutingForcer, random_spec, spec_from_record
from rr.routing.record import RoutingRecorder
from rr.scoring import score_mc


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    model_cfg = load_model_config(cfg.model_config)
    model, tokenizer = load_model(
        model_cfg.name, model_cfg.dtype, model_cfg.device, model_cfg.revision
    )
    structure = inspect_moe_structure(model)
    names = structure.router_module_names

    for path in sorted(cfg.prompts_dir.glob("*.jsonl")):
        prompts = load_prompts(path)
        # TODO(Phase 3): for each (zero-shot, few-shot) prompt pair of a target:
        #   1. record zero-shot routing; align target tokens to the few-shot prompt
        #   2. spec = spec_from_record(record, zs_positions, fs_positions, ...)
        #   3. score the few-shot prompt inside RoutingForcer(model, names, spec)
        #   4. sanity: identity forcing on zero-shot (no change); random_spec (hurts)
        #   5. aggregate scores, call fraction_of_gain_retained, write to cfg.output_dir
        _ = (prompts, names, RoutingRecorder, RoutingForcer, random_spec, spec_from_record)
        _ = (align_target_tokens, score_mc, fraction_of_gain_retained, tokenizer)
        raise NotImplementedError("TODO(Phase 3): forced-routing experiment loop")


if __name__ == "__main__":
    main()
