"""Phase 3 reverse check: zero-shot prompt with few-shot routing forced in.

Usage: uv run python scripts/04_reverse_check.py --config configs/experiments/reverse_check.yaml
"""

from rr.analysis import fraction_of_gain_recovered
from rr.config import load_experiment_config, load_model_config, parse_config_arg
from rr.models import inspect_moe_structure, load_model
from rr.prompts import load_prompts
from rr.routing.align import align_target_tokens
from rr.routing.force import RoutingForcer, spec_from_record
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
        # TODO(Phase 3): mirror of 03_force_routing.py with roles swapped:
        #   record few-shot routing, force it onto the zero-shot prompt's target
        #   tokens, then call fraction_of_gain_recovered and write results.
        _ = (prompts, names, RoutingRecorder, RoutingForcer, spec_from_record)
        _ = (align_target_tokens, score_mc, fraction_of_gain_recovered, tokenizer)
        raise NotImplementedError("TODO(Phase 3): reverse-check experiment loop")


if __name__ == "__main__":
    main()
