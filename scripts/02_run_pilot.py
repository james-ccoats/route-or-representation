"""Phase 2 pilot: accuracy per condition and routing shift (Jaccard) on target tokens.

Usage: uv run python scripts/02_run_pilot.py --config configs/experiments/pilot.yaml
"""

from tqdm import tqdm

from rr.config import load_experiment_config, load_model_config, parse_config_arg
from rr.models import inspect_moe_structure, load_model
from rr.prompts import PromptCondition, load_prompts
from rr.routing.align import align_target_tokens
from rr.routing.metrics import per_layer_jaccard
from rr.routing.record import RoutingRecorder
from rr.scoring import score_mc


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    model_cfg = load_model_config(cfg.model_config)
    model, tokenizer = load_model(
        model_cfg.name, model_cfg.dtype, model_cfg.device, model_cfg.revision
    )
    structure = inspect_moe_structure(model)

    for path in sorted(cfg.prompts_dir.glob("*.jsonl")):
        prompts = load_prompts(path)
        zero_shot = {
            p.target.question: p for p in prompts if p.condition is PromptCondition.ZERO_SHOT
        }
        for prompt in tqdm(prompts, desc=path.stem):
            with RoutingRecorder(model, structure.router_module_names, structure.top_k) as rec:
                score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
            baseline = zero_shot[prompt.target.question]
            alignment = align_target_tokens(tokenizer, baseline, prompt)
            # TODO(Phase 2): record the zero-shot run too, restrict both records
            # to aligned positions, compute per_layer_jaccard, and write results
            # to cfg.output_dir.
            _ = (rec.get_record(), alignment, per_layer_jaccard)


if __name__ == "__main__":
    main()
