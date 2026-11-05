"""Phase 1: build the four prompt conditions for every domain pair and direction.

Usage: uv run python scripts/01_build_prompts.py --config configs/experiments/pilot.yaml
"""

from rr.config import (
    load_experiment_config,
    load_model_config,
    load_pairs_config,
    parse_config_arg,
)
from rr.data import load_mmlu_subject, sample_questions
from rr.models import load_model
from rr.prompts import build_condition_prompts, save_prompts


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    pairs = load_pairs_config(cfg.pairs_config)
    model_cfg = load_model_config(cfg.model_config)
    # Only the tokenizer is needed here; TODO(Phase 1): load the tokenizer alone
    # (AutoTokenizer) instead of the full model.
    _, tokenizer = load_model(model_cfg.name, model_cfg.dtype, model_cfg.device)

    irrelevant_pool = [
        q for s in pairs.irrelevant_subjects for q in load_mmlu_subject(s, pairs.shot_split)
    ]
    for run_name, source, target in pairs.directions():
        source_shots = sample_questions(
            load_mmlu_subject(source, pairs.shot_split), cfg.num_shots, cfg.seed
        )
        target_shots = sample_questions(
            load_mmlu_subject(target, pairs.shot_split), cfg.num_shots, cfg.seed
        )
        eval_qs = sample_questions(
            load_mmlu_subject(target, pairs.eval_split),
            cfg.num_questions_per_target,
            cfg.seed,
            exclude=target_shots,
        )
        prompts = [
            p
            for q in eval_qs
            for p in build_condition_prompts(
                q,
                source_examples=source_shots,
                same_domain_examples=target_shots,
                irrelevant_pool=irrelevant_pool,
                conditions=cfg.conditions,
                tokenizer=tokenizer,
                seed=cfg.seed,
            ).values()
        ]
        save_prompts(prompts, cfg.prompts_dir / f"{run_name.replace(':', '_')}.jsonl")


if __name__ == "__main__":
    main()
