"""Phase 1: build the four prompt conditions for every domain pair and direction.

Usage: uv run python scripts/01_build_prompts.py --config configs/experiments/pilot.yaml
"""

from transformers import AutoTokenizer

from rr.config import (
    load_experiment_config,
    load_model_config,
    load_pairs_config,
    parse_config_arg,
)
from rr.data import load_mmlu_subject, sample_questions
from rr.prompts import (
    build_condition_prompts,
    count_example_tokens,
    save_prompts,
    select_irrelevant_examples,
    verify_target_tokens,
)


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    pairs = load_pairs_config(cfg.pairs_config)
    model_cfg = load_model_config(cfg.model_config)
    # Only the tokenizer is needed to build and length-match prompts, not the weights.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, revision=model_cfg.revision)

    irrelevant_pool = [
        q for s in pairs.irrelevant_subjects for q in load_mmlu_subject(s, pairs.irrelevant_split)
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
        # The control depends only on the source shots, so select it once per direction.
        irrelevant_shots = select_irrelevant_examples(
            irrelevant_pool,
            len(source_shots),
            count_example_tokens(source_shots, tokenizer),
            tokenizer,
            cfg.seed,
        )
        prompts = []
        for question in eval_qs:
            built = list(
                build_condition_prompts(
                    question,
                    source_examples=source_shots,
                    same_domain_examples=target_shots,
                    irrelevant_examples=irrelevant_shots,
                    conditions=cfg.conditions,
                    tokenizer=tokenizer,
                ).values()
            )
            verify_target_tokens(built, tokenizer)
            prompts.extend(built)
        save_prompts(prompts, cfg.prompts_dir / f"{run_name.replace(':', '_')}.jsonl")


if __name__ == "__main__":
    main()
