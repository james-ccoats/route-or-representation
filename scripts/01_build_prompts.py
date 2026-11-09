"""Phase 1: build the prompt conditions for every domain direction, replicate and shot count.

Each (direction, replicate, k) is one prompt file named like ``bio_med_a2b_r1_k3.jsonl``.
Example sets are drawn from the pooled shot splits; see D14 in docs/design_decisions.md.

Usage: uv run python scripts/01_build_prompts.py --config configs/experiments/pilot.yaml
"""

from transformers import AutoTokenizer

from rr.config import (
    load_experiment_config,
    load_model_config,
    load_pairs_config,
    parse_config_arg,
)
from rr.data import load_mmlu_subject, load_shot_pool
from rr.prompt_build import build_direction_prompts, without_overlap, write_runs


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    pairs = load_pairs_config(cfg.pairs_config)
    model_cfg = load_model_config(cfg.model_config)
    # Only the tokenizer is needed to build and length-match prompts, not the weights.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, revision=model_cfg.revision)

    irrelevant_pool = [
        q for s in pairs.irrelevant_subjects for q in load_mmlu_subject(s, pairs.irrelevant_split)
    ]
    runs = {}
    for direction, source, target in pairs.directions():
        eval_pool = load_mmlu_subject(target, pairs.eval_split)
        full_target_pool = load_shot_pool(target, pairs.shot_splits)
        target_pool = without_overlap(full_target_pool, eval_pool)
        if len(target_pool) < len(full_target_pool):
            print(
                f"{direction}: dropped {len(full_target_pool) - len(target_pool)} of "
                f"{len(full_target_pool)} {target} shot questions that also appear in "
                f"{pairs.eval_split}"
            )
        built = build_direction_prompts(
            direction=direction,
            source_pool=load_shot_pool(source, pairs.shot_splits),
            target_pool=target_pool,
            eval_pool=eval_pool,
            irrelevant_pool=irrelevant_pool,
            tokenizer=tokenizer,
            conditions=cfg.conditions,
            num_shots=cfg.num_shots,
            shot_counts=cfg.shot_counts,
            num_replicates=cfg.num_replicates,
            num_questions=cfg.num_questions_per_target,
            seed=cfg.seed,
        )
        runs.update(built)
        print(f"{direction}: {len(built)} runs")
    write_runs(runs, cfg.prompts_dir)
    print(f"wrote {len(runs)} prompt files to {cfg.prompts_dir}")


if __name__ == "__main__":
    main()
