"""Phase 2 pilot: accuracy per condition and routing shift (Jaccard) on target tokens.

Resumable: results are appended per question to ``<output_dir>/<run>.jsonl``, so
re-running the same command after a crash or preemption continues where it stopped.

Usage: uv run python scripts/02_run_pilot.py --config configs/experiments/pilot.yaml
"""

from rr.analysis import format_pilot_summary, load_results
from rr.config import load_experiment_config, load_model_config, parse_config_arg
from rr.models import inspect_moe_structure, load_model, model_provenance, verify_structure
from rr.pilot import run_prompt_file


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    prompt_files = sorted(cfg.prompts_dir.glob("*.jsonl"))
    if not prompt_files:
        raise FileNotFoundError(
            f"no prompt files in {cfg.prompts_dir}; run scripts/01_build_prompts.py first"
        )
    model_cfg = load_model_config(cfg.model_config)
    model, tokenizer = load_model(
        model_cfg.name, model_cfg.dtype, model_cfg.device, model_cfg.revision
    )
    structure = inspect_moe_structure(model)
    verify_structure(structure, model_cfg.architecture)

    for path in prompt_files:
        evaluated = run_prompt_file(
            model,
            tokenizer,
            structure,
            path,
            cfg.output_dir / path.name,
            model_provenance=model_provenance(model_cfg),
            show_progress=True,
        )
        print(f"{path.stem}: evaluated {evaluated} questions")
    print(format_pilot_summary(load_results(cfg.output_dir)))


if __name__ == "__main__":
    main()
