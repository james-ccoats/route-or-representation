"""Phase 0 smoke test: load the model, verify MoE structure, score one question, record routing.

Usage: uv run python scripts/00_smoke_test.py --config configs/models/olmoe.yaml
"""

from rr.config import load_model_config, parse_config_arg
from rr.data import MCQuestion
from rr.models import inspect_moe_structure, load_model, verify_structure
from rr.routing.record import RoutingRecorder
from rr.scoring import format_mc_prompt, score_mc

TOY_QUESTION = MCQuestion(
    subject="toy",
    question="What is 2 + 2?",
    choices=("3", "4", "5", "6"),
    answer_index=1,
)


def main() -> None:
    cfg = load_model_config(parse_config_arg(__doc__ or ""))
    model, tokenizer = load_model(cfg.name, cfg.dtype, cfg.device, cfg.revision)

    structure = inspect_moe_structure(model)
    print(structure)
    verify_structure(structure, cfg.architecture)

    prompt = format_mc_prompt(TOY_QUESTION)
    with RoutingRecorder(model, structure.router_module_names, structure.top_k) as recorder:
        score = score_mc(model, tokenizer, prompt, TOY_QUESTION.answer_index)
    print(score)
    print(recorder.get_record().topk_indices.shape)


if __name__ == "__main__":
    main()
