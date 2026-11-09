"""Phase 2 pilot: score every condition and measure routing shift vs. zero-shot.

Role in the project: for each target question, run its zero-shot, cross-domain,
same-domain and irrelevant prompts with the router recorded, then compare the
top-k routing on the *target-question tokens* of each prompt against the
zero-shot prompt's. This answers (a) does cross-domain few-shot beat zero-shot
and (b) do relevant examples shift routing more than irrelevant ones.

Methodological notes:

* Routing is compared only on aligned target-question tokens
  (:mod:`rr.routing.align`), never on the in-context examples.
* Runs are resumable per question (:func:`run_prompt_file`); scoring is
  deterministic so re-evaluating a half-finished question is safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rr.prompts import BuiltPrompt, PromptCondition
from rr.results import NO_INTERVENTION, ResultRecord
from rr.routing.align import align_target_tokens
from rr.routing.metrics import per_layer_jaccard
from rr.routing.record import RoutingRecord, RoutingRecorder
from rr.runner import PromptGroup, run_resumable
from rr.scoring import MCScore, score_mc

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from rr.models import MoEStructure


def evaluate_group(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    group: PromptGroup,
    *,
    run: str,
    question_index: int,
) -> list[ResultRecord]:
    """Score every prompt of one target question and compare routing to zero-shot.

    Args:
        model: Loaded MoE model.
        tokenizer: Matching tokenizer (fast: offset mappings are required).
        structure: Discovered MoE structure (router names and top-k).
        group: Prompts of one target question, keyed by condition.
        run: Run name stored on each record.
        question_index: Index of the question within the run.

    Returns:
        One record per condition, in ``group`` order.

    Raises:
        ValueError: If the group has no zero-shot prompt to compare against.
    """
    baseline = group.get(PromptCondition.ZERO_SHOT)
    if baseline is None:
        raise ValueError("the group needs a zero_shot prompt as the routing baseline")
    outcomes = {
        condition: score_with_routing(model, tokenizer, structure, prompt)
        for condition, prompt in group.items()
    }
    baseline_routing = outcomes[PromptCondition.ZERO_SHOT][1]
    records: list[ResultRecord] = []
    for condition, prompt in group.items():
        score, routing = outcomes[condition]
        shift = None
        if condition is not PromptCondition.ZERO_SHOT:
            shift = _target_jaccard(tokenizer, baseline, baseline_routing, prompt, routing)
        records.append(
            ResultRecord(
                run=run,
                question_index=question_index,
                subject=prompt.target.subject,
                condition=condition,
                option_logprobs=score.option_logprobs,
                prediction=score.prediction,
                is_correct=score.is_correct,
                correct_logprob=score.correct_logprob,
                num_examples=prompt.num_examples,
                example_tokens=prompt.example_tokens,
                jaccard_per_layer=shift,
            )
        )
    return records


def run_prompt_file(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    prompts_path: Path,
    output_path: Path,
    *,
    model_provenance: Mapping[str, Any] | None = None,
    show_progress: bool = False,
) -> int:
    """Evaluate every question in a prompt file, resuming from any existing results.

    ``model_provenance`` (see :func:`rr.models.model_provenance`) is recorded in the
    run manifest, so a resume under a different model is refused. See
    :func:`rr.runner.run_resumable` for the resume semantics and errors.

    Returns:
        The number of questions evaluated in this call (0 if already complete).
    """
    run = prompts_path.stem

    def evaluate(group: PromptGroup, question_index: int) -> list[ResultRecord]:
        return evaluate_group(
            model, tokenizer, structure, group, run=run, question_index=question_index
        )

    def expected_keys(group: PromptGroup) -> set[tuple[PromptCondition, str]]:
        return {(condition, NO_INTERVENTION) for condition in group}

    return run_resumable(
        prompts_path,
        output_path,
        evaluate,
        expected_keys,
        settings={"kind": "pilot", "model": model_provenance},
        show_progress=show_progress,
    )


def score_with_routing(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    prompt: BuiltPrompt,
) -> tuple[MCScore, RoutingRecord]:
    with RoutingRecorder(model, structure.router_module_names, structure.top_k) as recorder:
        score = score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
    return score, recorder.get_record()


def _target_jaccard(
    tokenizer: PreTrainedTokenizerBase,
    baseline: BuiltPrompt,
    baseline_routing: RoutingRecord,
    prompt: BuiltPrompt,
    routing: RoutingRecord,
) -> tuple[float, ...]:
    alignment = align_target_tokens(tokenizer, baseline, prompt)
    return tuple(
        per_layer_jaccard(
            baseline_routing.select_positions(alignment.positions_a).topk_indices,
            routing.select_positions(alignment.positions_b).topk_indices,
        ).tolist()
    )
