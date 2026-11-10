"""Phase 3 forcing experiments: run a prompt with another prompt's routing forced in.

Role in the project: ``03_force_routing.py`` runs each few-shot prompt with the
*zero-shot* routing forced on its target-question tokens (how much of the gain
survives -> "representation"); ``04_reverse_check.py`` runs the zero-shot prompt
with *few-shot* routing forced in (how much of the gain routing alone recovers
-> "route"). Both are the same operation with the roles swapped, described by a
:class:`ForcingPlan`: which prompt is run, and whose routing is forced onto it.

Every evaluated question also emits the plain ("none") score of each prompt
involved, so the gain fractions in :mod:`rr.analysis` can be computed from one
results directory. Two sanity controls are plans too: forcing a prompt's own
routing back onto itself (must change nothing) and forcing random experts
(must hurt).

Methodological notes:

* Forcing applies only on the aligned target-question tokens (design decision D2,
  option A); example tokens route naturally.
* ``force_weights`` selects between a full routing replay (the default: the recorded
  mixing weights are forced along with the experts) and forcing only expert
  *selection* (weights are the router's own probabilities over the forced experts);
  decision D1. The random control never forces weights.
* The random control seeds from ``random_seed + question_index``, so it is
  reproducible and resumable but differs between questions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rr.config import parse_bool_option
from rr.pilot import score_with_routing
from rr.prompts import BuiltPrompt, PromptCondition
from rr.results import NO_INTERVENTION, ResultRecord
from rr.routing.align import align_target_tokens, target_token_positions
from rr.routing.force import ForcingSpec, RoutingForcer, random_spec, spec_from_record
from rr.routing.record import RoutingRecord
from rr.runner import PromptGroup, run_resumable
from rr.scoring import MCScore, score_mc

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from rr.models import MoEStructure

RANDOM_INTERVENTION = "force_random_routing"
SUPPORTED_FORCE_SCOPE = "target_question_tokens"


@dataclass(frozen=True)
class ForcingPlan:
    """One forced run.

    Attributes:
        prompt_condition: Which prompt is run.
        source: Whose routing is forced onto its target tokens; ``None`` forces
            uniformly random experts (the "forcing hurts" control).
    """

    prompt_condition: PromptCondition
    source: PromptCondition | None

    @property
    def intervention(self) -> str:
        """Name stored on the result records, e.g. ``"force_zero_shot_routing"``."""
        if self.source is None:
            return RANDOM_INTERVENTION
        return f"force_{self.source.value}_routing"


@dataclass(frozen=True)
class ForcingOptions:
    """How forcing is applied.

    Attributes:
        force_weights: Also force the recorded mixing weights (full routing replay; the default).
        layers: Layers to force; ``None`` forces all, ``()`` forces nothing.
        random_seed: Base seed of the random-expert control.
    """

    force_weights: bool = True
    layers: tuple[int, ...] | None = None
    random_seed: int = 0


def _few_shot(conditions: Sequence[PromptCondition]) -> list[PromptCondition]:
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"duplicate conditions: {[c.value for c in conditions]}")
    few_shot = [c for c in conditions if c is not PromptCondition.ZERO_SHOT]
    if not few_shot:
        raise ValueError("forcing needs at least one few-shot condition besides zero_shot")
    return few_shot


def plans_for_force_routing(
    conditions: Sequence[PromptCondition], *, identity_check: bool, random_check: bool
) -> tuple[ForcingPlan, ...]:
    """Plans for ``03``: each few-shot prompt with zero-shot routing forced in.

    Args:
        conditions: The experiment's conditions (must include a few-shot one).
        identity_check: Add the control forcing zero-shot routing onto the zero-shot prompt.
        random_check: Add the control forcing random experts onto each few-shot prompt.
    """
    few_shot = _few_shot(conditions)
    zero = PromptCondition.ZERO_SHOT
    plans = [ForcingPlan(condition, zero) for condition in few_shot]
    if identity_check:
        plans.append(ForcingPlan(zero, zero))
    if random_check:
        plans.extend(ForcingPlan(condition, None) for condition in few_shot)
    return tuple(plans)


def plans_for_reverse_check(conditions: Sequence[PromptCondition]) -> tuple[ForcingPlan, ...]:
    """Plans for ``04``: the zero-shot prompt with each few-shot routing forced in."""
    return tuple(ForcingPlan(PromptCondition.ZERO_SHOT, c) for c in _few_shot(conditions))


def _needed_conditions(plans: Sequence[ForcingPlan]) -> set[PromptCondition]:
    needed = {PromptCondition.ZERO_SHOT}
    for plan in plans:
        needed.add(plan.prompt_condition)
        if plan.source is not None:
            needed.add(plan.source)
    return needed


def expected_keys(
    group: PromptGroup, plans: Sequence[ForcingPlan]
) -> set[tuple[PromptCondition, str]]:
    """The ``(condition, intervention)`` records :func:`evaluate_forcing_group` emits."""
    plain = {(c, NO_INTERVENTION) for c in _needed_conditions(plans)}
    return plain | {(plan.prompt_condition, plan.intervention) for plan in plans}


def evaluate_forcing_group(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    group: PromptGroup,
    plans: Sequence[ForcingPlan],
    *,
    run: str,
    question_index: int,
    options: ForcingOptions,
) -> list[ResultRecord]:
    """Score the plain prompts and every forced run for one target question.

    Args:
        model: Loaded MoE model.
        tokenizer: Matching fast tokenizer.
        structure: Discovered MoE structure.
        group: Prompts of one target question, keyed by condition.
        plans: Forced runs to perform.
        run: Run name stored on each record.
        question_index: Index of the question within the run.
        options: How to force.

    Returns:
        Plain ("none") records for every prompt the plans involve, then one record
        per plan, in plan order.

    Raises:
        ValueError: If the group lacks a prompt a plan needs.
    """
    needed = _needed_conditions(plans)
    missing = sorted(c.value for c in needed if c not in group)
    if missing:
        raise ValueError(f"the group lacks the {missing} prompts needed by the forcing plans")
    plain = {
        c: score_with_routing(model, tokenizer, structure, group[c]) for c in group if c in needed
    }
    records = [
        _record(run, question_index, group[c], score, NO_INTERVENTION)
        for c, (score, _) in plain.items()
    ]
    for plan in plans:
        prompt = group[plan.prompt_condition]
        spec = _spec_for(tokenizer, structure, group, plain, plan, options, question_index)
        with RoutingForcer(model, structure.router_module_names, spec):
            score = score_mc(model, tokenizer, prompt.text, prompt.target.answer_index)
        records.append(_record(run, question_index, prompt, score, plan.intervention))
    return records


def _spec_for(
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    group: PromptGroup,
    plain: Mapping[PromptCondition, tuple[MCScore, RoutingRecord]],
    plan: ForcingPlan,
    options: ForcingOptions,
    question_index: int,
) -> ForcingSpec:
    prompt = group[plan.prompt_condition]
    if plan.source is None:
        return random_spec(
            structure.num_layers,
            structure.num_experts,
            structure.top_k,
            target_token_positions(tokenizer, prompt),
            seed=options.random_seed + question_index,
            layers=options.layers,
        )
    alignment = align_target_tokens(tokenizer, group[plan.source], prompt)
    return spec_from_record(
        plain[plan.source][1],
        alignment.positions_a,
        alignment.positions_b,
        force_weights=options.force_weights,
        layers=options.layers,
    )


def _record(
    run: str, question_index: int, prompt: BuiltPrompt, score: MCScore, intervention: str
) -> ResultRecord:
    return ResultRecord(
        run=run,
        question_index=question_index,
        subject=prompt.target.subject,
        condition=prompt.condition,
        option_logprobs=score.option_logprobs,
        prediction=score.prediction,
        is_correct=score.is_correct,
        correct_logprob=score.correct_logprob,
        num_examples=prompt.num_examples,
        example_tokens=prompt.example_tokens,
        jaccard_per_layer=None,
        intervention=intervention,
    )


def run_forcing_file(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    structure: MoEStructure,
    prompts_path: Path,
    output_path: Path,
    plans: Sequence[ForcingPlan],
    options: ForcingOptions,
    *,
    model_provenance: Mapping[str, Any] | None = None,
    show_progress: bool = False,
) -> int:
    """Run the forcing plans over every question of a prompt file, resumably.

    The plans, ``options`` and ``model_provenance`` (see
    :func:`rr.models.model_provenance`) are recorded in the run manifest, so resuming
    under different ones is refused. See :func:`rr.runner.run_resumable` for the
    resume semantics and errors.

    Returns:
        The number of questions evaluated in this call (0 if already complete).
    """
    run = prompts_path.stem

    def evaluate(group: PromptGroup, question_index: int) -> list[ResultRecord]:
        return evaluate_forcing_group(
            model,
            tokenizer,
            structure,
            group,
            plans,
            run=run,
            question_index=question_index,
            options=options,
        )

    return run_resumable(
        prompts_path,
        output_path,
        evaluate,
        lambda group: expected_keys(group, plans),
        settings={
            "kind": "forcing",
            "model": model_provenance,
            "plans": [
                [plan.prompt_condition.value, None if plan.source is None else plan.source.value]
                for plan in plans
            ],
            "options": {
                "force_weights": options.force_weights,
                "layers": options.layers,
                "random_seed": options.random_seed,
            },
        },
        show_progress=show_progress,
    )


def parse_forcing_options(options: Mapping[str, Any], *, seed: int) -> ForcingOptions:
    """Validate the ``options`` block of a forcing experiment config.

    Raises:
        ValueError: Naming the offending key.
    """
    scope = options.get("force_scope", SUPPORTED_FORCE_SCOPE)
    if scope != SUPPORTED_FORCE_SCOPE:
        raise ValueError(f"force_scope: only {SUPPORTED_FORCE_SCOPE!r} is supported, got {scope!r}")
    layers = options.get("layers")
    if layers is not None:
        valid = isinstance(layers, (list, tuple)) and all(
            isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in layers
        )
        if not valid:
            raise ValueError(f"layers: must be null or a list of layer indices, got {layers!r}")
        layers = tuple(layers)
    return ForcingOptions(
        force_weights=parse_bool_option(options, "force_mixing_weights", True),
        layers=layers,
        random_seed=seed,
    )
