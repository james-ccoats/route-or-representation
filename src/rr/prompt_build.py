"""Build the prompts of one domain direction for every replicate and shot count.

Role in the project: ``scripts/01_build_prompts.py`` loads data and the tokenizer; this
module turns pools of questions into the prompt lists, one per run (see :mod:`rr.shots`).
It needs no network access, so it is tested on small synthetic pools.

Design (see D14 in docs/design_decisions.md):

* Each replicate draws one ordered example set per domain from a pooled shot split.
  The k-shot prompt uses the first ``k`` examples of that set, so going from k to k+1
  adds one example and keeps the rest; the curve over k is a clean dose-response.
* Within a run, every target question sees the same examples.
* The evaluation questions are the same in every run of a direction and never include a
  question used as a same-domain example.
* The irrelevant control is re-matched in count and token length for each replicate and k.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rr.data import MCQuestion, draw_shot_sets, sample_questions
from rr.prompts import (
    BuiltPrompt,
    PromptCondition,
    build_condition_prompts,
    count_example_tokens,
    save_prompts,
    select_irrelevant_examples,
    verify_target_tokens,
)
from rr.shots import run_stem

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Spacing between the seeds of the irrelevant-control searches, so each (replicate, k)
# gets its own search without colliding with a neighbour's.
IRRELEVANT_SEED_STRIDE = 1000


def build_direction_prompts(
    *,
    direction: str,
    source_pool: Sequence[MCQuestion],
    target_pool: Sequence[MCQuestion],
    eval_pool: Sequence[MCQuestion],
    irrelevant_pool: Sequence[MCQuestion],
    tokenizer: PreTrainedTokenizerBase,
    conditions: Sequence[PromptCondition],
    num_shots: int,
    shot_counts: Sequence[int],
    num_replicates: int,
    num_questions: int,
    seed: int,
) -> dict[str, list[BuiltPrompt]]:
    """Build every run's prompts for one source -> target direction.

    Args:
        direction: Run name such as ``"bio_med:a2b"``.
        source_pool: Questions the cross-domain examples are drawn from.
        target_pool: Questions the same-domain examples are drawn from.
        eval_pool: Questions the evaluated targets are sampled from (a different split
            from the shot pools).
        irrelevant_pool: Candidates for the length-matched control.
        tokenizer: Fast tokenizer, used for length matching and target-token checks.
        conditions: Conditions to build (zero-shot is built in every run).
        num_shots: Examples drawn per replicate and domain; the largest ``k``.
        shot_counts: The values of ``k`` to build, each in ``[1, num_shots]``.
        num_replicates: Number of independent example sets.
        num_questions: Evaluated questions per direction.
        seed: Seed for example draws, question sampling and the control search.

    Returns:
        ``{run stem: prompts}`` with every question's prompts for every condition.

    Raises:
        ValueError: If a shot count is out of range, a pool is too small, or the
            irrelevant control cannot be matched.
    """
    if not shot_counts or any(not 1 <= k <= num_shots for k in shot_counts):
        raise ValueError(f"shot_counts must lie in [1, num_shots={num_shots}], got {shot_counts}")
    cross_sets = draw_shot_sets(source_pool, num_shots, num_replicates, seed)
    same_sets = draw_shot_sets(target_pool, num_shots, num_replicates, seed)
    used_as_shots = [q for shots in same_sets for q in shots]
    eval_questions = sample_questions(eval_pool, num_questions, seed, exclude=used_as_shots)
    needs_control = PromptCondition.IRRELEVANT in conditions

    runs: dict[str, list[BuiltPrompt]] = {}
    for replicate in range(num_replicates):
        for k in shot_counts:
            cross = cross_sets[replicate][:k]
            irrelevant = (
                select_irrelevant_examples(
                    irrelevant_pool,
                    k,
                    count_example_tokens(cross, tokenizer),
                    tokenizer,
                    seed + IRRELEVANT_SEED_STRIDE * replicate + k,
                )
                if needs_control
                else []
            )
            runs[run_stem(direction, replicate, k)] = _prompts_for(
                eval_questions,
                cross,
                same_sets[replicate][:k],
                irrelevant,
                conditions,
                tokenizer,
            )
    return runs


def _prompts_for(
    questions: Sequence[MCQuestion],
    cross: Sequence[MCQuestion],
    same: Sequence[MCQuestion],
    irrelevant: Sequence[MCQuestion],
    conditions: Sequence[PromptCondition],
    tokenizer: PreTrainedTokenizerBase,
) -> list[BuiltPrompt]:
    prompts: list[BuiltPrompt] = []
    for question in questions:
        built = list(
            build_condition_prompts(
                question,
                source_examples=cross,
                same_domain_examples=same,
                irrelevant_examples=irrelevant,
                conditions=conditions,
                tokenizer=tokenizer,
            ).values()
        )
        verify_target_tokens(built, tokenizer)
        prompts.extend(built)
    return prompts


def without_overlap(
    pool: Sequence[MCQuestion], eval_pool: Sequence[MCQuestion]
) -> list[MCQuestion]:
    """Drop shot-pool questions that also appear in the evaluation split.

    MMLU repeats a few questions across splits (college_physics has 6 shared between
    its pooled dev/validation questions and its test split). A repeated question used as
    an example would show the model the answer to a question it is later scored on.
    """
    evaluated = set(eval_pool)
    return [question for question in pool if question not in evaluated]


def write_runs(runs: Mapping[str, Sequence[BuiltPrompt]], prompts_dir: Path) -> None:
    """Write each run to ``<prompts_dir>/<stem>.jsonl``.

    Refuses to write if the directory already holds prompt files this build does not
    produce, because the pilot and forcing scripts run every ``*.jsonl`` in the directory
    and a leftover file from an earlier build would silently join the experiment.
    Files with the same names are overwritten.

    Raises:
        ValueError: Listing the leftover files; nothing is written.
    """
    leftover = (
        sorted(path.name for path in prompts_dir.glob("*.jsonl") if path.stem not in runs)
        if prompts_dir.is_dir()
        else []
    )
    if leftover:
        raise ValueError(
            f"{prompts_dir} already holds prompt files this build does not produce: "
            f"{leftover}. Remove them or use another prompts_dir."
        )
    for stem, prompts in runs.items():
        save_prompts(prompts, prompts_dir / f"{stem}.jsonl")
