"""Resumable per-file experiment runner shared by the pilot and the forcing experiments.

Role in the project: every experiment evaluates one prompt file (a "run") question
by question and appends results to ``<output_dir>/<run>.jsonl``. This module owns
the resume logic so a crashed or preempted GPU job continues where it stopped.

Methodological notes:

* A question counts as done when every record the experiment *expects* for it is
  on disk. A half-finished question is re-evaluated and only its missing records
  are appended; scoring is deterministic, so the re-evaluated values are identical.
* Existing results must belong to this run and fit the prompt file; otherwise the
  runner fails loudly rather than mixing results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rr.manifest import build_manifest, ensure_manifest
from rr.prompts import BuiltPrompt, PromptCondition, load_prompts
from rr.results import ResultRecord, append_records, read_records, repair_torn_tail

PromptGroup = Mapping[PromptCondition, BuiltPrompt]
RecordKey = tuple[int, PromptCondition, str]  # (question_index, condition, intervention)

Evaluator = Callable[[PromptGroup, int], list[ResultRecord]]
ExpectedKeys = Callable[[PromptGroup], set[tuple[PromptCondition, str]]]


def group_prompts_by_target(
    prompts: Sequence[BuiltPrompt],
) -> list[dict[PromptCondition, BuiltPrompt]]:
    """Group a prompt file's prompts by target question, keeping file order.

    Raises:
        ValueError: If a target has two prompts for the same condition.
    """
    groups: dict[tuple[str, str], dict[PromptCondition, BuiltPrompt]] = {}
    for prompt in prompts:
        group = groups.setdefault((prompt.target.subject, prompt.target.question), {})
        if prompt.condition in group:
            raise ValueError(
                f"duplicate {prompt.condition} prompt for target {prompt.target.question[:60]!r}"
            )
        group[prompt.condition] = prompt
    return list(groups.values())


def run_resumable(
    prompts_path: Path,
    output_path: Path,
    evaluate: Evaluator,
    expected_keys: ExpectedKeys,
    *,
    settings: Mapping[str, Any],
    show_progress: bool = False,
) -> int:
    """Evaluate every question in a prompt file, resuming from existing results.

    Args:
        prompts_path: A JSONL file from ``scripts/01_build_prompts.py``; its stem is the run name.
        output_path: Results JSONL; created if missing, resumed if present.
        evaluate: ``(group, question_index) -> records`` for one target question.
        expected_keys: ``group -> {(condition, intervention)}`` the evaluator will emit.
        settings: JSON-serializable settings that shape the results (options, planned
            interventions, model identity). Stored in a manifest next to the results and
            required to match on every resume (see :mod:`rr.manifest`).
        show_progress: Show a progress bar.

    Returns:
        The number of questions evaluated in this call (0 if already complete).

    Raises:
        ValueError: If the existing manifest differs from this run's, or results exist
            without one; if existing results belong to another run, repeat a record, or
            do not fit the prompt file; or if ``evaluate`` emits unexpected records.
    """
    if output_path.resolve() == prompts_path.resolve():
        raise ValueError(
            f"{output_path} is the prompt file itself; results would be written over the prompts"
        )
    run = prompts_path.stem
    groups = group_prompts_by_target(load_prompts(prompts_path))
    repair_torn_tail(output_path)
    ensure_manifest(output_path, build_manifest(run, prompts_path, settings))
    done = _completed_keys(output_path, run, num_questions=len(groups))
    evaluated = 0
    progress = tqdm(enumerate(groups), total=len(groups), desc=run, disable=not show_progress)
    for index, group in progress:
        expected = {(index, condition, name) for condition, name in expected_keys(group)}
        if expected <= done:
            continue
        records = evaluate(group, index)
        produced = {_key(record) for record in records}
        if produced != expected:
            raise ValueError(
                f"question {index}: evaluator produced {sorted(map(str, produced ^ expected))} "
                "that do not match the expected records"
            )
        append_records(output_path, [r for r in records if _key(r) not in done])
        evaluated += 1
    return evaluated


def _key(record: ResultRecord) -> RecordKey:
    return (record.question_index, record.condition, record.intervention)


def _completed_keys(output_path: Path, run: str, num_questions: int) -> set[RecordKey]:
    if not output_path.exists():
        return set()
    done: set[RecordKey] = set()
    for record in read_records(output_path):
        if record.run != run:
            raise ValueError(f"{output_path} holds run {record.run!r} results, expected {run!r}")
        if record.question_index >= num_questions:
            raise ValueError(
                f"{output_path} does not match the prompts: question_index "
                f"{record.question_index} but the prompt file has {num_questions} questions"
            )
        key = _key(record)
        if key in done:
            raise ValueError(f"{output_path} has duplicate records for {key}")
        done.add(key)
    return done
