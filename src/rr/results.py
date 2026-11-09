"""Result records written by the experiment scripts, and their JSONL persistence.

Role in the project: the on-disk contract between the experiment scripts that
produce numbers (``scripts/02_run_pilot.py`` and later phases) and the analysis
that consumes them. One JSON object per line, one line per
``question x condition``, in ``<output_dir>/<run>.jsonl``.

Methodological notes:

* Records are appended one question at a time, so a crashed or preempted GPU
  run can resume. A crash can leave a torn final line; :func:`repair_torn_tail`
  drops it before resuming. Every other malformed line fails loudly on read.
* Scores are deterministic (no sampling), so re-evaluating a question after a
  crash reproduces the lost records exactly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rr.hidden import HiddenShift
from rr.prompts import PromptCondition

RESULT_SUFFIX = ".jsonl"
NO_INTERVENTION = "none"


@dataclass(frozen=True)
class ResultRecord:
    """Outcome of one prompt condition for one target question.

    Attributes:
        run: Run name, i.e. the prompt file stem (e.g. ``"bio_med_a2b"``).
        question_index: Position of the target question within the run's prompt file.
        subject: MMLU subject of the target question.
        condition: Prompt condition.
        option_logprobs: Log-prob of each answer option, in option order.
        prediction: Index of the highest-scoring option.
        is_correct: Whether ``prediction`` is the correct option.
        correct_logprob: Log-prob assigned to the correct option.
        num_examples: In-context examples in the prompt (0 for zero-shot).
        example_tokens: Token count of the example block (0 for zero-shot).
        jaccard_per_layer: Per-layer mean Jaccard overlap between this prompt's
            top-k routing on the target-question tokens and the zero-shot
            prompt's; ``None`` for the zero-shot prompt itself.
        intervention: ``"none"`` for a plain run, otherwise what was done to the
            routing, e.g. ``"force_zero_shot_routing"`` (the prompt of ``condition``
            was run with the zero-shot prompt's routing forced on its target tokens).
        hidden_shift: Per-layer shift of the hidden state each router received on the
            aligned target tokens, relative to the zero-shot prompt (see :mod:`rr.hidden`);
            ``None`` for the zero-shot prompt itself or when it was not recorded.
    """

    run: str
    question_index: int
    subject: str
    condition: PromptCondition
    option_logprobs: tuple[float, ...]
    prediction: int
    is_correct: bool
    correct_logprob: float
    num_examples: int
    example_tokens: int
    jaccard_per_layer: tuple[float, ...] | None
    intervention: str = NO_INTERVENTION
    hidden_shift: HiddenShift | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.run, str) and self.run != "", "run", "must be a non-empty string")
        _require(_is_count(self.question_index), "question_index", "must be a non-negative int")
        _require(isinstance(self.subject, str), "subject", "must be a string")
        _require(
            isinstance(self.intervention, str) and self.intervention != "",
            "intervention",
            "must be a non-empty string",
        )
        _require(
            isinstance(self.condition, PromptCondition), "condition", "must be a PromptCondition"
        )
        _require(
            len(self.option_logprobs) > 0 and all(_is_finite(v) for v in self.option_logprobs),
            "option_logprobs",
            "must be a non-empty sequence of finite numbers",
        )
        _require(
            _is_count(self.prediction) and self.prediction < len(self.option_logprobs),
            "prediction",
            f"must be an index into {len(self.option_logprobs)} options, got {self.prediction!r}",
        )
        _require(isinstance(self.is_correct, bool), "is_correct", "must be a bool")
        _require(_is_finite(self.correct_logprob), "correct_logprob", "must be a finite number")
        _require(
            self.correct_logprob in self.option_logprobs,
            "correct_logprob",
            "must be one of option_logprobs",
        )
        _require(
            not self.is_correct or self.correct_logprob == self.option_logprobs[self.prediction],
            "is_correct",
            "is True but correct_logprob differs from the predicted option's log-prob",
        )
        _require(_is_count(self.num_examples), "num_examples", "must be a non-negative int")
        _require(_is_count(self.example_tokens), "example_tokens", "must be a non-negative int")
        _require(
            self.hidden_shift is None or isinstance(self.hidden_shift, HiddenShift),
            "hidden_shift",
            "must be a HiddenShift or None",
        )
        if self.hidden_shift is not None and self.jaccard_per_layer is not None:
            _require(
                len(self.hidden_shift.relative_l2) == len(self.jaccard_per_layer),
                "hidden_shift",
                "must cover the same number of layers as jaccard_per_layer",
            )
        if self.jaccard_per_layer is not None:
            _require(
                len(self.jaccard_per_layer) > 0
                and all(_is_finite(v) and 0.0 <= v <= 1.0 for v in self.jaccard_per_layer),
                "jaccard_per_layer",
                "must be None or a non-empty sequence of values in [0, 1]",
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "run": self.run,
            "question_index": self.question_index,
            "subject": self.subject,
            "condition": self.condition.value,
            "intervention": self.intervention,
            "option_logprobs": list(self.option_logprobs),
            "prediction": self.prediction,
            "is_correct": self.is_correct,
            "correct_logprob": self.correct_logprob,
            "num_examples": self.num_examples,
            "example_tokens": self.example_tokens,
            "jaccard_per_layer": (
                None if self.jaccard_per_layer is None else list(self.jaccard_per_layer)
            ),
            "hidden_shift": None if self.hidden_shift is None else self.hidden_shift.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResultRecord:
        """Parse and validate one record (external data is never trusted).

        Raises:
            ValueError: Naming the offending field.
        """
        try:
            raw_condition = data["condition"]
        except KeyError:
            raise ValueError("missing field 'condition'") from None
        try:
            condition = PromptCondition(raw_condition)
        except ValueError:
            raise ValueError(f"condition: unknown condition {raw_condition!r}") from None
        jaccard = _field(data, "jaccard_per_layer")
        return cls(
            run=_field(data, "run"),
            question_index=_field(data, "question_index"),
            subject=_field(data, "subject"),
            condition=condition,
            option_logprobs=tuple(_field(data, "option_logprobs")),
            prediction=_field(data, "prediction"),
            is_correct=_field(data, "is_correct"),
            correct_logprob=_field(data, "correct_logprob"),
            num_examples=_field(data, "num_examples"),
            example_tokens=_field(data, "example_tokens"),
            jaccard_per_layer=None if jaccard is None else tuple(jaccard),
            intervention=data.get("intervention", NO_INTERVENTION),
            hidden_shift=(
                None
                if data.get("hidden_shift") is None
                else HiddenShift.from_dict(data["hidden_shift"])
            ),
        )


def _field(data: Mapping[str, Any], key: str) -> Any:
    try:
        return data[key]
    except KeyError:
        raise ValueError(f"missing field {key!r}") from None


def _require(condition: bool, field: str, problem: str) -> None:
    if not condition:
        raise ValueError(f"{field}: {problem}")


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def append_records(path: Path, records: Sequence[ResultRecord]) -> None:
    """Append records to a JSONL file (created, with parents, if missing).

    All lines go out in a single write followed by a flush, so a crash can tear
    at most the final line.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(lines)
        handle.flush()


def read_records(path: Path) -> list[ResultRecord]:
    """Read every record in a JSONL file.

    Raises:
        ValueError: Naming the file and line of the first malformed record.
    """
    records: list[ResultRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ResultRecord.from_dict(json.loads(line)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name} line {line_number}: {error}") from error
    return records


def repair_torn_tail(path: Path) -> bool:
    """Drop a final line that was cut off by a crash (one lacking its newline).

    Returns:
        ``True`` if the file was truncated, ``False`` if it was clean or absent.
    """
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    with path.open("r+b") as handle:
        handle.truncate(data.rfind(b"\n") + 1)
    return True


def read_results_dir(results_dir: Path) -> list[ResultRecord]:
    """Read every ``*.jsonl`` file in a results directory, in file-name order.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If it holds no result files, or any record is malformed.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    paths = sorted(results_dir.glob(f"*{RESULT_SUFFIX}"))
    if not paths:
        raise ValueError(f"no result files ({RESULT_SUFFIX}) in {results_dir}")
    return [record for path in paths for record in read_records(path)]
