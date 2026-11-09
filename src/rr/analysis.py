"""Aggregate results and compute the headline decomposition metrics.

Role in the project: turns per-condition accuracy / log-prob numbers into the
quantities the research question asks about. Used in Phase 3 (main
experiment) and Phase 4 (extensions); the plotting/loading helpers support
``scripts/05_make_figures.py``.

Definitions, with ``Z`` = zero-shot score, ``F`` = few-shot score:

* Fraction of gain **retained** =  (forced - Z) / (F - Z), where *forced* is the
  few-shot prompt run with zero-shot routing forced on the target tokens.
  Low retention means the routing change carried the gain ("route"); high
  retention means the hidden-state shift carried it ("representation").
* Fraction of gain **recovered** = (forced - Z) / (F - Z), where *forced* is the
  zero-shot prompt run with few-shot routing forced in.

Both are ratios of *aggregate* scores (e.g. accuracy averaged over questions
first), not averages of per-question ratios, which are unstable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rr.prompts import PromptCondition
from rr.results import ResultRecord, read_results_dir

MIN_GAIN = 1e-9  # smallest |F - Z| for which the ratio is considered defined
DEFAULT_N_BOOT = 2000
DEFAULT_CONFIDENCE = 0.95


def _gain_fraction(forced: float, zero_shot: float, few_shot: float, min_gain: float) -> float:
    gain = few_shot - zero_shot
    if abs(gain) < min_gain:
        raise ValueError(
            f"Few-shot gain is ~0 (few_shot={few_shot}, zero_shot={zero_shot}); "
            "the fraction of gain is undefined. Skip this pair/condition."
        )
    return (forced - zero_shot) / gain


def fraction_of_gain_retained(
    forced: float, zero_shot: float, few_shot: float, min_gain: float = MIN_GAIN
) -> float:
    """Fraction of the few-shot gain kept when zero-shot routing is forced.

    Args:
        forced: Score of the few-shot prompt with zero-shot routing forced on
            target-question tokens.
        zero_shot: Score of the plain zero-shot prompt.
        few_shot: Score of the plain few-shot prompt.
        min_gain: Minimum ``|few_shot - zero_shot|`` for the ratio to be defined.

    Returns:
        ``(forced - zero_shot) / (few_shot - zero_shot)``. ``1.0`` means forcing
        removed none of the gain; ``0.0`` means it removed all of it. Values
        outside ``[0, 1]`` are possible and meaningful (e.g. forcing helps).

    Raises:
        ValueError: If the gain is (near) zero.
    """
    return _gain_fraction(forced, zero_shot, few_shot, min_gain)


def fraction_of_gain_recovered(
    forced: float, zero_shot: float, few_shot: float, min_gain: float = MIN_GAIN
) -> float:
    """Fraction of the few-shot gain recovered by forcing few-shot routing in.

    Args:
        forced: Score of the zero-shot prompt with few-shot routing forced on
            target-question tokens.
        zero_shot: Score of the plain zero-shot prompt.
        few_shot: Score of the plain few-shot prompt.
        min_gain: Minimum ``|few_shot - zero_shot|`` for the ratio to be defined.

    Returns:
        ``(forced - zero_shot) / (few_shot - zero_shot)``. ``1.0`` means routing
        alone reproduces the whole gain; ``0.0`` means none of it.

    Raises:
        ValueError: If the gain is (near) zero.
    """
    return _gain_fraction(forced, zero_shot, few_shot, min_gain)


def load_results(results_dir: Path) -> list[ResultRecord]:
    """Load per-question result records written by the experiment scripts.

    Args:
        results_dir: Directory such as ``results/pilot``.

    Returns:
        A list of records (one per question x condition), in file-name order.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If it holds no result files or a record is malformed.
    """
    return read_results_dir(results_dir)


def plot_gain_fractions(fractions_by_pair: Mapping[str, float], output_path: Path) -> None:
    """Bar chart of gain fractions per domain pair/direction.

    Args:
        fractions_by_pair: Mapping from ``"pair:direction"`` to a fraction.
        output_path: Where to save the figure (under ``figures/``).
    """
    # TODO(Phase 3): matplotlib bar chart with bootstrap CIs; order by domain
    # distance (Phase 4 relates the fraction to distance and direction).
    raise NotImplementedError("TODO(Phase 3): plot gain fractions")


# --- Phase 2 pilot summaries ---------------------------------------------------------------

RunCondition = tuple[str, PromptCondition]


@dataclass(frozen=True)
class BootstrapResult:
    """Mean paired difference with a percentile bootstrap interval."""

    mean_diff: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class Contrast:
    """One paired comparison within a run.

    ``mean_diff`` is ``metric(condition_a) - metric(condition_b)`` averaged over
    questions, with a bootstrap interval resampling questions.
    """

    run: str
    name: str
    mean_diff: float
    low: float
    high: float
    n: int


def _mean_jaccard(record: ResultRecord) -> float | None:
    return None if record.jaccard_per_layer is None else float(np.mean(record.jaccard_per_layer))


METRICS: Mapping[str, Callable[[ResultRecord], float | None]] = {
    "accuracy": lambda r: float(r.is_correct),
    "correct_logprob": lambda r: r.correct_logprob,
    "jaccard": _mean_jaccard,
}

# (metric, condition_a, condition_b): reports metric(a) - metric(b).
# Pilot question (a): do examples help?  Pilot question (b): do relevant examples
# move routing more? Jaccard is an overlap, so LOWER overlap means MORE routing
# change; "irrelevant - cross_domain" > 0 means cross-domain examples shifted routing more.
PILOT_CONTRASTS: tuple[tuple[str, PromptCondition, PromptCondition], ...] = (
    ("accuracy", PromptCondition.CROSS_DOMAIN, PromptCondition.ZERO_SHOT),
    ("accuracy", PromptCondition.SAME_DOMAIN, PromptCondition.ZERO_SHOT),
    ("accuracy", PromptCondition.CROSS_DOMAIN, PromptCondition.IRRELEVANT),
    ("correct_logprob", PromptCondition.CROSS_DOMAIN, PromptCondition.ZERO_SHOT),
    ("jaccard", PromptCondition.IRRELEVANT, PromptCondition.CROSS_DOMAIN),
    ("jaccard", PromptCondition.IRRELEVANT, PromptCondition.SAME_DOMAIN),
)


def _metric_by_run_condition(
    records: Sequence[ResultRecord], metric: str
) -> dict[RunCondition, dict[int, float]]:
    """``{(run, condition): {question_index: value}}``, skipping undefined values."""
    extract = METRICS[metric]
    grouped: dict[RunCondition, dict[int, float]] = defaultdict(dict)
    for record in records:
        value = extract(record)
        if value is not None:
            grouped[(record.run, record.condition)][record.question_index] = value
    return dict(grouped)


def _mean_by_run_condition(
    records: Sequence[ResultRecord], metric: str
) -> dict[RunCondition, float]:
    return {
        key: float(np.mean(list(values.values())))
        for key, values in _metric_by_run_condition(records, metric).items()
    }


def accuracy_by_condition(records: Sequence[ResultRecord]) -> dict[RunCondition, float]:
    """Mean accuracy per ``(run, condition)``."""
    return _mean_by_run_condition(records, "accuracy")


def mean_jaccard_by_condition(records: Sequence[ResultRecord]) -> dict[RunCondition, float]:
    """Mean target-token routing overlap with zero-shot, per ``(run, condition)``.

    Averaged over layers, then questions. Zero-shot has no entry (it is the baseline).
    """
    return _mean_by_run_condition(records, "jaccard")


def paired_bootstrap_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    """Mean of ``a - b`` with a percentile bootstrap interval over pairs.

    Args:
        values_a: Per-question values for condition A.
        values_b: Per-question values for condition B, in the same question order.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed (the interval is reproducible).
        confidence: Interval coverage, strictly between 0 and 1.

    Raises:
        ValueError: On empty input, unequal lengths, or invalid ``n_boot``/``confidence``.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"values must have the same length, got {len(values_a)} and {len(values_b)}"
        )
    if len(values_a) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = differences[rng.integers(0, len(differences), size=(n_boot, len(differences)))]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled.mean(axis=1), [alpha, 1.0 - alpha])
    return BootstrapResult(
        mean_diff=float(differences.mean()), low=float(low), high=float(high), n=len(differences)
    )


def pilot_contrasts(
    records: Sequence[ResultRecord],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[Contrast]:
    """Paired pilot comparisons for every run (see :data:`PILOT_CONTRASTS`).

    A contrast is skipped for a run that lacks either condition.

    Raises:
        ValueError: If two compared conditions cover different questions.
    """
    contrasts: list[Contrast] = []
    runs = sorted({record.run for record in records})
    values = {metric: _metric_by_run_condition(records, metric) for metric in METRICS}
    for run in runs:
        for metric, condition_a, condition_b in PILOT_CONTRASTS:
            a = values[metric].get((run, condition_a))
            b = values[metric].get((run, condition_b))
            if a is None or b is None:
                continue
            if a.keys() != b.keys():
                raise ValueError(
                    f"{run}: {condition_a} and {condition_b} are not paired "
                    f"(they cover different questions)"
                )
            order = sorted(a)
            result = paired_bootstrap_difference(
                [a[i] for i in order],
                [b[i] for i in order],
                n_boot=n_boot,
                seed=seed,
                confidence=confidence,
            )
            contrasts.append(
                Contrast(
                    run=run,
                    name=f"{metric}: {condition_a} - {condition_b}",
                    mean_diff=result.mean_diff,
                    low=result.low,
                    high=result.high,
                    n=result.n,
                )
            )
    return contrasts


def format_pilot_summary(records: Sequence[ResultRecord], *, seed: int = 0) -> str:
    """Plain-text pilot report: per-condition means, then paired contrasts with intervals."""
    accuracy = accuracy_by_condition(records)
    logprob = _mean_by_run_condition(records, "correct_logprob")
    jaccard = mean_jaccard_by_condition(records)
    lines = [f"{'run':<22}{'condition':<14}{'accuracy':>9}{'logprob':>9}{'jaccard':>9}"]
    for key in sorted(accuracy, key=lambda k: (k[0], list(PromptCondition).index(k[1]))):
        shift = f"{jaccard[key]:.3f}" if key in jaccard else "-"
        lines.append(
            f"{key[0]:<22}{key[1].value:<14}{accuracy[key]:>9.3f}{logprob[key]:>9.3f}{shift:>9}"
        )
    lines.append("")
    lines.append("Paired contrasts (mean difference [95% bootstrap interval], n questions):")
    for contrast in pilot_contrasts(records, seed=seed):
        lines.append(
            f"{contrast.run:<22}{contrast.name:<46}"
            f"{contrast.mean_diff:+.3f} [{contrast.low:+.3f}, {contrast.high:+.3f}]  n={contrast.n}"
        )
    return "\n".join(lines)
