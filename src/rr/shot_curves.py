"""Gain against the number of examples ``k``, with spread across example-set replicates.

Role in the project: every (direction, replicate, k) is its own run (see :mod:`rr.shots`).
This module groups runs back together to answer how the few-shot gain, the routing
shift and the retained/recovered fractions change as ``k`` goes from 1 to 5, and how much
of that depends on which examples were drawn.

Methodological notes:

* The replicates of one direction and k ask the same questions, so a gain is first
  computed per replicate (condition minus zero-shot, paired by question). The reported
  gain is the mean over replicates, and the replicate range shows how much the draw of
  examples matters.
* The interval resamples questions jointly across replicates. It covers question noise
  only. The spread between replicates is a separate quantity, reported beside it rather
  than folded in, because three replicates are too few to estimate a variance of draws.
* Runs whose names are not in the shot-grid form are ignored here; use the per-run
  summaries for those.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rr.analysis import DEFAULT_CONFIDENCE, DEFAULT_N_BOOT, format_pilot_summary
from rr.analysis import METRICS as RECORD_METRICS
from rr.force_analysis import METRICS as GAIN_METRICS
from rr.force_analysis import GainFraction, format_force_summary, gain_fractions, sanity_reports
from rr.prompts import PromptCondition
from rr.results import NO_INTERVENTION, ResultRecord
from rr.shots import ShotRun, parse_run_stem

ZERO = PromptCondition.ZERO_SHOT
CONDITION_ORDER = {condition: i for i, condition in enumerate(PromptCondition)}


@dataclass(frozen=True)
class ShotGain:
    """Gain of one few-shot condition over zero-shot at one ``k``.

    Attributes:
        direction: Domain direction, e.g. ``"bio_med_a2b"``.
        condition: The few-shot condition.
        shots: Number of examples ``k``.
        metric: ``"accuracy"`` or ``"correct_logprob"``.
        mean_gain: Gain averaged over replicates (and questions).
        low: Lower bound of the interval over questions.
        high: Upper bound of the interval over questions.
        replicate_gains: The gain of each replicate, in replicate order.
        n_replicates: Number of replicates.
        n_questions: Number of questions per replicate.
    """

    direction: str
    condition: PromptCondition
    shots: int
    metric: str
    mean_gain: float
    low: float
    high: float
    replicate_gains: tuple[float, ...]
    n_replicates: int
    n_questions: int


@dataclass(frozen=True)
class ShotMean:
    """A per-record metric (routing overlap, hidden-state shift...) averaged at one ``k``."""

    direction: str
    condition: PromptCondition
    shots: int
    metric: str
    mean: float
    replicate_means: tuple[float, ...]


@dataclass(frozen=True)
class FractionSummary:
    """A gain fraction at one ``k``, summarised over replicates.

    ``mean`` averages the replicates whose fraction is defined (``None`` if none is);
    ``n_defined`` of ``n_replicates`` were.
    """

    direction: str
    condition: PromptCondition
    kind: str
    metric: str
    shots: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    n_defined: int
    n_replicates: int


def _grid(records: Sequence[ResultRecord]) -> list[tuple[ShotRun, ResultRecord]]:
    """Plain (non-intervention) records of runs that are part of the shot grid."""
    parsed = []
    for record in records:
        run = parse_run_stem(record.run)
        if run is not None and record.intervention == NO_INTERVENTION:
            parsed.append((run, record))
    return parsed


def gain_by_shots(
    records: Sequence[ResultRecord],
    metric: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[ShotGain]:
    """Gain over zero-shot for every direction, few-shot condition and ``k``.

    Raises:
        ValueError: On an unknown metric, a few-shot run without its zero-shot records, or
            replicates (or a condition and its zero-shot) covering different questions.
    """
    if metric not in GAIN_METRICS:
        raise ValueError(f"metric must be one of {sorted(GAIN_METRICS)}, got {metric!r}")
    extract = GAIN_METRICS[metric]
    values: dict[tuple, dict[int, float]] = defaultdict(dict)
    for run, record in _grid(records):
        key = (run.direction, run.shots, run.replicate, record.condition)
        values[key][record.question_index] = extract(record)

    gains = []
    cells = sorted({(d, k, c) for d, k, _, c in values if c is not ZERO}, key=_cell_order)
    for direction, shots, condition in cells:
        replicates = sorted(
            r for d, k, r, c in values if (d, k, c) == (direction, shots, condition)
        )
        differences = []
        for replicate in replicates:
            few = values[(direction, shots, replicate, condition)]
            zero = values.get((direction, shots, replicate, ZERO))
            where = f"{direction} r{replicate} k{shots} {condition.value}"
            if zero is None:
                raise ValueError(f"{where}: no zero-shot records to compare against")
            if few.keys() != zero.keys():
                raise ValueError(f"{where}: covers different questions than its zero-shot")
            order = sorted(few)
            differences.append(np.array([few[i] - zero[i] for i in order], dtype=float))
        if len({tuple(sorted(values[(direction, shots, r, condition)])) for r in replicates}) != 1:
            raise ValueError(
                f"{direction} k{shots} {condition.value}: replicates ask different questions"
            )
        gains.append(
            _shot_gain(direction, condition, shots, metric, differences, n_boot, seed, confidence)
        )
    return gains


def _cell_order(cell: tuple[str, int, PromptCondition]) -> tuple[str, int, int]:
    direction, shots, condition = cell
    return (direction, CONDITION_ORDER[condition], shots)


def _shot_gain(
    direction: str,
    condition: PromptCondition,
    shots: int,
    metric: str,
    differences: list[np.ndarray],
    n_boot: int,
    seed: int,
    confidence: float,
) -> ShotGain:
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    stacked = np.vstack(differences)  # [replicates, questions]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, stacked.shape[1], size=(n_boot, stacked.shape[1]))
    resampled = stacked[:, idx].mean(axis=(0, 2))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [alpha, 1.0 - alpha])
    return ShotGain(
        direction=direction,
        condition=condition,
        shots=shots,
        metric=metric,
        mean_gain=float(stacked.mean()),
        low=float(low),
        high=float(high),
        replicate_gains=tuple(float(row.mean()) for row in stacked),
        n_replicates=stacked.shape[0],
        n_questions=stacked.shape[1],
    )


def mean_by_shots(records: Sequence[ResultRecord], metric: str) -> list[ShotMean]:
    """A per-record metric averaged over questions, then replicates, for few-shot conditions.

    Records where the metric is undefined (the zero-shot prompt's routing overlap, or
    hidden-state shift when it was not recorded) are skipped.
    """
    if metric not in RECORD_METRICS:
        raise ValueError(f"metric must be one of {sorted(RECORD_METRICS)}, got {metric!r}")
    extract = RECORD_METRICS[metric]
    per_replicate: dict[tuple, list[float]] = defaultdict(list)
    for run, record in _grid(records):
        value = extract(record)
        if value is not None and record.condition is not ZERO:
            per_replicate[(run.direction, record.condition, run.shots, run.replicate)].append(value)
    cells = sorted(
        {(d, c, k) for d, c, k, _ in per_replicate},
        key=lambda t: (t[0], CONDITION_ORDER[t[1]], t[2]),
    )
    summaries = []
    for direction, condition, shots in cells:
        replicates = sorted(
            r for d, c, k, r in per_replicate if (d, c, k) == (direction, condition, shots)
        )
        means = tuple(
            float(np.mean(per_replicate[(direction, condition, shots, r)])) for r in replicates
        )
        summaries.append(
            ShotMean(direction, condition, shots, metric, float(np.mean(means)), means)
        )
    return summaries


def fractions_by_shots(fractions: Sequence[GainFraction]) -> list[FractionSummary]:
    """Group per-run gain fractions by direction, condition, kind, metric and ``k``."""
    groups: dict[tuple, list[float | None]] = defaultdict(list)
    for fraction in fractions:
        run = parse_run_stem(fraction.run)
        if run is None:
            continue
        key = (run.direction, fraction.condition, fraction.kind, fraction.metric, run.shots)
        groups[key].append(fraction.estimate)
    summaries = []
    for key in sorted(groups, key=lambda t: (t[0], CONDITION_ORDER[t[1]], t[2], t[3], t[4])):
        estimates = groups[key]
        defined = [e for e in estimates if e is not None]
        summaries.append(
            FractionSummary(
                *key,
                mean=float(np.mean(defined)) if defined else None,
                minimum=min(defined) if defined else None,
                maximum=max(defined) if defined else None,
                n_defined=len(defined),
                n_replicates=len(estimates),
            )
        )
    return summaries


# --- text reports --------------------------------------------------------------------------

METRIC_LABELS = {"accuracy": "accuracy", "correct_logprob": "correct-option log-prob"}


def _grid_text(
    title: str, rows: dict[tuple[str, str], dict[int, str]], shots: Sequence[int], width: int
) -> list[str]:
    header = f"{'direction':<22}{'condition':<14}" + "".join(f"{f'k={k}':>{width}}" for k in shots)
    lines = [title, header]
    for (direction, label), cells in rows.items():
        lines.append(
            f"{direction:<22}{label:<14}" + "".join(f"{cells.get(k, '-'):>{width}}" for k in shots)
        )
    return lines


def format_shot_gain_table(
    records: Sequence[ResultRecord], *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> str:
    """Gain over zero-shot against ``k``: means with intervals, replicate ranges, shift columns."""
    blocks: list[str] = []
    for metric in GAIN_METRICS:
        gains = gain_by_shots(records, metric, n_boot=n_boot, seed=seed)
        if not gains:
            continue
        shots = sorted({g.shots for g in gains})
        means: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
        ranges: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
        for g in gains:
            row = (g.direction, g.condition.value)
            means[row][g.shots] = f"{g.mean_gain:+.3f} [{g.low:+.3f},{g.high:+.3f}]"
            ranges[row][g.shots] = f"{min(g.replicate_gains):+.3f}..{max(g.replicate_gains):+.3f}"
        label = METRIC_LABELS[metric]
        blocks += _grid_text(
            f"Gain over zero-shot, {label} (mean over replicates [95% interval over questions])",
            means,
            shots,
            24,
        )
        blocks += [""] + _grid_text(f"Range of the per-replicate gain, {label}", ranges, shots, 24)
        blocks.append("")
    for metric, title in (
        ("jaccard", "Routing overlap with zero-shot (Jaccard)"),
        ("hidden_shift", "Hidden-state shift from zero-shot (relative L2)"),
    ):
        summaries = mean_by_shots(records, metric)
        if summaries:
            shots = sorted({s.shots for s in summaries})
            rows: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
            for s in summaries:
                rows[(s.direction, s.condition.value)][s.shots] = f"{s.mean:.3f}"
            blocks += _grid_text(title, rows, shots, 10) + [""]
    return "\n".join(blocks).rstrip()


def format_shot_force_summary(
    records: Sequence[ResultRecord], *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> str:
    """Retained/recovered fractions against ``k`` (replicate mean and range), then the controls."""
    summaries = fractions_by_shots(gain_fractions(records, n_boot=n_boot, seed=seed))
    lines: list[str] = []
    for metric in GAIN_METRICS:
        for kind in ("retained", "recovered"):
            chosen = [s for s in summaries if s.metric == metric and s.kind == kind]
            if not chosen:
                continue
            shots = sorted({s.shots for s in chosen})
            rows: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
            for s in chosen:
                rows[(s.direction, s.condition.value)][s.shots] = _fraction_cell(s)
            lines += _grid_text(
                f"Fraction of gain {kind}, {METRIC_LABELS[metric]} "
                "(replicate mean, range, defined/total)",
                rows,
                shots,
                30,
            ) + [""]
    lines += ["Sanity controls over all runs:", _controls_line(records)]
    return "\n".join(lines)


def _fraction_cell(summary: FractionSummary) -> str:
    if summary.mean is None:
        return f"undefined 0/{summary.n_replicates}"
    return (
        f"{summary.mean:+.2f} ({summary.minimum:+.2f}..{summary.maximum:+.2f}) "
        f"{summary.n_defined}/{summary.n_replicates}"
    )


def _controls_line(records: Sequence[ResultRecord]) -> str:
    reports = [r for r in sanity_reports(records) if parse_run_stem(r.run) is not None]
    identity = [r for r in reports if r.identity_ok is not None]
    random = [r for r in reports if r.random_hurts is not None]
    parts = []
    if identity:
        worst = max(r.identity_max_abs_logprob_diff for r in identity)
        passed = sum(r.identity_ok for r in identity)
        parts.append(f"identity ok in {passed}/{len(identity)} runs (worst |dlogprob| {worst:.2g})")
    else:
        parts.append("identity check not run")
    if random:
        hurt = sum(r.random_hurts for r in random)
        parts.append(f"random experts hurt accuracy in {hurt}/{len(random)} runs")
    else:
        parts.append("random check not run")
    return "; ".join(parts)


def _has_shot_grid(records: Sequence[ResultRecord]) -> bool:
    return any(parse_run_stem(record.run) is not None for record in records)


def summarize_pilot_results(
    records: Sequence[ResultRecord], *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> str:
    """The pilot report: gain against ``k`` for shot-grid runs, per-run tables otherwise."""
    if _has_shot_grid(records):
        return format_shot_gain_table(records, n_boot=n_boot, seed=seed)
    return format_pilot_summary(records, seed=seed)


def summarize_force_results(
    records: Sequence[ResultRecord], *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> str:
    """The forcing report: fractions against ``k`` for shot-grid runs, per-run otherwise."""
    if _has_shot_grid(records):
        return format_shot_force_summary(records, n_boot=n_boot, seed=seed)
    return format_force_summary(records, seed=seed)
