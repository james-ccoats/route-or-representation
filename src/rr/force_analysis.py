"""Gain fractions and sanity checks from forcing-experiment records.

Role in the project: turns the records written by ``03_force_routing.py`` and
``04_reverse_check.py`` into the headline quantities, per run (domain pair and
direction) and few-shot condition ``C``, with ``Z`` the zero-shot score and
``F`` the plain ``C`` score:

* **retained** = (score of ``C`` with zero-shot routing forced - Z) / (F - Z)
* **recovered** = (score of zero-shot with ``C`` routing forced - Z) / (F - Z)

Methodological notes:

* Ratios are taken of *aggregate* scores (means over questions), never averaged
  per question. Intervals resample questions jointly across Z, F and the forced
  score, because all three come from the same questions.
* A bootstrap resample whose few-shot gain is ~0 has no defined ratio and is
  dropped; ``undefined_fraction`` reports how many were. A large share means the
  fraction is unstable and should not be over-read.
* Both accuracy and the mean log-prob of the correct option are reported;
  log-prob is smoother (see design decision D3).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from rr.analysis import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOT,
    MIN_GAIN,
    fraction_of_gain_recovered,
    fraction_of_gain_retained,
)
from rr.force_experiment import ForcingPlan
from rr.prompts import PromptCondition
from rr.results import NO_INTERVENTION, ResultRecord

METRICS: dict[str, Callable[[ResultRecord], float]] = {
    "accuracy": lambda r: float(r.is_correct),
    "correct_logprob": lambda r: r.correct_logprob,
}
# Forcing a prompt's own routing back onto itself should reproduce its option log-probs.
# Starting value; tighten once identity-check numerics on the real model are known.
IDENTITY_TOLERANCE = 1e-3

ZERO = PromptCondition.ZERO_SHOT
SeriesKey = tuple[str, PromptCondition, str]  # (run, condition, intervention)


@dataclass(frozen=True)
class GainFraction:
    """One gain fraction with its bootstrap interval.

    Attributes:
        run: Run name (domain pair and direction).
        condition: The few-shot condition being decomposed.
        kind: ``"retained"`` (zero-shot routing forced into the few-shot prompt) or
            ``"recovered"`` (few-shot routing forced into the zero-shot prompt).
        metric: ``"accuracy"`` or ``"correct_logprob"``.
        estimate: Ratio of aggregate scores; ``None`` if the few-shot gain is ~0.
        low: Lower bootstrap bound (``None`` if undefined).
        high: Upper bootstrap bound (``None`` if undefined).
        gain: ``F - Z`` on this metric; the denominator of the ratio.
        n: Number of questions.
        undefined_fraction: Share of bootstrap resamples dropped for a ~0 gain.
    """

    run: str
    condition: PromptCondition
    kind: str
    metric: str
    estimate: float | None
    low: float | None
    high: float | None
    gain: float
    n: int
    undefined_fraction: float


@dataclass(frozen=True)
class SanityReport:
    """Outcome of the two sanity controls for one run.

    Attributes:
        run: Run name.
        identity_max_abs_logprob_diff: Largest absolute option log-prob change when
            the zero-shot prompt's own routing is forced back onto it; ``None`` if not run.
        identity_ok: ``True`` if that change is within :data:`IDENTITY_TOLERANCE`.
        random_accuracy_change: Per few-shot condition, accuracy with random experts
            forced minus plain accuracy.
        random_hurts: ``True`` if random forcing lowered accuracy for every condition
            tested; ``None`` if the control was not run.
    """

    run: str
    identity_max_abs_logprob_diff: float | None
    identity_ok: bool | None
    random_accuracy_change: dict[PromptCondition, float]
    random_hurts: bool | None


def _series(records: Sequence[ResultRecord], metric: str) -> dict[SeriesKey, dict[int, float]]:
    extract = METRICS[metric]
    series: dict[SeriesKey, dict[int, float]] = defaultdict(dict)
    for record in records:
        series[(record.run, record.condition, record.intervention)][record.question_index] = (
            extract(record)
        )
    return dict(series)


def _aligned(where: str, *series: dict[int, float]) -> list[np.ndarray]:
    """Per-question arrays in a common question order; the series must cover the same questions."""
    if any(s.keys() != series[0].keys() for s in series):
        raise ValueError(f"{where}: series are not paired (they cover different questions)")
    order = sorted(series[0])
    return [np.array([s[i] for i in order], dtype=float) for s in series]


def gain_fractions(
    records: Sequence[ResultRecord],
    *,
    metrics: Sequence[str] = tuple(METRICS),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[GainFraction]:
    """Retained and recovered gain fractions for every run and few-shot condition.

    A fraction is skipped when its forced records are absent from ``records``.

    Raises:
        ValueError: If the series that enter one fraction cover different questions.
    """
    fractions: list[GainFraction] = []
    runs = sorted({r.run for r in records})
    for metric in metrics:
        series = _series(records, metric)
        for run in runs:
            for condition in PromptCondition:
                if condition is ZERO:
                    continue
                for kind in ("retained", "recovered"):
                    fraction = _one_fraction(
                        series, run, condition, kind, metric, n_boot, seed, confidence
                    )
                    if fraction is not None:
                        fractions.append(fraction)
    return fractions


def _one_fraction(
    series: dict[SeriesKey, dict[int, float]],
    run: str,
    condition: PromptCondition,
    kind: str,
    metric: str,
    n_boot: int,
    seed: int,
    confidence: float,
) -> GainFraction | None:
    if kind == "retained":
        forced_key = (run, condition, ForcingPlan(condition, ZERO).intervention)
        point = fraction_of_gain_retained
    else:
        forced_key = (run, ZERO, ForcingPlan(ZERO, condition).intervention)
        point = fraction_of_gain_recovered
    zero = series.get((run, ZERO, NO_INTERVENTION))
    few = series.get((run, condition, NO_INTERVENTION))
    forced = series.get(forced_key)
    if zero is None or few is None or forced is None:
        return None
    z, f, g = _aligned(f"{run} {condition} {kind} {metric}", zero, few, forced)
    gain = float(f.mean() - z.mean())
    try:
        estimate: float | None = point(float(g.mean()), float(z.mean()), float(f.mean()))
    except ValueError:
        estimate = None
    low, high, undefined = (None, None, 1.0)
    if estimate is not None:
        low, high, undefined = _bootstrap_fraction(z, f, g, n_boot, seed, confidence)
    return GainFraction(
        run=run,
        condition=condition,
        kind=kind,
        metric=metric,
        estimate=estimate,
        low=low,
        high=high,
        gain=gain,
        n=len(z),
        undefined_fraction=undefined,
    )


def _bootstrap_fraction(
    zero: np.ndarray,
    few: np.ndarray,
    forced: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float,
) -> tuple[float | None, float | None, float]:
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(zero), size=(n_boot, len(zero)))
    z, f, g = zero[idx].mean(axis=1), few[idx].mean(axis=1), forced[idx].mean(axis=1)
    gain = f - z
    defined = np.abs(gain) >= MIN_GAIN
    undefined_fraction = float(1.0 - defined.mean())
    if not defined.any():
        return None, None, undefined_fraction
    ratios = (g - z)[defined] / gain[defined]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(ratios, [alpha, 1.0 - alpha])
    return float(low), float(high), undefined_fraction


def sanity_reports(records: Sequence[ResultRecord]) -> list[SanityReport]:
    """The identity and random-forcing controls for every run.

    Raises:
        ValueError: If a control and its plain counterpart cover different questions.
    """
    by_key: dict[SeriesKey, dict[int, ResultRecord]] = defaultdict(dict)
    for record in records:
        by_key[(record.run, record.condition, record.intervention)][record.question_index] = record
    return [_sanity_for_run(run, by_key) for run in sorted({r.run for r in records})]


def _sanity_for_run(run: str, by_key: dict[SeriesKey, dict[int, ResultRecord]]) -> SanityReport:
    identity_name = ForcingPlan(ZERO, ZERO).intervention
    identity = by_key.get((run, ZERO, identity_name))
    plain_zero = by_key.get((run, ZERO, NO_INTERVENTION))
    max_diff: float | None = None
    if identity is not None and plain_zero is not None:
        if identity.keys() != plain_zero.keys():
            raise ValueError(f"{run}: identity and plain zero-shot records are not paired")
        max_diff = max(
            abs(a - b)
            for i in identity
            for a, b in zip(identity[i].option_logprobs, plain_zero[i].option_logprobs, strict=True)
        )
    random_name = ForcingPlan(ZERO, None).intervention
    changes: dict[PromptCondition, float] = {}
    for condition in PromptCondition:
        random = by_key.get((run, condition, random_name))
        plain = by_key.get((run, condition, NO_INTERVENTION))
        if random is None or plain is None:
            continue
        if random.keys() != plain.keys():
            raise ValueError(f"{run}: random and plain {condition} records are not paired")
        changes[condition] = float(
            np.mean([r.is_correct for r in random.values()])
            - np.mean([r.is_correct for r in plain.values()])
        )
    return SanityReport(
        run=run,
        identity_max_abs_logprob_diff=max_diff,
        identity_ok=None if max_diff is None else max_diff <= IDENTITY_TOLERANCE,
        random_accuracy_change=changes,
        random_hurts=all(change < 0 for change in changes.values()) if changes else None,
    )


def format_force_summary(records: Sequence[ResultRecord], *, seed: int = 0) -> str:
    """Plain-text report: gain fractions with intervals, then the sanity controls."""
    lines = [
        f"{'run':<22}{'condition':<14}{'kind':<11}{'metric':<17}"
        f"{'estimate [95% bootstrap interval]':<38}{'gain':>8}  n"
    ]
    for f in gain_fractions(records, seed=seed):
        if f.estimate is None:
            estimate = "undefined (no few-shot gain)"
        else:
            estimate = f"{f.estimate:+.3f} [{f.low:+.3f}, {f.high:+.3f}]"
            if f.undefined_fraction > 0:
                estimate += f" ({f.undefined_fraction:.0%} resamples undefined)"
        lines.append(
            f"{f.run:<22}{f.condition.value:<14}{f.kind:<11}{f.metric:<17}"
            f"{estimate:<38}{f.gain:>+8.3f}  {f.n}"
        )
    lines += ["", "Sanity controls:"]
    for report in sanity_reports(records):
        lines.append(f"{report.run}: {_describe_sanity(report)}")
    return "\n".join(lines)


def _describe_sanity(report: SanityReport) -> str:
    if report.identity_ok is None:
        identity = "identity check not run"
    else:
        verdict = "ok" if report.identity_ok else f"FAIL (> {IDENTITY_TOLERANCE:g})"
        identity = f"identity max |dlogprob| = {report.identity_max_abs_logprob_diff:.2g} {verdict}"
    if report.random_hurts is None:
        random = "random check not run"
    else:
        changes = ", ".join(f"{c.value} {d:+.3f}" for c, d in report.random_accuracy_change.items())
        verdict = "hurts, as expected" if report.random_hurts else "DOES NOT HURT"
        random = f"random experts accuracy change: {changes} ({verdict})"
    return f"{identity}; {random}"
