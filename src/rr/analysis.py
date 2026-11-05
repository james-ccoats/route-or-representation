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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

MIN_GAIN = 1e-9  # smallest |F - Z| for which the ratio is considered defined


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


def load_results(results_dir: Path) -> list[Mapping[str, Any]]:
    """Load per-question result records written by the experiment scripts.

    Args:
        results_dir: Directory such as ``results/force_routing``.

    Returns:
        A list of records (one per question x condition x intervention).
    """
    # TODO(Phase 3): define the on-disk record schema (JSONL) and parse it.
    raise NotImplementedError("TODO(Phase 3): load result records")


def plot_gain_fractions(fractions_by_pair: Mapping[str, float], output_path: Path) -> None:
    """Bar chart of gain fractions per domain pair/direction.

    Args:
        fractions_by_pair: Mapping from ``"pair:direction"`` to a fraction.
        output_path: Where to save the figure (under ``figures/``).
    """
    # TODO(Phase 3): matplotlib bar chart with bootstrap CIs; order by domain
    # distance (Phase 4 relates the fraction to distance and direction).
    raise NotImplementedError("TODO(Phase 3): plot gain fractions")
