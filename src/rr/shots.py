"""Run names for the (direction, replicate, shot count) grid.

Role in the project: every combination of domain direction, example-set replicate and
shot count ``k`` is one *run*: its own prompt file, results file and manifest. That keeps
the runner and the per-run analysis unchanged; what is new is naming runs so a later
step can group them back together (gain against ``k``, spread across replicates).

A run stem looks like ``bio_med_a2b_r1_k3``: direction ``bio_med_a2b``, replicate 1,
k = 3 examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STEM = re.compile(r"^(?P<direction>.+)_r(?P<replicate>\d+)_k(?P<shots>\d+)$")


@dataclass(frozen=True)
class ShotRun:
    """The parts of a run stem."""

    direction: str
    replicate: int
    shots: int

    @property
    def stem(self) -> str:
        return f"{self.direction}_r{self.replicate}_k{self.shots}"


def run_stem(direction: str, replicate: int, shots: int) -> str:
    """The run name for one direction, replicate and shot count.

    ``direction`` may use the ``pair:a2b`` form of :meth:`rr.config.PairsConfig.directions`;
    the colon is replaced so the stem is a safe file name.
    """
    return ShotRun(direction.replace(":", "_"), replicate, shots).stem


def parse_run_stem(stem: str) -> ShotRun | None:
    """Split a run stem into its parts, or ``None`` if it is not a shot-grid run name."""
    match = _STEM.match(stem)
    if match is None:
        return None
    return ShotRun(
        direction=match["direction"],
        replicate=int(match["replicate"]),
        shots=int(match["shots"]),
    )
