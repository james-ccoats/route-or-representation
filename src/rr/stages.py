"""Ordered experiment stages, so one command can run the numbered scripts in sequence.

Role in the project: ``scripts/run_all.py`` is a thin wrapper over this module. Each
stage is a list of ``(script, config)`` steps run as subprocesses in order, stopping at
the first failure. The scripts are resumable, so rerunning a stage after a crash or
preemption continues where it stopped.

Methodological note: the pilot is a decision gate (research_plan.md, Phase 2). If
cross-domain few-shot never beats zero-shot, or relevant examples do not shift routing
more than irrelevant ones, the pairs, shots or model should be revisited before the
main experiment. ``all`` therefore stops after the pilot stages unless the caller
explicitly passes ``continue_past_gates``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path("scripts")
EXPERIMENTS = Path("configs/experiments")
PAIRS = Path("configs/pairs")


@dataclass(frozen=True)
class Step:
    """One script invocation: ``<script> --config <config>``."""

    script: str
    config: Path

    def command(self, python: str = sys.executable) -> list[str]:
        return [python, str(SCRIPTS / self.script), "--config", str(self.config)]


def _step(script: str, config: Path) -> Step:
    return Step(script, config)


STAGES: dict[str, tuple[Step, ...]] = {
    "pilot": (
        _step("01_build_prompts.py", EXPERIMENTS / "pilot.yaml"),
        _step("02_run_pilot.py", EXPERIMENTS / "pilot.yaml"),
    ),
    "ladder": (
        _step("06_measure_distance.py", PAIRS / "mmlu_ladder.yaml"),
        _step("01_build_prompts.py", EXPERIMENTS / "ladder_pilot.yaml"),
        _step("02_run_pilot.py", EXPERIMENTS / "ladder_pilot.yaml"),
    ),
    "main": (
        _step("01_build_prompts.py", EXPERIMENTS / "force_routing.yaml"),
        _step("03_force_routing.py", EXPERIMENTS / "force_routing.yaml"),
        _step("04_reverse_check.py", EXPERIMENTS / "reverse_check.yaml"),
    ),
    # Reuses the prompts that the ladder stage built, so there is no second build step.
    "ladder-forcing": (
        _step("03_force_routing.py", EXPERIMENTS / "ladder_force_routing.yaml"),
        _step("04_reverse_check.py", EXPERIMENTS / "ladder_reverse_check.yaml"),
    ),
}

PRE_GATE_STAGES = ("pilot", "ladder")
POST_GATE_STAGES = ("main", "ladder-forcing")
ALL = "all"


class StageError(ValueError):
    """Raised for an unknown stage name."""


def plan_stages(stage: str, continue_past_gates: bool = False) -> tuple[Step, ...]:
    """Return the ordered steps for ``stage`` (a stage name or ``"all"``).

    ``"all"`` is the pilot stages, then the main stages only when
    ``continue_past_gates`` is true. A single named stage always runs in full.
    """
    if stage == ALL:
        names = PRE_GATE_STAGES + (POST_GATE_STAGES if continue_past_gates else ())
    elif stage in STAGES:
        names = (stage,)
    else:
        raise StageError(f"unknown stage {stage!r}; choose from {[*STAGES, ALL]}")
    return tuple(step for name in names for step in STAGES[name])


def run_steps(
    steps: Sequence[Step],
    *,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    echo: Callable[[str], None] = print,
) -> int:
    """Run steps in order from the repo root; stop at the first failure.

    Returns:
        0 when every step succeeded (or ``dry_run``), else the failing step's exit code.
    """
    for index, step in enumerate(steps, start=1):
        command = step.command()
        echo(f"[{index}/{len(steps)}] {' '.join(command)}")
        if dry_run:
            continue
        result = runner(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            echo(f"step {index} failed with exit code {result.returncode}; stopping")
            return result.returncode
    return 0
