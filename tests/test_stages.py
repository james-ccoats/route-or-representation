"""Tests for the ordered experiment stages."""

import subprocess
from pathlib import Path

import pytest

from rr.stages import REPO_ROOT, STAGES, StageError, Step, plan_stages, run_steps


def _names(steps) -> list[str]:
    return [f"{s.script}:{s.config.stem}" for s in steps]


def test_every_stage_script_and_config_exists() -> None:
    for steps in STAGES.values():
        for step in steps:
            assert (REPO_ROOT / "scripts" / step.script).is_file(), step
            assert (REPO_ROOT / step.config).is_file(), step


def test_pilot_stage_builds_prompts_before_running() -> None:
    assert _names(plan_stages("pilot")) == [
        "01_build_prompts.py:pilot",
        "02_run_pilot.py:pilot",
    ]


def test_main_stage_builds_before_forcing_and_reverse_check() -> None:
    assert _names(plan_stages("main")) == [
        "01_build_prompts.py:force_routing",
        "03_force_routing.py:force_routing",
        "04_reverse_check.py:reverse_check",
    ]


def test_all_stops_before_the_main_experiments_by_default() -> None:
    names = _names(plan_stages("all"))
    assert names == _names(plan_stages("pilot")) + _names(plan_stages("ladder"))
    assert not any(n.startswith(("03_", "04_")) for n in names)


def test_all_with_continue_runs_pilot_stages_first_then_main() -> None:
    names = _names(plan_stages("all", continue_past_gates=True))
    assert names == (
        _names(plan_stages("pilot"))
        + _names(plan_stages("ladder"))
        + _names(plan_stages("main"))
        + _names(plan_stages("ladder-forcing"))
    )


def test_ladder_forcing_comes_after_the_ladder_build() -> None:
    names = _names(plan_stages("all", continue_past_gates=True))
    assert names.index("01_build_prompts.py:ladder_pilot") < names.index(
        "03_force_routing.py:ladder_force_routing"
    )


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(StageError, match="unknown stage"):
        plan_stages("nope")


def test_command_passes_the_config_to_the_script() -> None:
    command = Step("02_run_pilot.py", Path("configs/experiments/pilot.yaml")).command("py")

    assert command == [
        "py",
        "scripts/02_run_pilot.py",
        "--config",
        "configs/experiments/pilot.yaml",
    ]


def _fake_runner(codes: list[int], calls: list[list[str]]):
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, codes[len(calls) - 1])

    return run


def test_run_steps_stops_at_the_first_failure() -> None:
    calls: list[list[str]] = []

    code = run_steps(
        plan_stages("main"), runner=_fake_runner([0, 3, 0], calls), echo=lambda _: None
    )

    assert code == 3
    assert len(calls) == 2


def test_run_steps_returns_zero_when_every_step_succeeds() -> None:
    calls: list[list[str]] = []

    code = run_steps(plan_stages("pilot"), runner=_fake_runner([0, 0], calls), echo=lambda _: None)

    assert code == 0
    assert len(calls) == 2


def test_dry_run_executes_nothing() -> None:
    calls: list[list[str]] = []
    lines: list[str] = []

    code = run_steps(
        plan_stages("pilot"), dry_run=True, runner=_fake_runner([], calls), echo=lines.append
    )

    assert code == 0 and calls == [] and len(lines) == 2
