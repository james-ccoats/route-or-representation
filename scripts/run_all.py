"""Run the experiment scripts in order, one stage at a time.

Stages: pilot, ladder, main, ladder-forcing, or all. ``all`` runs the pilot and ladder
stages and then stops: check the pilot results against the Phase 2 criteria in
docs/research_plan.md, then continue with ``--stage main`` (and ``ladder-forcing``), or
rerun ``--stage all --continue-past-gates``. Every script is resumable, so rerunning a
stage after a crash continues where it stopped.

Usage: uv run python scripts/run_all.py --stage pilot [--dry-run]
"""

import argparse
import sys

from rr.stages import ALL, STAGES, plan_stages, run_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, ALL], required=True)
    parser.add_argument(
        "--continue-past-gates",
        action="store_true",
        help="with --stage all, also run the main experiments after the pilot stages",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()
    steps = plan_stages(args.stage, args.continue_past_gates)
    code = run_steps(steps, dry_run=args.dry_run)
    if code == 0 and args.stage == ALL and not args.continue_past_gates:
        print("Stopped before the main experiments: check the pilot results first.")
    return code


if __name__ == "__main__":
    sys.exit(main())
