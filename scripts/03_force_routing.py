"""Phase 3: few-shot prompt with zero-shot routing forced on target-question tokens.

For each few-shot condition this measures the fraction of the few-shot gain retained
when the zero-shot routing is forced onto the target-question tokens (a high value
means the gain survives without the routing change, i.e. it is carried by the
hidden-state shift). It also runs the identity and random-expert sanity controls.

Resumable: results are appended per question to ``<output_dir>/<run>.jsonl``, so
re-running the same command after a crash or preemption continues where it stopped.

Usage: uv run python scripts/03_force_routing.py --config configs/experiments/force_routing.yaml
"""

from rr.force_cli import force_routing_plans, run_forcing_script


def main() -> None:
    run_forcing_script(__doc__ or "", force_routing_plans)


if __name__ == "__main__":
    main()
