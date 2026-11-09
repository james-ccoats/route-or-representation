"""Phase 3 reverse check: zero-shot prompt with few-shot routing forced in.

For each few-shot condition this measures the fraction of the few-shot gain recovered
when that condition's routing is forced onto the zero-shot prompt's target-question
tokens (a low value means routing alone does not carry the gain).

Resumable: results are appended per question to ``<output_dir>/<run>.jsonl``.

Usage: uv run python scripts/04_reverse_check.py --config configs/experiments/reverse_check.yaml
"""

from rr.force_cli import reverse_check_plans, run_forcing_script


def main() -> None:
    run_forcing_script(__doc__ or "", reverse_check_plans)


if __name__ == "__main__":
    main()
