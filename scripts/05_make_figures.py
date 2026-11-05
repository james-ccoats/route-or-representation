"""Make figures from saved results.

Usage: uv run python scripts/05_make_figures.py --config configs/experiments/force_routing.yaml
"""

from pathlib import Path

from rr.analysis import load_results, plot_gain_fractions
from rr.config import load_experiment_config, parse_config_arg

FIGURES_DIR = Path("figures")


def main() -> None:
    cfg = load_experiment_config(parse_config_arg(__doc__ or ""))
    records = load_results(cfg.output_dir)
    # TODO(Phase 3): aggregate `records` into {"pair:direction": fraction}.
    fractions: dict[str, float] = {}
    plot_gain_fractions(fractions, FIGURES_DIR / f"{cfg.name}_gain_fractions.png")
    _ = records


if __name__ == "__main__":
    main()
