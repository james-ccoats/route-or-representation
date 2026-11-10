"""Measure the (model-free, lexical) distance between the source and target of every pair.

Prints each run's expected label (near / medium / far, a hypothesis) beside a measured
TF-IDF distance, flags labels that disagree with the measured ordering, and lists each
target's nearest subjects so sources can be chosen from the data. Writes the same to
``results/distance/<pairs config name>.json``. Needs MMLU but not the model.

The measure is lexical: it sees shared terminology, not shared skills or routing.
See ``rr.distance`` for what it can and cannot tell you.

Usage: uv run python scripts/06_measure_distance.py --config configs/pairs/mmlu_ladder.yaml
"""

import json
from dataclasses import asdict
from pathlib import Path

from rr.config import load_pairs_config, parse_config_arg
from rr.data import load_mmlu_texts
from rr.distance import (
    direction_distances,
    format_direction_distances,
    nearest_subjects,
    subject_distances,
)

OUTPUT_DIR = Path("results/distance")
NEAREST_COUNT = 10


def main() -> None:
    config_path = parse_config_arg(__doc__ or "")
    pairs = load_pairs_config(config_path)
    # IDF is computed over every MMLU subject, not just the ones in this config.
    distances = subject_distances(
        load_mmlu_texts(pairs.eval_split, pairs.dataset, pairs.dataset_revision)
    )
    rows = direction_distances(pairs, distances)
    print(format_direction_distances(rows))

    targets = sorted({row.target for row in rows})
    nearest = {t: nearest_subjects(distances, t, NEAREST_COUNT) for t in targets}
    for target, ranked in nearest.items():
        print(f"\nNearest subjects to {target}:")
        for name, distance in ranked:
            print(f"  {name:<34}{distance:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{config_path.stem}.json"
    output.write_text(
        json.dumps(
            {"runs": [asdict(row) for row in rows], "nearest": nearest}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
