"""Run manifests: refuse to resume a results file under different settings.

Role in the project: experiments are resumable (see :mod:`rr.runner`), and a
resumed run appends to the results already on disk. If the settings changed in
between (forcing options, planned interventions, the model, the prompts), the new
records would silently mix with old ones produced under different settings, and
nothing in a record says which settings produced it. A manifest written next to the
results the first time a run starts, and compared on every resume, closes that gap.

The manifest holds the run name, the SHA-256 of the prompt file, and a
JSON-serializable ``settings`` mapping chosen by the experiment. It is written
before the first question is evaluated, so a crash on question 0 still leaves it.
Device is deliberately not part of the settings: resuming on a different GPU is fine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_SUFFIX = ".manifest.json"


def manifest_path_for(output_path: Path) -> Path:
    """The manifest path for a results file, e.g. ``run.jsonl`` -> ``run.manifest.json``."""
    return output_path.with_name(output_path.stem + MANIFEST_SUFFIX)


def build_manifest(run: str, prompts_path: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a run: its name, the exact prompts, and the settings that shape results.

    ``settings`` is normalized through JSON so tuples and lists compare equal after a
    round trip to disk.
    """
    digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    return _normalize({"run": run, "prompts_sha256": digest, "settings": dict(settings)})


def ensure_manifest(output_path: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest on first start; on a resume, require it to match.

    Raises:
        ValueError: If the existing manifest differs (every difference is listed), is
            unreadable, or results exist with no manifest to compare against.
    """
    path = manifest_path_for(output_path)
    expected = _normalize(dict(manifest))
    if path.exists():
        differences = _diff(_read(path), expected, "")
        if differences:
            raise ValueError(
                f"{path} was written with different settings; refusing to resume "
                "(delete the results and manifest to start over):\n  " + "\n  ".join(differences)
            )
        return
    if output_path.exists() and output_path.stat().st_size > 0:
        raise ValueError(
            f"{output_path} has results but no manifest ({path.name}), so the settings that "
            "produced them are unknown; delete the results to start over"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"manifest {path} is unreadable: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must hold a JSON object")
    return data


def _diff(old: Any, new: Any, path: str) -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        differences: list[str] = []
        for key in sorted(old.keys() | new.keys()):
            where = f"{path}.{key}" if path else key
            if key not in old:
                differences.append(f"{where}: added ({new[key]!r})")
            elif key not in new:
                differences.append(f"{where}: removed (was {old[key]!r})")
            else:
                differences.extend(_diff(old[key], new[key], where))
        return differences
    return [] if old == new else [f"{path}: was {old!r}, now {new!r}"]
