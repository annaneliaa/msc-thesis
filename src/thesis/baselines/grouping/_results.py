"""
Save/load helpers for the grouping baseline scripts, mirroring
thesis.baselines._results's sibling pattern (RESULTS_DIR next to the
scripts, notebook-friendly load with a clear "run the script first" error)
but rows-shaped rather than per-condition-shaped -- each grouping script
produces one row per (method, param, scenario) setting via
thesis.baselines.grouping._metrics.evaluate, not a handful of named
conditions, so this is a genuinely different shape from
thesis.baselines._results and not reused from there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

_SIZE_KEY_SEP = "||"


def save_grouping_results(
    name: str,
    description: str,
    rows: list[dict],
    size_arrays: dict[tuple, np.ndarray] | None = None,
) -> Path:
    """
    Writes results/{name}.json ({"name", "description", "rows": [...]}).
    If size_arrays is given (raw per-setting group-size arrays, keyed by
    (method, param, scenario)), also writes results/{name}_sizes.npz --
    needed for the final boxplot, which needs full distributions, not just
    the mean/median/max already in each row.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"name": name, "description": description, "rows": rows}, f, indent=2)
    print(f"Results written to {out_path} ({len(rows)} rows)")

    if size_arrays:
        sizes_path = RESULTS_DIR / f"{name}_sizes.npz"
        np.savez_compressed(
            sizes_path,
            **{
                _SIZE_KEY_SEP.join(map(str, key)): arr
                for key, arr in size_arrays.items()
            },
        )
        print(
            f"Group-size arrays written to {sizes_path} ({len(size_arrays)} settings)"
        )

    return out_path


def load_grouping_results(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python {name}.py` from "
            "src/thesis/baselines/grouping/ first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["rows"])


def load_group_size_arrays(name: str) -> dict[tuple, np.ndarray]:
    """{} if {name}_sizes.npz doesn't exist -- not every artifact has one
    (e.g. cscas_grouping_sensitivity, deepcase_manual_review)."""
    sizes_path = RESULTS_DIR / f"{name}_sizes.npz"
    if not sizes_path.exists():
        return {}
    with np.load(sizes_path) as npz:
        return {tuple(key.split(_SIZE_KEY_SEP)): npz[key] for key in npz.files}
