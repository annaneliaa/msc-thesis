"""
Shared helper for cscas.py / cscas_base.py / cscas_bert.py to persist their
averaged Baseline 1 / Baseline 2 metrics to disk, and for the comparison
notebook (notebooks/cscas_baseline_comparison.ipynb) to load them back.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def save_baseline_results(
    name: str,
    description: str,
    results_b1: list[dict[str, float]],
    results_b2: list[dict[str, float]],
) -> Path:
    import pandas as pd

    avg_b1 = pd.DataFrame(results_b1).mean()
    avg_b2 = pd.DataFrame(results_b2).mean()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "description": description,
                "baseline_1": {
                    "precision": float(avg_b1.precision),
                    "recall": float(avg_b1.recall),
                    "f1": float(avg_b1.f1),
                    "seeds": results_b1,
                },
                "baseline_2": {
                    "precision": float(avg_b2.precision),
                    "recall": float(avg_b2.recall),
                    "f1": float(avg_b2.f1),
                    "seeds": results_b2,
                },
            },
            f,
            indent=2,
        )
    print(f"Results written to {out_path}")
    return out_path


def load_baseline_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python {name}.py` from src/thesis/baselines/ first."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)
