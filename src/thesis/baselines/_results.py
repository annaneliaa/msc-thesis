"""
Shared helper for the trainable CSCAS baseline scripts (cscas.py,
cscas_base.py, cscas_bert.py, and future cscas_logreg.py / cscas_xgboost.py
/ cscas_securebert.py) to persist their averaged per-condition metrics to
disk, for cscas_zeroshot.py to persist its single-shot flat metrics, and
for the comparison notebook (notebooks/cscas_baseline_comparison.ipynb) to
load either shape back.
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def save_baseline_results(
    name: str,
    description: str,
    results: dict[str, list[dict[str, float]]],
) -> Path:
    """Save averaged per-seed metrics for every trainable-baseline
    condition, e.g. {"random": [...5 seed dicts...],
    "class_weighted": [...], "guided": [...]}. Accepts an arbitrary
    conditions dict so the number of conditions can change without a
    signature change.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"

    conditions = {}
    for condition_name, seed_results in results.items():
        avg = pd.DataFrame(seed_results).mean()
        conditions[condition_name] = {
            "precision": float(avg.precision),
            "recall": float(avg.recall),
            "f1": float(avg.f1),
            "seeds": seed_results,
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "description": description,
                "kind": "trainable",
                **conditions,
            },
            f,
            indent=2,
        )
    print(f"Results written to {out_path}")
    return out_path


def save_zeroshot_results(
    name: str,
    description: str,
    metrics: dict[str, float],
) -> Path:
    """Save zero-shot's flat {precision, recall, f1} -- no conditions, no
    seed-averaging (zero-shot runs once against the shared eval
    subsample). Deliberately not routed through save_baseline_results:
    the shape is genuinely different, not just a smaller case of it.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "description": description,
                "kind": "zero_shot",
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
            f,
            indent=2,
        )
    print(f"Results written to {out_path}")
    return out_path


def save_anomaly_results(
    name: str,
    description: str,
    metrics: dict[str, float],
) -> Path:
    """Save the anomaly-detector baseline's flat {auc, precision, recall,
    f1} -- no conditions, no seed-averaging, same "single deterministic
    run" shape as save_zeroshot_results (the anomaly model factories have
    no seed/random_state concept to average over either), but with `auc`
    included since that's this method's headline metric, not just a
    byproduct."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "description": description,
                "kind": "anomaly",
                "auc": float(metrics["auc"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
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


def is_zero_shot(data: dict) -> bool:
    """True if `data` (as returned by load_baseline_results) is a flat
    zero-shot result rather than a trainable multi-condition result."""
    return data.get("kind") == "zero_shot"


def is_anomaly(data: dict) -> bool:
    """True if `data` (as returned by load_baseline_results) is a flat
    anomaly-detector result -- same "no conditions" shape as
    is_zero_shot, but a semantically distinct kind (carries `auc`, comes
    from a benign-only-trained one-class model, not a prompted LLM)."""
    return data.get("kind") == "anomaly"


def results_exist(name: str) -> bool:
    """True if RESULTS_DIR/{name}.json already exists -- for callers that
    want to skip already-computed (condition/scenario/grouping_method/...)
    combos on a resumed run, the same way
    scripts/system_eval/run_model_comparison_attribute.py skips existing
    compare_*.json files unless --force is passed."""
    return (RESULTS_DIR / f"{name}.json").exists()
