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
    metrics: dict[str, float] | None = None,
    *,
    seeds: list[dict[str, float]] | None = None,
    workload: dict[str, dict | None] | None = None,
) -> Path:
    """Save the anomaly-detector baseline's flat {auc, precision, recall,
    f1} -- no pool conditions, same "no three-condition structure" shape as
    save_zeroshot_results, but with `auc` included (this method's headline
    metric) and an optional per-seed breakdown.

    Pass `seeds` (a list of per-seed {auc, precision, recall, f1} dicts) to
    record the same 5-seed protocol the trainable baselines use: the
    headline auc/precision/recall/f1 become the seed mean and the raw
    per-seed dicts are persisted under "seeds" so the comparison notebook
    can report mean +/- sd. Pass `metrics` instead for a genuinely
    single-run detector (e.g. OneClassSVM, a deterministic convex fit with
    no random_state -- its "sd" is exactly 0). Exactly one of
    `metrics`/`seeds` must be given.

    `workload` (a training.workload.compute_workload_at_recall result, or a
    seed-averaged one via average_workload_at_recall) is persisted under
    "workload_at_recall": precision / FP / analyst-workload-reduction at the
    threshold that hits each target recall. The headline precision/recall/f1
    above are still at the model's own default cut (nu / contamination =
    0.05); this is the tuned-operating-point view that's comparable to the
    classifier baselines.
    """
    if (metrics is None) == (seeds is None):
        raise ValueError("pass exactly one of `metrics` or `seeds`")

    keys = ("auc", "precision", "recall", "f1")
    if seeds is not None:
        if not seeds:
            raise ValueError("`seeds` is empty")
        agg = {k: float(sum(s[k] for s in seeds) / len(seeds)) for k in keys}
    else:
        agg = {k: float(metrics[k]) for k in keys}

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"

    payload = {"name": name, "description": description, "kind": "anomaly", **agg}
    if seeds is not None:
        payload["seeds"] = [{k: float(s[k]) for k in keys} for s in seeds]
    if workload is not None:
        payload["workload_at_recall"] = workload

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
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


def anomaly_results_current(name: str, *, require_seeds: bool = False) -> bool:
    """Like results_exist(), but also checks the on-disk anomaly result is
    in the *current* format so a resumed overnight run re-does combos left
    over from an older run.

    Current format = has `workload_at_recall` (the tuned-operating-point
    view) and, when `require_seeds` (the IsolationForest scripts), a
    non-trivial `seeds` list. A stale-format file returns False so the
    caller recomputes it -- no need to hand-delete old JSONs or force a
    full re-run.
    """
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if "workload_at_recall" not in data:
        return False
    if require_seeds and len(data.get("seeds") or []) < 2:
        return False
    return True
