"""Shared helpers across experiment modules under thesis.experiments.

Threshold decision, threshold-scored metrics, and scenario setup used to
live in temporal_decay.py (Experiment 2) only; they're promoted here now
that rolling_walk_forward.py (Experiment 3) needs the same pieces, so both
mine/fit/evaluate/aggregate the same way instead of drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from thesis.config import load_mining_settings
from thesis.configs import dataset_for_scenario, load_base_features
from thesis.pipeline.pipeline import (
    ensure_feature_manifest,
    ingest_ait_scenario,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
)
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.training.workload import compute_workload_at_recall

_ROOT = Path(__file__).resolve().parents[3]

LABEL_MAP = {"benign": 0.0, "attack": 1.0}

METRIC_COLS = ["auc", "f1", "accuracy", "precision", "recall", "fpr"]
CONFIG_COLS = ["feature_set", "mining_setting", "granularity", "model"]


def labels_and_mask(window_rows: list[AlertGroup]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row 0/1 label and a mask of which rows carry a usable label
    (drops unlabelled/mixed alert_groups)."""
    labels = np.array(
        [LABEL_MAP.get(t.group_label, np.nan) for t in window_rows], dtype=float
    )
    return labels, ~np.isnan(labels)


def decide_threshold(
    y_src: np.ndarray, proba_src: np.ndarray, mode: str, recall_target: float
) -> float:
    """mode="fixed" -> 0.5. mode="calibrated_recall" -> the threshold that
    achieves at least `recall_target` recall on the given scores
    (compute_workload_at_recall), falling back to 0.5 (with a warning, never
    raising) if that target isn't reachable (e.g. single-class input)."""
    if mode == "fixed":
        return 0.5
    if mode == "calibrated_recall":
        result = compute_workload_at_recall(y_src, proba_src, targets=(recall_target,))
        entry = result.get(f"{recall_target:.2f}")
        if entry is None:
            print(
                f"    [warn] calibrated_recall target {recall_target:.2f} "
                "unreachable -- falling back to 0.5"
            )
            return 0.5
        return float(entry["threshold"])
    raise ValueError(f"unknown threshold_mode {mode!r}")


def metrics_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict:
    """Metrics at a caller-supplied (frozen) threshold -- deliberately not
    train.train_eval_holdout (hardcodes 0.5, computes importances/SHAP
    handled separately) or evaluation.eval_subset_metrics (assumes the same
    window a fitted result's test split came from); callers here score the
    same fitted model against a different window."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else np.nan

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": fpr,
    }


def nan_metrics() -> dict:
    return {
        "auc": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "fpr": np.nan,
    }


def sample_rows(X: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    return X.sample(min(n, len(X)), random_state=random_state) if len(X) else X


@dataclass(slots=True)
class ScenarioContext:
    """Everything downstream of "ingest + build the feature manifest" that
    doesn't depend on which shortlisted config is being run -- shared by
    every experiment module that consumes a shortlist (temporal_decay.py,
    rolling_walk_forward.py) and the on-demand case-study CLI
    (scripts/mining/explain_instances.py), so all of them set up the same
    scenario the same way."""

    alert_groups: list
    alert_groups_path: Path
    n_total: int
    base_schema: FeatureSchema
    mining_settings_by_name: dict
    mining_settings_path: Path


def load_scenario_context(
    scenario: str,
    cache_dir: Path,
    grouping,
    alerts_json_path: Path | None,
    mining_settings_path: Path,
) -> ScenarioContext:
    is_cscas = dataset_for_scenario(scenario) == "cscas"

    print(f"\n[ScenarioContext] Scenario: '{scenario}'")

    print("[1/4] Ingesting scenario...")
    if is_cscas:
        ingest_cscas_scenario(cache_dir=cache_dir)
    else:
        ingest_ait_scenario(
            scenario,
            alerts_json_path=alerts_json_path,
            cache_dir=cache_dir,
            grouping=grouping,
        )

    print("[2/4] Checking feature manifest...")
    ensure_feature_manifest(scenario)

    print("[3/4] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(scenario, cache_dir)
    alert_groups_path = cache_dir / "alert_groups" / "alert_groups_raw.json"
    alert_groups.sort(key=lambda t: t.start_ts or "")
    n_total = len(alert_groups)
    print(f"  {n_total} alert_groups total")

    dataset = dataset_for_scenario(scenario)
    if dataset is None:
        raise ValueError(
            f"Scenario '{scenario}' is not listed under any dataset in scenarios.json."
        )
    base_schema = FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(load_base_features(dataset)),
        symbolic=None,
    )

    if not mining_settings_path.is_absolute():
        mining_settings_path = _ROOT / mining_settings_path
    mining_settings_by_name = {
        s.name: s for s in load_mining_settings(mining_settings_path)
    }

    return ScenarioContext(
        alert_groups=alert_groups,
        alert_groups_path=alert_groups_path,
        n_total=n_total,
        base_schema=base_schema,
        mining_settings_by_name=mining_settings_by_name,
        mining_settings_path=mining_settings_path,
    )
