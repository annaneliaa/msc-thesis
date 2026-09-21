"""Shared helpers across experiment modules under thesis.experiments.

Threshold decision, threshold-scored metrics, and scenario setup used to
live in temporal_decay.py (Experiment 2) only; they're promoted here now
that rolling_walk_forward.py (Experiment 3) needs the same pieces, so both
mine/fit/evaluate/aggregate the same way instead of drifting apart.
"""

from __future__ import annotations

import time
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
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.pipeline.pipeline import (
    ensure_feature_manifest,
    ingest_ait_scenario,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
)
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.training.model_factory import get_model_factory
from thesis.training.pool_sampling import class_weighted_extra_kwargs
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
    y_src: np.ndarray,
    proba_src: np.ndarray,
    mode: str,
    recall_target: float,
    model=None,
) -> float:
    """mode="fixed" -> the model's own no-tuning operating point: 0.5 for a
    supervised classifier (probability midpoint), or a one-class detector's
    own contamination cut (`_PlattScaledOneClass.default_threshold` -- the
    Platt probability at which its `predict()` flips). A flat 0.5 in
    Platt-probability space is not a meaningful cut for a one-class model on
    an imbalanced scenario -- it usually sits above every calibrated
    probability, so everything is predicted benign.

    mode="calibrated_recall" -> the threshold that achieves at least
    `recall_target` recall on the given scores (compute_workload_at_recall),
    falling back to that fixed operating point (with a warning, never
    raising) if the target isn't reachable (e.g. single-class input)."""
    fixed = float(getattr(model, "default_threshold", 0.5))
    if mode == "fixed":
        return fixed
    if mode == "calibrated_recall":
        result = compute_workload_at_recall(y_src, proba_src, targets=(recall_target,))
        entry = result.get(f"{recall_target:.2f}")
        if entry is None:
            print(
                f"    [warn] calibrated_recall target {recall_target:.2f} "
                f"unreachable -- falling back to {fixed:.3f}"
            )
            return fixed
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


_EMPTY_FUNNEL = {
    "fast_route_wall_s": None,
    "fast_route_cpu_s": None,
    "n_alerts_in": 0,
    "n_groups_escalated": None,
    "n_groups_suppressed": None,
    "n_alerts_escalated": None,
    "n_alerts_suppressed": None,
}


def fast_route_and_funnel(
    raw_rows: list[AlertGroup], schema: FeatureSchema, model, threshold: float
) -> dict:
    """System-operationality instrumentation, shared by temporal_decay.py
    (never-retrain anchor), rolling_walk_forward.py (always-retrain anchor),
    and monitor_attached.py (the monitor-gated policy between them), so the
    three land on directly comparable columns for a 3-way cost comparison.

    A single batched encode+predict call over EVERY incoming group in
    `raw_rows` -- labeled or not, since production scores every alert, not
    just the ones that later get a label -- timed with wall (perf_counter)
    and CPU (process_time) clocks. The same proba array is then thresholded
    into the escalated/suppressed workload funnel at both group count and
    raw-alert count (AlertGroup.n_alerts) granularity, with zero extra
    model calls. Deliberately a *second*, separate encode+predict pass from
    whatever labeled-only one a caller does for its own quality metrics
    (X_h/proba_h etc.) -- kept apart so this stays a pure addition on top of
    each experiment's existing masked-encoding path, not a refactor of it.

    Returns the all-None/0 shape (_EMPTY_FUNNEL) for an empty window, so a
    caller can always spread the result into a row dict without its own
    branch."""
    if not raw_rows:
        return dict(_EMPTY_FUNNEL)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    encoded = encode_alert_groups_for_schema(raw_rows, schema)
    proba = model.predict_proba(encoded)[:, 1]
    fast_route_wall_s = time.perf_counter() - t0_wall
    fast_route_cpu_s = time.process_time() - t0_cpu

    y_pred = (proba >= threshold).astype(int)
    alerts_arr = np.array([tx.n_alerts for tx in raw_rows])
    return {
        "fast_route_wall_s": fast_route_wall_s,
        "fast_route_cpu_s": fast_route_cpu_s,
        "n_alerts_in": int(alerts_arr.sum()),
        "n_groups_escalated": int(y_pred.sum()),
        "n_groups_suppressed": int((y_pred == 0).sum()),
        "n_alerts_escalated": int(alerts_arr[y_pred == 1].sum()),
        "n_alerts_suppressed": int(alerts_arr[y_pred == 0].sum()),
    }


def sample_rows(X: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    return X.sample(min(n, len(X)), random_state=random_state) if len(X) else X


# One-class anomaly detectors (thesis.training.model_factory). They fit
# unsupervised (`fit(X)`, no labels) on benign traffic and expose
# `decision_function` / `predict` (-1/+1), not `predict_proba`.
ONE_CLASS_MODELS = frozenset({"iforest", "ocsvm", "bernoulli_oc", "autoencoder_oc"})


class _PlattScaledOneClass:
    """Adapts a fitted one-class anomaly detector to the binary-classifier
    interface the experiments expect (`predict_proba(X)[:, 1]` = attack
    likelihood).

    The detector is trained unsupervised on benign rows only. This wrapper
    adds a 1-D logistic (Platt) calibration of its signed anomaly score
    (`-decision_function`, higher = more anomalous) against the labels that
    *are* available on W_src's train split -- fit once and frozen. So a
    Platt-scaled one-class model drops into the exact same threshold /
    metric / SHAP (via `predict_proba`) / LIME (classification mode) path as
    logreg or xgboost, with no parallel anomaly-scoring branch.

    `default_threshold` is the Platt probability at which the underlying
    detector's own `predict()` flips (its `contamination` cut) -- the
    label-free operating point `decide_threshold(mode="fixed")` uses for a
    one-class model, since a flat 0.5 in this probability space usually
    predicts everything benign on an imbalanced scenario.
    """

    def __init__(self, inner):
        self.inner = inner
        self._platt = None
        self.default_threshold = 0.5

    def _raw(self, X) -> np.ndarray:
        return -np.asarray(self.inner.decision_function(X), dtype=float)

    def calibrate(self, X, y) -> "_PlattScaledOneClass":
        from sklearn.linear_model import LogisticRegression

        raw = self._raw(X)
        self._platt = LogisticRegression().fit(raw.reshape(-1, 1), np.asarray(y))

        # Where does inner.predict() flip on this same data? Midpoint between
        # the highest raw score it still calls normal and the lowest it calls
        # anomalous -- works whatever the detector's internal convention
        # (threshold_ attribute, decision_function sign, ...). Map that raw
        # boundary through the (monotonic) Platt scaler to a probability, so
        # `proba >= default_threshold` reproduces inner.predict() == -1.
        flagged = np.asarray(self.inner.predict(X)) == -1
        if flagged.any() and (~flagged).any():
            boundary = 0.5 * (raw[flagged].min() + raw[~flagged].max())
        else:  # degenerate (all/none flagged) -- fall back to the score's own tail
            boundary = float(np.quantile(raw, 0.95))
        self.default_threshold = float(self._platt.predict_proba([[boundary]])[0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._platt.predict_proba(self._raw(X).reshape(-1, 1))


def fit_scored_model(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray):
    """Fit `model_name` on W_src's train split; return an object exposing
    `predict_proba(X)[:, 1]` as attack likelihood.

    Supervised models fit on the whole split, class-imbalance-aware: every
    one is handed `class_weighted_extra_kwargs` (class_weight="balanced" for
    sklearn, scale_pos_weight for xgboost/torch_nn), so a fixed 0.5
    threshold is a meaningful operating point for all of them -- not just
    logreg/rf, which hardcode "balanced" and ignore the kwargs. Without
    this, an unweighted xgboost on this scenario's ~1.5% attack rate sits
    at a low-recall corner at 0.5, not comparable to logreg's.

    One-class models (`ONE_CLASS_MODELS`) fit unsupervised on the benign
    rows only, then get a frozen Platt scaler over the labeled split (see
    `_PlattScaledOneClass`).

    Returns None (never raises) when the split can't support a fit -- a
    supervised model needs both classes present, a one-class model needs
    enough benign rows to fit and at least one attack row to calibrate.
    """
    y_train = np.asarray(y_train)

    if model_name not in ONE_CLASS_MODELS:
        if len(np.unique(y_train)) < 2:
            return None
        est = get_model_factory(model_name, **class_weighted_extra_kwargs(y_train))()
        est.fit(X_train, y_train)
        return est

    est = get_model_factory(model_name)()
    benign = y_train == 0
    if benign.sum() < 10 or (y_train == 1).sum() < 1:
        return None
    est.fit(X_train[benign])
    return _PlattScaledOneClass(est).calibrate(X_train, y_train)


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
