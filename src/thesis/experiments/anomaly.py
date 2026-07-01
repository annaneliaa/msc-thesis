"""
Anomaly detection experiment: one-class model trained on benign traffic only.

Split (temporal):
  Mine window  [0,   mine_frac)   — if schema_name includes "symbolic"
  Train        [0,   1-test_frac) — benign alert_groups only
  Test         [1-test_frac, 1)   — all alert_groups (mixed benign + attack)

The difference from classifier experiments:
  - Only benign-labelled training alert_groups are passed to the model fit.
  - Models score each test alert_group; lower decision_function output = more anomalous.
  - Anomaly score = -decision_function(X_test) so higher = more likely attack.
  - AUC, F1, precision, recall are computed against the true attack labels.

Supported models:
  bernoulli_oc   — Bernoulli log-likelihood; analytic SHAP values; fast and interpretable.
  autoencoder_oc — MLP reconstruction error; per-feature error attribution as SHAP proxy.
  ocsvm          — One-class SVM with RBF kernel; no SHAP output.

For schema_name="base+symbolic", frequent itemsets are mined from the benign
portion of the training window and registered as a symbolic schema before
encoding — same mining code as the classifier symbolic experiment.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from thesis.config import load_mining_filter_config
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.experiments.baseline import (
    ALERTBERT_METHOD,
    _EXPERIMENTS_DIR,
    _ROOT,
)
from thesis.pipeline.pipeline import (
    convert_ait_alerts_to_json,
    ensure_feature_manifest,
    ingest_ait_alert_batch,
    load_or_build_alert_groups,
)
from thesis.experiments.symbolic import _mine_and_register_symbolic_schema
from thesis.features.schema_registry import FeatureSchemaRegistry
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.experiments import AnomalyExperimentConfig, ExperimentResult
from thesis.training.model_factory import get_model_factory

_ANOMALY_MODELS = {"bernoulli_oc", "autoencoder_oc", "ocsvm"}


def _encode_for_anomaly(
    scenario: str,
    alert_groups: list,
    schema_name: str,
    feature_selection=None,
    filter_attack_leaning: bool = False,
) -> tuple[pd.DataFrame, object]:
    """Encode alert_groups without caching (avoids order/schema conflicts with classifier runs)."""
    from thesis.features.util import select_symbolic_features

    registry = FeatureSchemaRegistry(root_dir=_ROOT / "artifacts" / "features")
    schema = registry.load(
        scenario_name=scenario,
        schema_name=schema_name,
        schema_version=None,
    )

    if filter_attack_leaning and schema.symbolic is not None:
        n_before = len(schema.symbolic.features)
        filtered_features = [
            f for f in schema.symbolic.features if f.source_label != "attack"
        ]
        n_dropped = n_before - len(filtered_features)
        if n_dropped:
            print(
                f"  [anomaly] Dropped {n_dropped} attack-leaning symbolic features ({len(filtered_features)} kept)"
            )
            schema = replace(
                schema, symbolic=replace(schema.symbolic, features=filtered_features)
            )

    if feature_selection is not None and (
        feature_selection.top_k is not None
        or feature_selection.min_utility_score is not None
        or feature_selection.filter_cross_host_or
    ):
        schema = select_symbolic_features(schema, feature_selection)

    feature_df = encode_alert_groups_for_schema(
        alert_groups=alert_groups,
        schema=schema,
        top_k=None,
    )
    meta_df = pd.DataFrame(
        [
            {
                "alert_group_id": t.alert_group_id,
                "group_label": t.group_label,
                "n_alerts": t.n_alerts,
            }
            for t in alert_groups
        ]
    )
    df = pd.concat(
        [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1,
    )
    return df, schema


def _compute_anomaly_metrics(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit model on benign training data and evaluate on mixed test set."""
    model.fit(X_train)

    scores = -model.decision_function(X_test)  # higher = more anomalous
    y_pred = (model.predict(X_test) == -1).astype(int)  # 1 = anomaly = attack

    if len(np.unique(y_test)) < 2:
        return {"single_class_test": True}

    auc = float(roc_auc_score(y_test, scores))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    ba = float(balanced_accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # --- Native model importances ("by_coefficient") ---
    # BernoulliOneClass: log odds ratio per feature (log(p/(1-p)))
    # BinaryAutoencoder: mean per-feature reconstruction error on benign training data
    native_importances: dict = {}
    if hasattr(model, "log_odds_"):
        pairs = sorted(
            zip(feature_names, model.log_odds_),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        native_importances = {name: {"importance": float(v)} for name, v in pairs}
    elif hasattr(model, "train_feature_errors_"):
        pairs = sorted(
            zip(feature_names, model.train_feature_errors_),
            key=lambda kv: kv[1],
            reverse=True,
        )
        native_importances = {
            name: {"importance": float(v)} for name, v in pairs if v > 0
        }

    # --- SHAP importances ("by_shap") ---
    # Models that implement shap_values() provide analytic or pseudo-SHAP attributions.
    # OneClassSVM has no such method and SHAP is skipped (KernelExplainer is impractical
    # for 1000+ binary features with an SVM scoring function).
    shap_importances: dict = {}
    if hasattr(model, "shap_values"):
        x_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        x_explain = x_arr[:200] if len(x_arr) > 200 else x_arr
        try:
            sv = model.shap_values(x_explain)
            mean_signed = np.asarray(sv).mean(axis=0)
            pairs = sorted(
                zip(feature_names, mean_signed),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            shap_importances = {name: {"importance": float(v)} for name, v in pairs}
        except Exception as shap_err:
            print(f"  [warn] shap_values() failed ({shap_err}); no SHAP importances")
    else:
        print(
            f"  [info] SHAP skipped for {type(model).__name__} "
            "(no shap_values() method; use bernoulli_oc or autoencoder_oc for SHAP)"
        )

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": ba,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_train_benign": int(len(X_train)),
        "top_feature_importances": {
            "by_coefficient": native_importances,
            "by_shap": shap_importances,
        },
    }


def run_anomaly_experiment(config: AnomalyExperimentConfig) -> ExperimentResult:
    ensure_artifact_dirs()
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    do_symbolic = "symbolic" in config.schema_name

    print(
        f"\n[Anomaly:{config.model_name}] Scenario: '{config.scenario}' schema='{config.schema_name}'"
    )

    # 1. Alerts → JSON
    print("[1/6] Converting alerts to JSON...")
    alerts_path = convert_ait_alerts_to_json(config.scenario, config.alerts_json_path)

    # 2. Process alert batch
    print("[2/6] Processing alert batch...")
    ingest_ait_alert_batch(
        config.scenario,
        alerts_path,
        config.cache_dir,
        grouping_mode=config.grouping.mode,
        grouping=config.grouping,
    )

    # 3. Feature manifest
    print("[3/6] Checking feature manifest...")
    ensure_feature_manifest(config.scenario)

    # 4. AlertGroups (always temporal order)
    print("[4/6] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(config.scenario, config.cache_dir)
    alert_groups.sort(key=lambda t: t.start_ts or "")

    n_total = len(alert_groups)
    # n_test_start = int((1 - config.test_frac) * n_total)
    n_mine = int(config.mine_frac * n_total) if config.mine_frac < 1.0 else n_total

    # 5. Mine symbolic schema if needed
    symbolic_schema_path: Path | None = None
    mining_run_dir: Path | None = None
    if do_symbolic:
        print("[5/6] Mining alert_groups for symbolic schema...")

        feature_selection = None
        if config.filter_config is not None:
            resolved = (
                config.filter_config
                if config.filter_config.is_absolute()
                else _ROOT / config.filter_config
            )
            mining_filters = load_mining_filter_config(resolved)
            feature_selection = mining_filters.feature_selection

        if config.prebuilt_symbolic_schema_path is not None:
            print(
                f"  [skip] Reusing prebuilt schema: {config.prebuilt_symbolic_schema_path}"
            )
            symbolic_schema_path = config.prebuilt_symbolic_schema_path
            mining_stats: dict = {"prebuilt": True}
        else:
            alert_groups_path = (
                config.cache_dir / "alert_groups" / "alert_groups_raw.json"
            )
            mine_path = alert_groups_path
            if config.mine_frac < 1.0:
                mine_path = (
                    alert_groups_path.parent
                    / f"alert_groups_mine_{config.mine_frac}_anomaly.json"
                )
                mine_path.write_text(
                    json.dumps(
                        [
                            {
                                "alert_group_id": t.alert_group_id,
                                "group_id": t.group_id,
                                "method": t.method,
                                "start_ts": t.start_ts,
                                "end_ts": t.end_ts,
                                "n_alerts": t.n_alerts,
                                "alert_ids": t.alert_ids,
                                "abs_items": sorted(list(t.abs_items)),
                                "raw_items": sorted(list(t.raw_items))
                                if t.raw_items is not None
                                else None,
                                "sorted_items": [sorted(s) for s in t.sorted_items],
                                "alert_ips": sorted(list(t.alert_ips)),
                                "group_label": t.group_label,
                                "alert_labels": sorted(list(t.alert_labels))
                                if t.alert_labels is not None
                                else None,
                                "weight": t.weight,
                            }
                            for t in alert_groups[:n_mine]
                        ]
                    )
                )

            symbolic_schema_path, mining_run_dir, mining_stats = (
                _mine_and_register_symbolic_schema(
                    scenario=config.scenario,
                    alert_groups_path=mine_path,
                    run_name=f"anomaly_symbolic_{config.scenario}",
                    min_support=0.05,
                    max_itemset_size=3,
                    max_seq_len=5,
                    target_label="benign",
                    filter_config=config.filter_config,
                    abstraction_map_path=config.abstraction_map_path,
                    abstraction_level=config.abstraction_level,
                    run_attack_pass=False,
                )
            )
    else:
        print("[5/6] Skipping mining (base schema).")
        mining_stats = {}
        feature_selection = None

    # 6. Encode + split + fit + evaluate
    print(f"[6/6] Encoding, fitting, and evaluating {config.model_name}...")
    try:
        df, schema = _encode_for_anomaly(
            config.scenario,
            alert_groups,
            config.schema_name,
            feature_selection=feature_selection if do_symbolic else None,
            filter_attack_leaning=config.filter_attack_leaning
            if do_symbolic
            else False,
        )
    except Exception as e:
        print(f"  [error] Failed to encode: {e}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir = config.results_dir or _EXPERIMENTS_DIR / config.scenario
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"anomaly_{config.model_name}_{timestamp}.json"
        with results_file.open("w") as f:
            json.dump(
                {"experiment": "anomaly", "scenario": config.scenario, "error": str(e)},
                f,
            )
        return ExperimentResult(
            scenario=config.scenario,
            model_name=config.model_name,
            model_version=config.model_version,
            schema_name=config.schema_name,
            schema_version="error",
            auc=float("nan"),
            n_alert_groups=n_total,
            n_features=0,
            metrics={"error": str(e)},
            results_file=results_file,
            grouping_mode=config.grouping.mode,
        )

    feature_names = schema.feature_names()
    label_map = {"benign": 0, "attack": 1}
    y = df["group_label"].map(label_map)
    X = df[feature_names].fillna(0)

    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    n_actual = len(X)
    split = int((1 - config.test_frac) * n_actual)
    if split <= 0 or split >= n_actual:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir = config.results_dir or _EXPERIMENTS_DIR / config.scenario
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"anomaly_{config.model_name}_{timestamp}.json"
        with results_file.open("w") as f:
            json.dump(
                {
                    "experiment": "anomaly",
                    "scenario": config.scenario,
                    "metrics": {"single_class_split": True},
                },
                f,
            )
        return ExperimentResult(
            scenario=config.scenario,
            model_name=config.model_name,
            model_version=config.model_version,
            schema_name=config.schema_name,
            schema_version="skipped",
            auc=float("nan"),
            n_alert_groups=n_actual,
            n_features=len(feature_names),
            metrics={"single_class_split": True},
            results_file=results_file,
            grouping_mode=config.grouping.mode,
        )

    X_train_all = X.iloc[:split]
    y_train_all = y.iloc[:split].to_numpy()
    X_test = X.iloc[split:]
    y_test = y.iloc[split:].to_numpy().astype(int)

    # Train only on benign
    benign_mask = y_train_all == 0
    X_train_benign = X_train_all[benign_mask]
    n_benign_train = int(benign_mask.sum())
    print(
        f"  Training on {n_benign_train} benign alert_groups (of {split} total in train window)"
    )

    if n_benign_train == 0:
        print("  [skip] No benign alert_groups in training window.")
        full_metrics = {"single_class_split": True}
        auc_val = float("nan")
    else:
        model = get_model_factory(config.model_name)()
        full_metrics = _compute_anomaly_metrics(
            model, X_train_benign, X_test, y_test, feature_names
        )
        auc_val = full_metrics.get("auc", float("nan"))
        print(f"  AUC: {auc_val:.4f}  F1: {full_metrics.get('f1', float('nan')):.4f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = config.results_dir or _EXPERIMENTS_DIR / config.scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"anomaly_{config.model_name}_{timestamp}.json"

    exp_label = "anomaly_symbolic" if do_symbolic else "anomaly_base"
    grouping_params = (
        config.grouping.alertbert.model_dump()
        if config.grouping.mode == ALERTBERT_METHOD
        else None
    )
    with results_file.open("w") as f:
        json.dump(
            {
                "experiment": exp_label,
                "scenario": config.scenario,
                "timestamp": timestamp,
                "model_name": config.model_name,
                "schema_name": config.schema_name,
                "grouping": {"mode": config.grouping.mode, "params": grouping_params},
                "n_alert_groups": n_actual,
                "n_train_total": split,
                "n_train_benign": n_benign_train,
                "n_test": int(len(X_test)),
                "n_features": len(feature_names),
                "test_frac": config.test_frac,
                **({"mining": mining_stats} if do_symbolic else {}),
                "metrics": full_metrics,
            },
            f,
            indent=2,
        )

    print(f"  Results → {results_file}")

    return ExperimentResult(
        scenario=config.scenario,
        model_name=config.model_name,
        model_version=config.model_version,
        schema_name=config.schema_name,
        schema_version="anomaly",
        auc=auc_val,
        n_alert_groups=n_actual,
        n_features=len(feature_names),
        metrics=full_metrics,
        results_file=results_file,
        grouping_mode=config.grouping.mode,
        symbolic_schema_path=symbolic_schema_path,
        mining_run_dir=mining_run_dir,
    )
