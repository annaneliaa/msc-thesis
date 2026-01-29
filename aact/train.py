import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from schema import FeatureSchema
from build_features import build_dyn_features, build_static_features
import symbolic_features

from plots import (
    plot_roc,
    plot_alert_reduction,
    plot_feature_importance,
    plot_confidence_distribution,
    plot_top_error_categories,
)

def burst_diagnostics(X_full, res, burst_col="is_suspicious_auth_burst", auth_col="is_auth_event"):
    split = res["test_idx_start"]
    X_test_full = X_full.iloc[split:].reset_index(drop=True)
    y_test = res["y_test"]
    s = res["proba_test"]

    out = {"auc": res["auc"]}

    if burst_col in X_test_full.columns:
        burst = X_test_full[burst_col].fillna(0).astype(int).values == 1
        out["burst_count_test"] = int(burst.sum())
        out["mean_score_burst_test"] = float(s[burst].mean()) if burst.any() else np.nan
        out["mean_score_nonburst_test"] = float(s[~burst].mean())
        if burst.any() and len(np.unique(y_test[burst])) > 1:
            out["auc_burst_test"] = float(roc_auc_score(y_test[burst], s[burst]))
        else:
            out["auc_burst_test"] = np.nan

    if auth_col in X_test_full.columns:
        auth = X_test_full[auth_col].fillna(0).astype(int).values == 1
        out["auth_count_test"] = int(auth.sum())
        if auth.any() and len(np.unique(y_test[auth])) > 1:
            out["auc_auth_test"] = float(roc_auc_score(y_test[auth], s[auth]))
        else:
            out["auc_auth_test"] = np.nan

    return out

def train_eval_holdout(X_full, y, schema, test_frac=0.3):
    X_full = X_full.reset_index(drop=True)
    y = np.asarray(y)
    assert len(X_full) == len(y)

    # select schema columns
    X = X_full[schema.features]
    feature_names = list(X.columns)

    n = len(X)
    split = int((1 - test_frac) * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # guard
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Train or test has only one class. Try a different split_frac or use all scenarios.")

    model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba_test))

    return {
        "schema": schema.name,
        "model": model,
        "auc": auc,
        "y_test": y_test,
        "proba_test": proba_test,
        "test_idx_start": split,
        "feature_names": feature_names
    }

def train_and_evaluate(X, y, schema, n_splits=3, burst_col="is_suspicious_auth_burst",
                       auth_col="is_auth_event"):

    # schema: FeatureSchema specifying which columns to use for the model

    X = X.reset_index(drop=True)
    y = np.asarray(y)

    assert len(X) == len(y)

    missing = [c for c in schema.features if c not in X.columns]
    if missing:
        raise KeyError(f"Schema '{schema.name}' is missing columns: {missing}")

    X = X[schema.features]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
    )

    aucs = []
    oof_proba = np.full(shape=len(X), fill_value=np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # skip folds that don't have both classes
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[{schema.name}] Fold {fold} skipped (single-class train/test).")
            continue


        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba

        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)
        print(f"[{schema.name}] Fold {fold} ROC-AUC: {auc:.3f}")
        
    if not aucs:
        raise ValueError(f"[{schema.name}] No valid folds (all single-class). Try fewer splits or a different scenario.")

    mean_auc = float(np.mean(aucs))
    print(f"Mean ROC-AUC: {mean_auc:.3f}")

    # --- Diagnostics: burst scoring and subset AUCs ---
    diagnostics = {}

    if burst_col in X.columns:
        burst_mask = X[burst_col].fillna(0).astype(int).values == 1
        nonburst_mask = ~burst_mask

        diagnostics["burst_count"] = int(burst_mask.sum())
        diagnostics["mean_score_burst"] = float(np.nanmean(oof_proba[burst_mask])) if burst_mask.any() else np.nan
        diagnostics["mean_score_nonburst"] = float(np.nanmean(oof_proba[nonburst_mask]))

        # burst-only AUC (only if both classes exist in burst subset)
        if burst_mask.any() and len(np.unique(y[burst_mask])) > 1:
            diagnostics["auc_burst_subset"] = float(roc_auc_score(y[burst_mask], oof_proba[burst_mask]))
        else:
            diagnostics["auc_burst_subset"] = np.nan

    if auth_col in X.columns:
        auth_mask = X[auth_col].fillna(0).astype(int).values == 1
        diagnostics["auth_count"] = int(auth_mask.sum())
        if auth_mask.any() and len(np.unique(y[auth_mask])) > 1:
            diagnostics["auc_auth_subset"] = float(roc_auc_score(y[auth_mask], oof_proba[auth_mask]))
        else:
            diagnostics["auc_auth_subset"] = np.nan

    results = {
        "schema": schema,
        "model": model,
        "aucs": aucs,
        "mean_auc": mean_auc,
        "y_true": y,
        "proba_oof": oof_proba,
        "diagnostics": diagnostics,
    }

    return results

def run_experiment(df, window_size, scenario_name=None):
    """
    Train baseline vs baseline+burst for one window size.
    Optionally restrict to a single scenario.
    """

    if scenario_name is not None:
        df = df[df["scenario"] == scenario_name].copy()
        print(f"\nRunning scenario = {scenario_name} ({len(df)} alerts)")

    # guard: need both classes
    if df["y"].nunique() < 2:
        raise ValueError(
            f"Scenario '{scenario_name}' has only one class in y: {df['y'].unique()}"
        )

    # --- build features ---
    X_dyn, y, df_used = build_dyn_features(df, window_size)    

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"After preprocessing, scenario '{scenario_name}' has only one class in y."
        )

    X_static = build_static_features(df_used)
    X_symbolic = symbolic_features.build_symbolic_features(df_used, X_dyn=X_dyn)

    print("\nSymbolic features generated:")
    print(sorted(X_symbolic.columns))

    # derive active features from what the builder emitted
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]
    sym_miss = [c for c in X_symbolic.columns if c.startswith("m_")]

    print("Active symbolic:", sym_feats)

    X_full = pd.concat([X_dyn, X_static, X_symbolic], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    assert len(X_full) == len(y)

    # --- schemas ---
    base_feats = list(X_dyn.columns) + list(X_static.columns)

    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]

    schema_base = FeatureSchema("base", base_feats)
    schema_symbolic = FeatureSchema(
        "base+symbolic",
        base_feats + sym_feats
    )

    print("\nSymbolic feature positives (auto):")
    for f in sorted([c for c in X_full.columns if c.startswith("is_")]):
        print(f"  {f}: {int(X_full[f].sum())}")


    print(f"Total features: {X_full.shape[1]} (static: {len(X_static.columns)}, dynamic: {len(X_dyn.columns)}, symbolic: {len(X_symbolic.columns)})")   

    # --- train ---

    print("\nTraining BASELINE model...")
    res_base = train_eval_holdout(X_full, y, schema_base, test_frac=0.3)
    diag_base = burst_diagnostics(X_full, res_base)

    # store per-feature results
    ablation_results = {}   # feat -> res
    ablation_diags = {}     # feat -> diag

    for feat in sym_feats:
        print(f"\nTraining base + '{feat}' ...")
        schema_feat = FeatureSchema(f"base+{feat}", base_feats + [feat])
        res_feat = train_eval_holdout(X_full, y, schema_feat, test_frac=0.3)

        ablation_results[feat] = res_feat
        ablation_diags[feat] = burst_diagnostics(X_full, res_feat)

    # also train the full bundle once (optional but useful)
    print("\nTraining BASE + ALL symbolic features...")
    schema_all = FeatureSchema("base+symbolic_all", base_feats + sym_feats)
    res_all = train_eval_holdout(X_full, y, schema_all, test_frac=0.3)
    diag_all = burst_diagnostics(X_full, res_all)

    return {
        "base": res_base,
        "diag_base": diag_base,

        # one-feature-at-a-time ablation
        "ablation": ablation_results,
        "diag_ablation": ablation_diags,

        # full symbolic bundle
        "sym_all": res_all,
        "diag_sym_all": diag_all,

        "X_full": X_full,
        "y": y,
        "sym_feats": sym_feats,
        "base_feats": base_feats,
    }

