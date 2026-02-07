import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from classes import FeatureSchema


def train_lr_l1(X_train, y_train):
    """
    Fit a sparse logistic regression model (L1-regularized) for feature selection
    """
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l1",
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_eval_holdout(X_full, y, schema, test_frac=0.3):
    """
    Train + evaluate one model on a single holdout split (NOT time-series CV, so it has higher variance)

    - Select columns from X_full according to `schema.features`
    - Split data by index order:
        - train = first (1-test_frac) fraction
        - test  = last test_frac fraction
       (important: this is NOT shuffled)
    - Train logistic regression (L2 default)
    - Compute ROC-AUC on the test split
    - Return a result dict that downstream functions can reuse (metrics + predictions + split point):
        - schema: schema name (string)
        - model: fitted model
        - auc: ROC-AUC on test split
        - y_test: labels for test split
        - proba_test: predicted probabilities for test split
        - test_idx_start: integer index where test split begins (used later for support checks etc.)
        - feature_names: list of actual columns used (after schema selection)
    """
    X_full = X_full.reset_index(drop=True)
    y = np.asarray(y)
    assert len(X_full) == len(y)

    # select schema columns
    X = X_full[schema.features].fillna(0)
    feature_names = list(X.columns)

    n = len(X)
    split = int((1 - test_frac) * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # guard: need both classes in both splits for AUC
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "Train or test has only one class. Try a different split_frac or use all scenarios."
        )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
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
        "feature_names": feature_names,
    }


def train_and_eval(
    X,
    y,
    schema,
    n_splits=3,
    burst_col="is_suspicious_auth_burst",
    auth_col="is_auth_event",
):
    """
    Time-series cross-validation training + evaluation with optional subgroup diagnostics
    Trains logistic regression per fold and computes ROC-AUC per fold.
    Stores out-of-fold probabilities (oof_proba) for every row that was evaluated.

    Inputs
    - X: DataFrame of candidate features (may contain more columns than schema uses)
    - y: labels aligned with X
    - schema: FeatureSchema object; determines which columns are used
    - n_splits: number of time-series folds
    - burst_col/auth_col: optional columns used only for diagnostics (if present)

    Returns dict with
    - schema: the schema object (not just name)
    - model: last fitted model (from last valid fold)
    - aucs: list of fold AUCs
    - mean_auc: mean of fold AUCs
    - y_true: full y array
    - proba_oof: out-of-fold probabilities aligned to rows (NaN where never evaluated)
    - diagnostics: burst/auth counts + mean scores + subset AUCs (when computable)
    """


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
        raise ValueError(
            f"[{schema.name}] No valid folds (all single-class). Try fewer splits or a different scenario."
        )

    mean_auc = float(np.mean(aucs))
    print(f"Mean ROC-AUC: {mean_auc:.3f}")

    # --- Diagnostics: burst scoring and subset AUCs ---
    diagnostics = {}

    if burst_col in X.columns:
        burst_mask = X[burst_col].fillna(0).astype(int).values == 1
        nonburst_mask = ~burst_mask

        diagnostics["burst_count"] = int(burst_mask.sum())
        diagnostics["mean_score_burst"] = (
            float(np.nanmean(oof_proba[burst_mask])) if burst_mask.any() else np.nan
        )
        diagnostics["mean_score_nonburst"] = float(np.nanmean(oof_proba[nonburst_mask]))

        # burst-only AUC (only if both classes exist in burst subset)
        if burst_mask.any() and len(np.unique(y[burst_mask])) > 1:
            diagnostics["auc_burst_subset"] = float(
                roc_auc_score(y[burst_mask], oof_proba[burst_mask])
            )
        else:
            diagnostics["auc_burst_subset"] = np.nan

    if auth_col in X.columns:
        auth_mask = X[auth_col].fillna(0).astype(int).values == 1
        diagnostics["auth_count"] = int(auth_mask.sum())
        if auth_mask.any() and len(np.unique(y[auth_mask])) > 1:
            diagnostics["auc_auth_subset"] = float(
                roc_auc_score(y[auth_mask], oof_proba[auth_mask])
            )
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


def make_train_fn(test_frac=0.3):
    """
    Returns train_fn(X_full, y, feature_list) -> result dict from train_eval_holdout with fixed holdout split fraction
    """
    def train_fn(X_full, y, feature_list):
        schema = FeatureSchema("tmp", feature_list)
        return train_eval_holdout(X_full, y, schema, test_frac=test_frac)

    return train_fn
