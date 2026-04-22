import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from thesis.schemas.features import FeatureSchema


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


def train_eval_holdout(X_full, y, schema, test_frac=0.3, time_col="timestamp"):
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

    def _find_valid_time_split(y, test_frac=0.3):
        n = len(y)
        split0 = int((1 - test_frac) * n)

        # search nearby split points until both sides have both classes
        for delta in range(0, n):
            for s in (split0 - delta, split0 + delta):
                if s <= 1 or s >= n - 1:
                    continue
                if np.unique(y[:s]).size == 2 and np.unique(y[s:]).size == 2:
                    return s
        return None

    # keep y as a Series aligned to X_full
    if hasattr(y, "reset_index"):
        y = y.reset_index(drop=True)
    else:
        y = pd.Series(y)

    df = X_full.reset_index(drop=True).copy()
    df["__y__"] = y.values

    # enforce time order if available
    if time_col in df.columns:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    X = df[schema.features].fillna(0)
    y = df["__y__"].to_numpy()

    # n = len(X)
    split = _find_valid_time_split(y, test_frac=test_frac)
    if split is None:
        raise ValueError(
            "Could not find a split where both train and test have both classes."
        )

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        # time-realistic runs can legitimately be single-class
        return {
            "schema": schema.name,
            "model": None,
            "auc": np.nan,
            "y_test": y_test,
            "proba_test": None,
            "test_idx_start": split,
            "feature_names": list(X.columns),
            "single_class_split": True,
        }

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
        "feature_names": list(X.columns),
        "single_class_split": False,
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
