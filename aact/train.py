import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from schema import FeatureSchema
from memory import SymbolicMemory
from build_features import build_dyn_features, build_static_features
import symbolic_features
from metrics import eval_subset_metrics


def train_lr_l1(X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        # penalty="l1",
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_eval_holdout(X_full, y, schema, test_frac=0.3):
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

    # guard
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


def train_and_evaluate(
    X,
    y,
    schema,
    n_splits=3,
    burst_col="is_suspicious_auth_burst",
    auth_col="is_auth_event",
):

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
    def train_fn(X_full, y, feature_list):
        schema = FeatureSchema("tmp", feature_list)
        return train_eval_holdout(X_full, y, schema, test_frac=test_frac)

    return train_fn


def greedy_symbolic_search(
    df,
    window_size,
    train_fn,  # function(X_full, y, feature_list) -> res dict with model, proba_test, test_idx_start
    threshold=0.5,
    max_k=6,
    lambda_fp=0.000001,  # penalty per FP (tune)
    min_subset_size=100,  # require feature fires at least this many times in test (optional)
):
    """Greedy forward feature-selection loop that tries to pick up max_k symbolic (is_X) features
    to add on top of a baseline (dynamic + static features), keeping only additions that improve
    the objective (currently, AUC and FP).
    """
    # --- build features ---
    X_dyn, y, df_used = build_dyn_features(df, window_size)
    X_static = build_static_features(df_used)
    X_symbolic = symbolic_features.build_symbolic_features(df_used, X_dyn=X_dyn)

    print("\nSymbolic features generated:")
    print(sorted(X_symbolic.columns))

    # derive active features from what the builder emitted
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]
    sym_miss = [c for c in X_symbolic.columns if c.startswith("m_")]

    print("Active symbolic:", sym_feats)

    # concatenate features
    X_full = pd.concat([X_dyn, X_static, X_symbolic], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    assert len(X_full) == len(y)

    # --- schemas ---
    base_feats = list(X_dyn.columns) + list(X_static.columns)

    # only columns starting with is_ are treated as symbolic features now
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]

    # train baseline feature set (dyn + static)
    res_base = train_fn(X_full, y, base_feats)
    # ccompute metrics
    m_base = eval_subset_metrics(X_full, y, res_base, threshold=threshold)

    # Greedy subset selection
    chosen = []
    remaining = list(sym_feats)
    history = []

    def objective(m, m_base):
        if m["subset_size"] is not None and m["subset_size"] < min_subset_size:
            return -np.inf

        fp_reduction = m_base["fp"] - m["fp"]

        return (
            (m["auc"] if not np.isnan(m["auc"]) else -1.0)
            + 0.00001 * fp_reduction
            - lambda_fp * m["fp"]
        )


    best_score = objective(m_base, m_base)

    for step in range(max_k):

        best_candidate = None
        best_candidate_res = None
        best_candidate_metrics = None
        best_candidate_score = best_score

        # For each remaining symbolic feature f
        # train a model on base_feats + chosen + [f]
        for f in remaining:
            feats = base_feats + chosen + [f]
            res = train_fn(X_full, y, feats)
            m = eval_subset_metrics(X_full, y, res, threshold=threshold, subset_col=f)

            # compute score based on objective
            score = objective(m, m_base)

            # stops when no candidate improves the objective
            if score > best_candidate_score:
                best_candidate = f
                best_candidate_res = res
                best_candidate_metrics = m
                best_candidate_score = score

        if best_candidate is None:
            break

        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score

        history.append(
            {
                "step": step + 1,
                "added": best_candidate,
                "chosen": chosen.copy(),
                **best_candidate_metrics,
            }
        )

    return {
        "base": res_base,
        "base_metrics": m_base,
        "chosen": chosen,
        "history": history,
    }
