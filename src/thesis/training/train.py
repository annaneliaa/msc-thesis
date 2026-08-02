import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

from thesis.schemas.features import FeatureSchema
from thesis.training.explain import compute_shap_signed_importances
from thesis.training.model_factory import unwrap_estimator
from thesis.training.workload import DEFAULT_RECALL_TARGETS, compute_workload_at_recall


def train_eval_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    schema: FeatureSchema,
    model_factory,
    test_idx_start: int | None = None,
    top_n_importances: int = 30,
    compute_importances: bool = True,
) -> dict:
    feature_names = list(X_train.columns)

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {
            "schema": schema.schema_name,
            "model": None,
            "auc": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "balanced_accuracy": np.nan,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "workload_at_recall": {f"{r:.2f}": None for r in DEFAULT_RECALL_TARGETS},
            "train_auc": np.nan,
            "top_feature_importances": {},
            "y_test": y_test,
            "proba_test": None,
            "test_idx_start": test_idx_start,
            "feature_names": feature_names,
            "single_class_split": True,
        }

    print(
        f"  [train] X_train={X_train.shape}, X_test={X_test.shape}, "
        f"n_features={len(feature_names)}",
        flush=True,
    )

    model = model_factory()
    print(f"  [train] fitting {type(model).__name__}...", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  [train] fit done in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba_test))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    proba_train = model.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, proba_train))
    print(f"  [train] predict + metrics done in {time.time() - t0:.1f}s", flush=True)

    coef_importances = {}
    perm_importances = {}
    shap_importances = {}

    # .coef_/.feature_importances_ live on the fitted estimator itself, not on
    # a Pipeline wrapping it (e.g. the scaled "logreg"/"logreg_l1" factories)
    # -- unwrap once and reuse below.
    estimator = unwrap_estimator(model)

    if compute_importances:
        if hasattr(estimator, "coef_"):
            coefs_signed = estimator.coef_[
                0
            ]  # preserve sign; negative = benign-correlated
            pairs = sorted(
                zip(feature_names, coefs_signed),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            coef_importances = {
                name: float(coef) for name, coef in pairs[:top_n_importances]
            }
        elif hasattr(estimator, "feature_importances_"):
            pairs = sorted(
                zip(feature_names, estimator.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
            coef_importances = {
                name: float(imp) for name, imp in pairs[:top_n_importances]
            }

        try:
            if getattr(model, "_skip_permutation", False):
                raise RuntimeError(
                    "Permutation importance skipped: model flagged as too expensive"
                )
            # _gpu_model (torch_nn/rf_gpu, see training/gpu_models.py) forces
            # single-process too -- joblib's multiprocess backend would need
            # to re-pickle a GPU-resident model into each worker and give it
            # its own CUDA context, which is exactly the kind of contention
            # this is avoiding. Unlike _skip_shap, this does NOT skip SHAP
            # itself further down -- we still want feature attributions for
            # these models, just computed in-process.
            _n_jobs = (
                1
                if getattr(model, "_skip_shap", False)
                or getattr(model, "_gpu_model", False)
                else -1
            )
            print(
                f"  [train] permutation importance: n_repeats=10, n_jobs={_n_jobs}, "
                f"X_test={X_test.shape} ({len(feature_names)} features x 10 repeats "
                f"= {len(feature_names) * 10} extra predict passes)...",
                flush=True,
            )
            t0 = time.time()
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=_n_jobs
            )
            print(
                f"  [train] permutation importance done in {time.time() - t0:.1f}s",
                flush=True,
            )
            pairs = sorted(
                zip(feature_names, perm_result.importances_mean),
                key=lambda x: x[1],
                reverse=True,
            )
            perm_importances = {
                name: float(imp) for name, imp in pairs[:top_n_importances]
            }
        except Exception as e:
            print(f"  [train] permutation importance skipped: {e}", flush=True)

        try:
            print("  [train] computing SHAP importances...", flush=True)
            t0 = time.time()

            bg = X_train.sample(min(100, len(X_train)), random_state=42)
            x_explain = X_test.iloc[:200] if len(X_test) > 200 else X_test

            shap_importances = compute_shap_signed_importances(
                model, bg, x_explain, feature_names, top_n=top_n_importances
            )
            print(f"  [train] SHAP done in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [train] SHAP skipped: {e}", flush=True)

    importances = {
        "by_coefficient": coef_importances,
        "by_permutation": perm_importances,
        "by_shap": shap_importances,
    }

    return {
        "schema": schema.schema_name,
        "model": model,
        "auc": auc,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "workload_at_recall": compute_workload_at_recall(y_test, proba_test),
        "train_auc": train_auc,
        "top_feature_importances": importances,
        "y_test": y_test,
        "proba_test": proba_test,
        "test_idx_start": test_idx_start,
        "feature_names": feature_names,
        "single_class_split": False,
    }
