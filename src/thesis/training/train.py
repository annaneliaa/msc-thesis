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

    model = model_factory()
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba_test))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    proba_train = model.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, proba_train))

    importances = {}

    coef_importances = {}
    if hasattr(model, "coef_"):
        coefs_signed = model.coef_[0]  # preserve sign; negative = benign-correlated
        pairs = sorted(
            zip(feature_names, coefs_signed),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        coef_importances = {
            name: float(coef) for name, coef in pairs[:top_n_importances]
        }
    elif hasattr(model, "feature_importances_"):
        pairs = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        coef_importances = {name: float(imp) for name, imp in pairs[:top_n_importances]}

    perm_importances = {}
    try:
        if getattr(model, "_skip_permutation", False):
            raise RuntimeError(
                "Permutation importance skipped: model flagged as too expensive"
            )
        _n_jobs = 1 if getattr(model, "_skip_shap", False) else -1
        perm_result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=_n_jobs
        )
        pairs = sorted(
            zip(feature_names, perm_result.importances_mean),
            key=lambda x: x[1],
            reverse=True,
        )
        perm_importances = {name: float(imp) for name, imp in pairs[:top_n_importances]}
    except Exception:
        pass

    shap_importances = {}
    try:
        import shap

        bg = X_train.sample(min(100, len(X_train)), random_state=42)
        x_explain = X_test.iloc[:200] if len(X_test) > 200 else X_test

        if hasattr(model, "get_shap_values"):
            # model provides its own fast SHAP path (e.g. GradientExplainer for LSTM)
            bg_arr = bg.values if hasattr(bg, "values") else np.asarray(bg)
            x_arr = (
                x_explain.values
                if hasattr(x_explain, "values")
                else np.asarray(x_explain)
            )
            vals = model.get_shap_values(bg_arr, x_arr)
        elif getattr(model, "_skip_shap", False):
            raise RuntimeError("SHAP skipped: model flagged as too expensive")
        elif hasattr(model, "feature_importances_"):
            # tree models: TreeExplainer is exact and fast
            sv = shap.TreeExplainer(model).shap_values(x_explain)
            vals = (
                sv[:, :, 1]
                if isinstance(sv, np.ndarray) and sv.ndim == 3
                else (sv[1] if isinstance(sv, list) else sv)
            )
        elif hasattr(model, "coef_"):
            # linear models: LinearExplainer
            vals = shap.LinearExplainer(model, bg).shap_values(x_explain)
        else:
            # fallback: PermutationExplainer via predict_proba
            sv = shap.Explainer(model.predict_proba, bg)(x_explain)
            vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

        mean_signed = vals.mean(axis=0)
        pairs = sorted(
            zip(feature_names, mean_signed),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        shap_importances = {name: float(imp) for name, imp in pairs[:top_n_importances]}
    except Exception:
        pass

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
