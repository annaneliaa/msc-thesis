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

from thesis.schemas.features import FeatureSchema


def train_eval_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    schema: FeatureSchema,
    model_factory,
    test_idx_start: int | None = None,
    top_n_importances: int = 10,
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
    if hasattr(model, "feature_importances_"):
        pairs = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        importances = {name: float(imp) for name, imp in pairs[:top_n_importances]}

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
        "train_auc": train_auc,
        "top_feature_importances": importances,
        "y_test": y_test,
        "proba_test": proba_test,
        "test_idx_start": test_idx_start,
        "feature_names": feature_names,
        "single_class_split": False,
    }
