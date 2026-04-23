import numpy as np
from sklearn.metrics import roc_auc_score
from thesis.schemas.features import FeatureSchema
import pandas as pd


def train_eval_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    schema: FeatureSchema,
    model_factory,
    test_idx_start: int | None = None,
) -> dict:
    feature_names = list(X_train.columns)

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {
            "schema": schema.name,
            "model": None,
            "auc": np.nan,
            "y_test": y_test,
            "proba_test": None,
            "test_idx_start": test_idx_start,
            "feature_names": feature_names,
            "single_class_split": True,
        }

    model = model_factory()
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba_test))

    return {
        "schema": schema.name,
        "model": model,
        "auc": auc,
        "y_test": y_test,
        "proba_test": proba_test,
        "test_idx_start": test_idx_start,
        "feature_names": feature_names,
        "single_class_split": False,
    }
