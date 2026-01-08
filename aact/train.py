import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from plots import (
    plot_roc,
    plot_alert_reduction,
    plot_feature_importance,
    plot_confidence_distribution,
    plot_top_error_categories,
)

def train_and_evaluate(X, y, n_splits=3):

    X = X.reset_index(drop=True)
    y = np.asarray(y)

    assert len(X) == len(y)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
    )

    aucs = []
    all_proba = []
    all_y = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)

        all_proba.extend(proba)
        all_y.extend(y_test)

        print(f"Fold {fold} ROC-AUC: {auc:.3f}")

    mean_auc = float(np.mean(aucs))
    print(f"Mean ROC-AUC: {mean_auc:.3f}")

    results = {
        "model": model,
        "aucs": aucs,
        "mean_auc": mean_auc,
        "y_true": np.array(all_y),
        "proba": np.array(all_proba),
    }

    return results
