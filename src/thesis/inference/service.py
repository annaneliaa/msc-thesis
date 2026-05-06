from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from thesis.encoders.service import encode_transactions_for_schema
from thesis.inference.model_loader import load_model
from thesis.inference.runtime_models import SklearnTabularModel
from thesis.schemas.features import FeatureSchema


@dataclass
class InferenceResult:
    n_transactions: int
    probabilities: list[float]
    predictions: list[int]
    labels: list[int] | None  # ground-truth 0/1; None if unavailable
    metrics: dict = field(default_factory=dict)


def load_model_for_inference(
    scenario: str, model_name: str, model_version: str
) -> SklearnTabularModel:
    return load_model(scenario, model_name, model_version)


def run_inference_on_transactions(
    model: SklearnTabularModel,
    schema: FeatureSchema,
    transactions: list,
) -> InferenceResult:
    feature_df = encode_transactions_for_schema(
        transactions=transactions,
        schema=schema,
    )

    probabilities = model.predict_proba(feature_df)
    predictions = (probabilities >= 0.5).astype(int).tolist()

    raw_labels = [getattr(t, "tx_label", None) for t in transactions]
    labels: list[int] | None = None
    if all(lbl in ("benign", "attack") for lbl in raw_labels):
        labels = [1 if lbl == "attack" else 0 for lbl in raw_labels]

    metrics: dict = {}
    if labels is not None:
        y_true = np.array(labels)
        y_pred = np.array(predictions)

        if np.unique(y_true).size < 2:
            metrics = {"single_class": True}
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics = {
                "auc": float(roc_auc_score(y_true, probabilities)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }

    return InferenceResult(
        n_transactions=len(transactions),
        probabilities=probabilities.tolist(),
        predictions=predictions,
        labels=labels,
        metrics=metrics,
    )
