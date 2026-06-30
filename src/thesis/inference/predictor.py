# row to result prediction
from typing import Any
from thesis.schemas.inference import PredictionResult

from thesis.inference.runtime_models import SklearnTabularModel


def predict_alert_group_row(
    row: dict[str, Any],
    encoder,
    model: SklearnTabularModel,
    threshold: float = 0.5,
) -> PredictionResult:
    """
    Predict a single incoming alert_group row.
    """
    X = encoder.transform_row(row)

    score = float(model.predict_proba(X)[0])
    label = int(score >= threshold)

    return PredictionResult(
        predicted_label=label,
        score=score,
        threshold=threshold,
        schema_name=model.schema_name,
        model_version=model.model_version,
    )
