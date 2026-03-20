from thesis.inference.model_loader import DummyModel
from thesis.inference.schemas import PredictResponse


def predict_text(model: DummyModel, text: str) -> PredictResponse:
    label, score = model.predict(text)
    return PredictResponse(
        label=label,
        score=score,
        model_name=model.model_name,
        model_version=model.model_version,
    )