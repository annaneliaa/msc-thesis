from fastapi import FastAPI

from thesis.config import load_settings
from thesis.inference.model_loader import load_model
from thesis.inference.schemas import PredictRequest, PredictResponse
from thesis.inference.service import predict_text
from thesis.paths import ensure_artifact_dirs

settings = load_settings()
ensure_artifact_dirs()
model = load_model(settings)

app = FastAPI(title=settings.app.name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return predict_text(model, req.text)
