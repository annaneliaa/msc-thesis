from fastapi import FastAPI

from thesis.config import load_settings
from thesis.inference.model_loader import load_model
from thesis.paths import ensure_artifact_dirs

settings = load_settings()
ensure_artifact_dirs()
model = load_model(
    settings.model.scenario, settings.model.model_name, settings.model.model_version
)

app = FastAPI(title=settings.app.name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# @app.post("/predict", response_model=PredictResponse)
# def predict(req: PredictRequest) -> PredictResponse:
#     return predict_text(model, req.text)
