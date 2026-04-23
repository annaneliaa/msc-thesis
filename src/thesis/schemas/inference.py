from pydantic import BaseModel
from dataclasses import dataclass


# Pydantic object models used in inference module (API payloads and metadata objects)
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    score: float
    model_name: str
    model_version: str


@dataclass
class PredictionResult:
    predicted_label: int
    score: float
    threshold: float
    schema_name: str
    model_version: str
