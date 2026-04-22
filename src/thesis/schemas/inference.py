from pydantic import BaseModel


# Pydantic object models used in inference module (API payloads and metadata objects)
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    score: float
    model_name: str
    model_version: str
