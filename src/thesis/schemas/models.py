from pydantic import BaseModel
from typing import Any


class ModelArtifact(BaseModel):
    model: Any
    schema_name: str
    schema_version: str
    features: list[str]
    model_type: str
    model_version: str
    training_config: dict
    metrics: dict

    class Config:
        arbitrary_types_allowed = True


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    features: list[str]
    model_type: str
    training_config: dict
    metrics: dict


class TrainedModelSummary(BaseModel):
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    output_dir: str
    auc: float
    n_features: int
    feature_names: list[str]
    test_idx_start: int
    test_size: int
    single_class_split: bool
