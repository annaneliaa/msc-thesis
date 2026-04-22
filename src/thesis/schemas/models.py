from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel

# data structures for model artifacts and metadata


@dataclass
class ModelArtifact:
    model: Any
    schema_name: str
    features: list[str]
    model_type: str
    model_version: str
    training_config: dict
    metrics: dict


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    schema_name: str
    features: list[str]
    model_type: str
    training_config: dict
    metrics: dict


@dataclass
class TrainedModelSummary:
    model_name: str
    model_version: str
    schema_name: str
    output_dir: str
    auc: float
    n_features: int
    feature_names: list[str]
    test_idx_start: int
    test_size: int
    single_class_split: bool
