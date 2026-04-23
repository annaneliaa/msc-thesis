import json
import joblib

from thesis.registry.models import resolve_model_paths
from thesis.inference.runtime_models import SklearnTabularModel


def load_model(name: str, version: str) -> SklearnTabularModel:
    model_path, metadata_path = resolve_model_paths(name, version)

    model = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata_payload = json.load(f)

    return SklearnTabularModel(
        model_name=name,
        model_version=metadata_payload["model_version"],
        schema_name=metadata_payload["schema_name"],
        features=metadata_payload["features"],
        model=model,
    )
