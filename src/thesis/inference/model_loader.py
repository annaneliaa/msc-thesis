import json
import joblib
from thesis.registry.models import resolve_model_paths
from thesis.inference.runtime_models import SklearnTabularModel
from thesis.schemas.models import ModelArtifact

# just load model artifact using the registry function


def load_model(name: str, version: str) -> SklearnTabularModel:
    model_path, metadata_path = resolve_model_paths(name, version)

    model = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata_payload = json.load(f)

    return ModelArtifact(
        model=model,
        schema_name=metadata_payload["schema_name"],
        features=metadata_payload["features"],
        model_type=metadata_payload.get("model_type", "unknown"),
        model_version=metadata_payload["model_version"],
        training_config=metadata_payload.get("training_config", {}),
        metrics=metadata_payload.get("metrics", {}),
    )
