from pathlib import Path
import json
import joblib

from thesis.paths import MODELS_DIR, ensure_artifact_dirs
from thesis.schemas.models import ModelArtifact, ModelMetadata
from thesis.schemas.features import FeatureSchema

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


def save_model_artifact(
    artifact: ModelArtifact,
    metadata: ModelMetadata,
    schema: FeatureSchema,
    output_dir: Path,
) -> None:
    ensure_artifact_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    feature_schema_path = output_dir / FEATURE_SCHEMA_FILENAME

    joblib.dump(artifact.model, model_path)

    metadata_payload = metadata.model_dump()

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    with feature_schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema.__dict__, f, indent=2)


def load_model_artifact(model_name: str, model_version: str) -> ModelArtifact:
    model_dir = MODELS_DIR / model_name / model_version
    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    model = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata_payload = json.load(f)

    metadata = ModelMetadata(
        model_name=metadata_payload["model_name"],
        model_version=metadata_payload["model_version"],
        schema_name=metadata_payload["schema_name"],
        features=metadata_payload["features"],
        metrics=metadata_payload["metrics"],
    )

    return ModelArtifact(
        model=model,
        schema_name=metadata.schema_name,
        features=metadata.features,
        model_type=metadata_payload.get("model_type", "unknown"),
        model_version=metadata.model_version,
        training_config=metadata_payload.get("training_config", {}),
        metrics=metadata.metrics,
    )
