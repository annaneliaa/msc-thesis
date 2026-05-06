from pathlib import Path
import json
import joblib
from dataclasses import asdict

from thesis.paths import (
    MODEL_FILENAME,
    METADATA_FILENAME,
    FEATURE_SCHEMA_FILENAME,
    ensure_artifact_dirs,
)
from thesis.schemas.models import ModelArtifact, ModelMetadata
from thesis.schemas.features import FeatureSchema
from thesis.registry.models import resolve_model_paths


def save_model_artifact(
    artifact: ModelArtifact,
    metadata: ModelMetadata,
    schema: FeatureSchema,
    output_dir: Path,
) -> None:
    ensure_artifact_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact.model, output_dir / MODEL_FILENAME)

    with (output_dir / METADATA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    with (output_dir / FEATURE_SCHEMA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)


def load_model_artifact(
    scenario: str, model_name: str, model_version: str
) -> ModelArtifact:
    (
        model_path,
        metadata_path,
        _,
    ) = resolve_model_paths(
        scenario=scenario,
        name=model_name,
        version=model_version,
    )

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
        schema_version=metadata_payload["schema_version"],
        features=metadata_payload["features"],
        model_type=metadata_payload["model_type"],
        training_config=metadata_payload["training_config"],
        metrics=metadata_payload["metrics"],
    )

    return ModelArtifact(
        model=model,
        schema_name=metadata.schema_name,
        schema_version=metadata.schema_version,
        features=metadata.features,
        model_type=metadata.model_type,
        model_version=metadata.model_version,
        training_config=metadata.training_config,
        metrics=metadata.metrics,
    )
