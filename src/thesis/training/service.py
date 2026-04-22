from thesis.schemas.features import FeatureSchema
import pandas as pd
from pathlib import Path
from thesis.schemas.models import ModelArtifact, ModelMetadata, TrainedModelSummary
from thesis.training.persistence import save_model_artifact
from thesis.training.train import train_eval_holdout


def get_feature_schema(name: str) -> FeatureSchema:
    return FeatureSchema(name=name)


def train_model_for_schema(
    X: pd.DataFrame,
    y,
    schema: FeatureSchema,
    model_name: str,
    model_version: str,
    output_dir: Path,
    test_frac: float = 0.3,
) -> dict:
    """
    Train, evaluate, and persist a model for a given feature schema.

    Steps:
    - validate schema columns exist in X
    - train/evaluate using holdout split
    - persist trained model + metadata + schema snapshot
    - return a compact training summary
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    missing = [col for col in schema.features if col not in X.columns]
    if missing:
        raise KeyError(f"Schema '{schema.name}' is missing columns in X: {missing}")

    result = train_eval_holdout(
        X_full=X,
        y=y,
        schema=schema,
        test_frac=test_frac,
    )

    if result["model"] is None:
        raise ValueError(
            f"Training for schema '{schema.name}' produced no fitted model "
            f"(single-class split)."
        )

    artifact = ModelArtifact(
        model=result["model"],
        schema_name=schema.name,
        features=result["feature_names"],
        model_type=type(result["model"]).__name__,
        model_version=model_version,
        training_config={
            "test_frac": test_frac,
            "schema_name": schema.name,
            "n_features": len(result["feature_names"]),
            "train_rows": int(result["test_idx_start"]),
            "test_rows": int(len(result["y_test"])),
        },
        metrics={
            "auc": float(result["auc"]),
            "single_class_split": bool(result["single_class_split"]),
        },
    )

    metadata = ModelMetadata(
        model_name=model_name,
        model_version=model_version,
        schema_name=schema.name,
        features=result["feature_names"],
        model_type=type(result["model"]).__name__,
        training_config=artifact.training_config,
        metrics=artifact.metrics,
    )

    save_model_artifact(
        artifact=artifact,
        metadata=metadata,
        schema=schema,
        output_dir=output_dir,
    )

    return TrainedModelSummary(
        model_name=model_name,
        model_version=model_version,
        schema_name=schema.name,
        output_dir=str(output_dir),
        auc=float(result["auc"]),
        n_features=len(result["feature_names"]),
        feature_names=result["feature_names"],
        test_idx_start=int(result["test_idx_start"]),
        test_size=int(len(result["y_test"])),
        single_class_split=bool(result["single_class_split"]),
    )
