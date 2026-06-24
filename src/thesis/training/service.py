import pandas as pd
from pathlib import Path

from thesis.schemas.models import (
    ModelArtifact,
    ModelMetadata,
    TrainedModelSummary,
)
from thesis.schemas.features import FeatureSchema
from thesis.training.persistence import save_model_artifact
from thesis.training.train import train_eval_holdout
from thesis.training.util import (
    prepare_training_frame,
    make_holdout_split,
)
from thesis.training.model_factory import get_model_factory


def train_model_for_schema(
    X: pd.DataFrame,
    y,
    schema: FeatureSchema,
    model_name: str,
    model_version: str,
    output_dir: Path,
    test_frac: float = 0.3,
    train_start: int = 0,
    random_split: bool = False,
    random_seed: int = 42,
) -> TrainedModelSummary:
    """
    Train, evaluate, and persist a model for a given feature schema.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    feature_names = schema.feature_names()

    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise KeyError(
            f"Schema '{schema.schema_name}' is missing columns in X: {missing}"
        )

    print("Creating new model instance...")
    model_factory = get_model_factory(model_name)

    print("Preparing training frame...")
    X_used, y_used = prepare_training_frame(
        X_full=X,
        y=y,
        schema=schema,
        random_split=random_split,
    )

    X_train, X_test, y_train, y_test, split = make_holdout_split(
        X=X_used,
        y=y_used,
        test_frac=test_frac,
        train_start=train_start,
    )

    print("Training and evaluating model...")
    result = train_eval_holdout(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        schema=schema,
        model_factory=model_factory,
        test_idx_start=split,
    )

    if result["model"] is None:
        print(
            f"  [warn] Schema '{schema.schema_name}': single-class split "
            f"(train={pd.Series(y_train).value_counts().to_dict()}, "
            f"test={pd.Series(y_test).value_counts().to_dict()}). Skipping model fit."
        )
        return TrainedModelSummary(
            model_name=model_name,
            model_version=model_version,
            schema_name=schema.schema_name,
            schema_version=schema.schema_version,
            output_dir=str(output_dir),
            auc=float("nan"),
            n_features=len(feature_names),
            feature_names=feature_names,
            test_idx_start=int(X_train.shape[0]),
            test_size=int(X_test.shape[0]),
            single_class_split=True,
        )

    feature_sparsity = float((X_train == 0).values.mean())
    n_symbolic = len(schema.symbolic.features) if schema.symbolic is not None else 0

    print("Saving model artifact...")
    artifact = ModelArtifact(
        model=result["model"],
        schema_name=schema.schema_name,
        schema_version=schema.schema_version,
        features=result["feature_names"],
        model_type=type(result["model"]).__name__,
        model_version=model_version,
        training_config={
            "test_frac": test_frac,
            "train_start": train_start,
            "schema_name": schema.schema_name,
            "schema_version": schema.schema_version,
            "model_name": model_name,
            "model_version": model_version,
            "n_features": len(result["feature_names"]),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_label_dist": pd.Series(y_train).value_counts().to_dict(),
            "test_label_dist": pd.Series(y_test).value_counts().to_dict(),
        },
        metrics={
            "auc": float(result["auc"]),
            "accuracy": float(result["accuracy"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1": float(result["f1"]),
            "balanced_accuracy": float(result["balanced_accuracy"]),
            "tp": int(result["tp"]),
            "fp": int(result["fp"]),
            "tn": int(result["tn"]),
            "fn": int(result["fn"]),
            "train_auc": float(result["train_auc"]),
            "performance_gap_train_vs_test": float(result["train_auc"])
            - float(result["auc"]),
            "feature_sparsity": feature_sparsity,
            "avg_feature_density": 1.0 - feature_sparsity,
            "n_symbolic_features_used": n_symbolic,
            "top_feature_importances": result["top_feature_importances"],
            "single_class_split": bool(result["single_class_split"]),
        },
    )

    print("Creating model metadata...")
    metadata = ModelMetadata(
        model_name=model_name,
        model_version=model_version,
        schema_name=schema.schema_name,
        schema_version=schema.schema_version,
        features=result["feature_names"],
        model_type=artifact.model_type,
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
        schema_name=schema.schema_name,
        schema_version=schema.schema_version,
        output_dir=str(output_dir),
        auc=float(result["auc"]),
        n_features=len(result["feature_names"]),
        feature_names=result["feature_names"],
        test_idx_start=int(result["test_idx_start"]),
        test_size=int(len(result["y_test"])),
        single_class_split=bool(result["single_class_split"]),
    )
