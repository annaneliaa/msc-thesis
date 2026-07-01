"""
Transfer experiment: train on scenario X, test on scenario Y.

Tests how well a model trained on one scenario generalises to another.
The train scenario's feature schema is used to encode the test alert_groups,
so symbolic features that never appear in the test data simply evaluate to 0.

Steps:
  1. Verify the train scenario has a feature schema; initialise if missing
  2. Check if a trained model exists; train one on the train scenario if not
  3. Convert test scenario alerts CSV → JSON (skipped if already done)
  4. Tokenise + ingest test alerts into cache (skipped if already done)
  5. Build alert_groups from test scenario cache
  6. Encode test alert_groups under the train scenario schema
  7. Run inference with the trained model
  8. Write results to artifacts/experiments/<train>_to_<test>/transfer_<ts>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from thesis.experiments.baseline import (
    _EXPERIMENTS_DIR,
    _ROOT,
    BaselineExperimentConfig,
    run_baseline_experiment,
)
from thesis.pipeline.pipeline import (
    convert_ait_alerts_to_json,
    ensure_feature_manifest,
    ingest_ait_alert_batch,
    load_or_build_alert_groups,
)
from thesis.features.schema_registry import FeatureSchemaRegistry
from thesis.inference.service import (
    InferenceResult,
    load_model_for_inference,
    run_inference_on_alert_groups,
)
from thesis.config import GroupingConfig
from thesis.paths import CACHE_DIR, MODELS_DIR, MODEL_FILENAME, ensure_artifact_dirs
from thesis.schemas.mining import FeatureSelectionConfig


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------


@dataclass
class TransferExperimentConfig:
    train_scenario: str
    test_scenario: str
    schema_name: str = "base+symbolic"
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    filter_config: Path | None = None
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)


@dataclass
class TransferExperimentResult:
    train_scenario: str
    test_scenario: str
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    n_test_alert_groups: int
    metrics: dict
    results_file: Path


# ---------------------------------------------------------------------------
# Private step helpers
# ---------------------------------------------------------------------------


def _model_exists(scenario: str, model_name: str, model_version: str) -> bool:
    return (
        MODELS_DIR / scenario / model_name / model_version / MODEL_FILENAME
    ).exists()


def _ensure_trained_model(config: TransferExperimentConfig) -> None:
    if _model_exists(config.train_scenario, config.model_name, config.model_version):
        print(
            f"  [skip] Model '{config.model_name}' v{config.model_version} already exists."
        )
        return

    print(
        f"  Model not found — training on '{config.train_scenario}' "
        f"(schema='{config.schema_name}')..."
    )

    if "symbolic" in config.schema_name:
        from thesis.experiments.symbolic import (
            SymbolicExperimentConfig,
            run_symbolic_experiment,
        )

        run_symbolic_experiment(
            SymbolicExperimentConfig(
                scenario=config.train_scenario,
                schema_name=config.schema_name,
                model_name=config.model_name,
                model_version=config.model_version,
                filter_config=config.filter_config,
                feature_selection=config.feature_selection,
                cache_dir=config.cache_dir,
            )
        )
    else:
        run_baseline_experiment(
            BaselineExperimentConfig(
                scenario=config.train_scenario,
                schema_name=config.schema_name,
                model_name=config.model_name,
                model_version=config.model_version,
                cache_dir=config.cache_dir,
            )
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_transfer_experiment(
    config: TransferExperimentConfig,
) -> TransferExperimentResult:
    ensure_artifact_dirs()
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Transfer] Train: '{config.train_scenario}' → Test: '{config.test_scenario}'"
    )

    # 1. Ensure train schema exists
    print("[1/8] Checking train scenario feature manifest...")
    ensure_feature_manifest(config.train_scenario)

    # 2. Ensure model is trained
    print("[2/8] Checking trained model...")
    _ensure_trained_model(config)

    # 3. Convert test alerts CSV → JSON
    print("[3/8] Converting test alerts to JSON...")
    alerts_path = convert_ait_alerts_to_json(config.test_scenario)

    # 4. Tokenise + ingest test alerts into cache
    print("[4/8] Processing test alert batch...")
    ingest_ait_alert_batch(
        config.test_scenario,
        alerts_path,
        config.cache_dir,
        grouping_mode=config.grouping.mode,
        grouping=config.grouping,
    )

    # 5. Build test alert_groups
    print("[5/8] Building test alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(config.test_scenario, config.cache_dir)

    # 6. Load model and train schema
    print("[6/8] Loading model and train schema...")
    model = load_model_for_inference(
        config.train_scenario, config.model_name, config.model_version
    )

    registry = FeatureSchemaRegistry(root_dir=_ROOT / "artifacts" / "features")
    schema = registry.load(
        scenario_name=config.train_scenario,
        schema_name=config.schema_name,
    )
    print(
        f"  Schema '{schema.schema_name}' v{schema.schema_version} "
        f"({len(schema.feature_names())} features)"
    )

    # 7. Run inference on test alert_groups
    print("[7/8] Running inference on test alert_groups...")
    result: InferenceResult = run_inference_on_alert_groups(
        model=model,
        schema=schema,
        alert_groups=alert_groups,
    )

    auc = result.metrics.get("auc", float("nan"))
    print(f"  AUC: {auc:.4f}  (n={result.n_alert_groups})")

    # 8. Save results
    print("[8/8] Saving transfer results...")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = (
        _EXPERIMENTS_DIR / f"{config.train_scenario}_to_{config.test_scenario}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"transfer_{timestamp}.json"

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "transfer",
                "train_scenario": config.train_scenario,
                "test_scenario": config.test_scenario,
                "timestamp": timestamp,
                "model_name": config.model_name,
                "model_version": config.model_version,
                "schema_name": schema.schema_name,
                "schema_version": schema.schema_version,
                "filter_config": (
                    str(config.filter_config)
                    if config.filter_config is not None
                    else None
                ),
                "n_test_alert_groups": result.n_alert_groups,
                "metrics": result.metrics,
            },
            f,
            indent=2,
        )

    print(f"  Results → {results_file}")

    return TransferExperimentResult(
        train_scenario=config.train_scenario,
        test_scenario=config.test_scenario,
        model_name=config.model_name,
        model_version=config.model_version,
        schema_name=schema.schema_name,
        schema_version=schema.schema_version,
        n_test_alert_groups=result.n_alert_groups,
        metrics=result.metrics,
        results_file=results_file,
    )
