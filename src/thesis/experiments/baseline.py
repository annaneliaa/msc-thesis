"""
Baseline experiment: full pipeline for a given scenario.

Steps:
  1. Convert raw alerts CSV to JSON
  2. Process alert batch (tokenise + ingest into cache)
  3. Build alert_groups from closed groups and save raw JSON
  4. Encode alert_groups under the baseline feature schema
  5. Train logistic regression on the encoded features
  6. Write full metrics to artifacts/experiments/<scenario>/

Prerequisite: the baseline feature schema must already be registered in
  artifacts/features/<scenario>/manifest.json
"""

from __future__ import annotations

import json
import random as _random
from datetime import datetime, timezone
from pathlib import Path

from thesis.paths import ensure_artifact_dirs
from thesis.configs import dataset_for_scenario
from thesis.schemas.experiments import BaselineExperimentConfig, ExperimentResult
from thesis.pipeline.pipeline import (
    convert_ait_alerts_to_json,
    encode_and_cache_alert_groups,
    ensure_feature_manifest,
    ingest_ait_alert_batch,
    ingest_cscas_scenario,
    is_single_class_split,
    load_or_build_alert_groups,
)
from thesis.registry.models import get_model_path, resolve_model_paths
from thesis.training.service import train_model_for_schema

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_baseline_experiment(
    config: BaselineExperimentConfig,
) -> ExperimentResult:
    ensure_artifact_dirs()
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Baseline] Scenario: '{config.scenario}'")

    # 1-2. Ingest raw data into alert_groups_raw.json under config.cache_dir
    if dataset_for_scenario(config.scenario) == "cscas":
        print("[1-2/7] Ingesting CSCAS scenario...")
        ingest_cscas_scenario(cache_dir=config.cache_dir)
    else:
        # 1. Convert alerts CSV → JSON
        print("[1/7] Converting alerts to JSON...")
        alerts_path = convert_ait_alerts_to_json(
            config.scenario, config.alerts_json_path
        )

        # 2. Tokenise + ingest into cache
        print("[2/7] Processing alert batch...")
        ingest_ait_alert_batch(
            config.scenario,
            alerts_path,
            config.cache_dir,
            grouping_mode=config.grouping.mode,
            grouping=config.grouping,
        )

    # 3. Ensure feature manifest exists (creates base + base+dynamic schemas if missing)
    print("[3/7] Checking feature manifest...")
    ensure_feature_manifest(config.scenario)

    # 4. Build alert_groups from closed groups
    print("[4/7] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(config.scenario, config.cache_dir)

    # Sort chronologically first, then shuffle if requested.
    # The encoding will preserve this order; prepare_training_frame skips the
    # timestamp sort when random_split=True so the shuffled order is kept intact.
    if config.random_split:
        alert_groups.sort(key=lambda t: t.start_ts or "")
        _random.Random(config.random_seed).shuffle(alert_groups)
        print(
            f"  [random-split] Shuffled {len(alert_groups)} alert_groups (seed={config.random_seed})"
        )
        # Invalidate the cached parquet so it is re-encoded in shuffled order.
        stale = (
            config.cache_dir
            / "alert_groups"
            / f"alert_groups_{config.schema_name.replace('+', '_')}.parquet"
        )
        if stale.exists():
            stale.unlink()
            print(f"  Removed stale parquet for random-split encoding: {stale.name}")

    if is_single_class_split(
        alert_groups,
        config.test_frac,
        random_split=config.random_split,
        random_seed=config.random_seed,
        train_frac=config.train_frac,
    ):
        print(
            f"  [skip] Single-class split detected for '{config.scenario}' "
            f"({int((1-config.test_frac)*len(alert_groups))} train / "
            f"{len(alert_groups)-int((1-config.test_frac)*len(alert_groups))} test) — skipping baseline."
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir = (
            config.results_dir
            if config.results_dir is not None
            else _EXPERIMENTS_DIR / config.scenario
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"baseline_{timestamp}.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": "baseline",
                    "scenario": config.scenario,
                    "timestamp": timestamp,
                    "skipped": True,
                    "test_frac": config.test_frac,
                    "train_frac": config.train_frac
                    if config.train_frac is not None
                    else 1.0 - config.test_frac,
                    "metrics": {"single_class_split": True},
                },
                f,
                indent=2,
            )
        return ExperimentResult(
            scenario=config.scenario,
            model_name=config.model_name,
            model_version=config.model_version,
            schema_name=config.schema_name,
            schema_version="skipped",
            auc=float("nan"),
            n_alert_groups=len(alert_groups),
            n_features=0,
            metrics={"single_class_split": True},
            results_file=results_file,
            grouping_mode=config.grouping.mode,
        )

    # 5. Encode under baseline schema
    print(f"[5/7] Encoding alert_groups (schema='{config.schema_name}')...")
    df, schema = encode_and_cache_alert_groups(
        config.scenario,
        alert_groups,
        config.schema_name,
        config.cache_dir,
    )

    # 6. Train model
    grouping_tag = config.grouping.mode.replace("-", "_")
    effective_version = (
        f"{config.model_version}_{config.schema_name.replace('+', '_')}_{grouping_tag}"
    )
    print(f"[6/7] Training '{config.model_name}' v{effective_version}...")
    y = df["group_label"].map({"benign": 0, "attack": 1})
    X = df.drop(columns=["group_label"])
    mask = y.notna()
    n_mixed = int((~mask).sum())
    if n_mixed:
        print(
            f"  [warn] Dropping {n_mixed} alert_groups with unlabelled/mixed group_label"
        )
        X, y = X[mask], y[mask]
    output_dir = get_model_path(config.scenario, config.model_name, effective_version)

    summary = train_model_for_schema(
        X=X,
        y=y,
        schema=schema,
        model_name=config.model_name,
        model_version=effective_version,
        output_dir=output_dir,
        test_frac=config.test_frac,
        train_frac=config.train_frac,
        random_split=config.random_split,
        random_seed=config.random_seed,
    )

    # 7. Load full metrics from saved metadata and write results file
    print("[7/7] Saving experiment results...")
    _, metadata_path, _ = resolve_model_paths(
        config.scenario, config.model_name, effective_version
    )
    if summary.single_class_split:
        full_metrics = {"single_class_split": True}
    else:
        with metadata_path.open("r", encoding="utf-8") as f:
            full_metrics = json.load(f).get("metrics", {})

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = (
        config.results_dir
        if config.results_dir is not None
        else _EXPERIMENTS_DIR / config.scenario
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"baseline_{timestamp}.json"

    grouping_params = None
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "baseline",
                "scenario": config.scenario,
                "timestamp": timestamp,
                "alerts_source": str(config.alerts_json_path)
                if config.alerts_json_path
                else "alerts.json",
                "model_name": config.model_name,
                "model_version": summary.model_version,
                "schema_name": summary.schema_name,
                "schema_version": summary.schema_version,
                "grouping": {"mode": config.grouping.mode, "params": grouping_params},
                "n_alert_groups": len(df),
                "n_mixed_dropped": n_mixed,
                "n_features": summary.n_features,
                "test_frac": config.test_frac,
                "train_frac": config.train_frac
                if config.train_frac is not None
                else 1.0 - config.test_frac,
                "n_train": summary.test_idx_start,
                "n_test": summary.test_size,
                "metrics": full_metrics,
            },
            f,
            indent=2,
        )

    print(f"  AUC: {summary.auc:.4f}")
    print(f"  Results → {results_file}")

    return ExperimentResult(
        scenario=config.scenario,
        model_name=config.model_name,
        model_version=summary.model_version,
        schema_name=summary.schema_name,
        schema_version=summary.schema_version,
        auc=summary.auc,
        n_alert_groups=len(df),
        n_mixed_dropped=n_mixed,
        n_features=summary.n_features,
        metrics=full_metrics,
        results_file=results_file,
        grouping_mode=config.grouping.mode,
    )
