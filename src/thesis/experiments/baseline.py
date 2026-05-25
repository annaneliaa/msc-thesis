"""
Baseline experiment: full pipeline for a given scenario.

Steps:
  1. Convert raw alerts CSV to JSON
  2. Process alert batch (tokenise + ingest into cache)
  3. Build transactions from closed groups and save raw JSON
  4. Encode transactions under the baseline feature schema
  5. Train logistic regression on the encoded features
  6. Write full metrics to artifacts/experiments/<scenario>/

Prerequisite: the baseline feature schema must already be registered in
  artifacts/features/<scenario>/manifest.json
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.encoders.service import encode_transactions_for_schema
from thesis.features.manifest import initialize_feature_manifest
from thesis.features.schema_registry import FeatureSchemaRegistry
from thesis.features.util import select_symbolic_features
from thesis.config import GroupingConfig
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.mining import FeatureSelectionConfig
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.preprocessing.mining_prep import build_transactions
from thesis.preprocessing.group_alerts import ALERTBERT_METHOD, FIXED_WINDOW_METHOD
from thesis.schemas.preprocessing import Transaction
from thesis.schemas.experiments import BaselineExperimentConfig, ExperimentResult
from thesis.preprocessing.service import (
    build_grouper,
    process_alert_batch,
    select_groups_from_cache,
)
from thesis.registry.models import get_model_path, resolve_model_paths
from thesis.training.service import train_model_for_schema

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments"


# ---------------------------------------------------------------------------
# Private step helpers
# ---------------------------------------------------------------------------


def _ensure_feature_manifest(scenario: str) -> None:
    manifest_path = _ROOT / "artifacts" / "features" / scenario / "manifest.json"
    if manifest_path.exists():
        print(f"  [skip] Feature manifest already exists at {manifest_path}")
        return
    print(f"  Feature manifest not found for '{scenario}', initialising...")
    initialize_feature_manifest(
        scenario_name=scenario,
        root_dir=_ROOT / "artifacts" / "features",
    )
    print(f"  Created feature manifest at {manifest_path}")


def _convert_alerts_to_json(scenario: str) -> Path:
    input_path = _ROOT / "data" / "alerts_csv" / f"{scenario}_alerts.txt"
    output_dir = _ROOT / "artifacts" / "processed-data" / scenario
    output_path = output_dir / "alerts.json"

    if output_path.exists():
        print(f"  [skip] alerts.json already exists at {output_path}")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw alerts file not found: {input_path}\n"
            "Place the alerts CSV in data/alerts_csv/ before running."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    alerts: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alerts.append(
                {
                    "time": int(row["time"]),
                    "name": row["name"],
                    "ip": row["ip"],
                    "host": row["host"],
                    "short": row["short"],
                    "time_label": row["time_label"],
                    "event_label": row["event_label"],
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    print(f"  Wrote {len(alerts)} alerts → {output_path}")
    return output_path


def _process_alert_batch(
    scenario: str,
    alerts_path: Path,
    cache_dir: Path,
    grouping_mode: str = FIXED_WINDOW_METHOD,
    grouping: GroupingConfig | None = None,
) -> None:
    alert_store_dir = cache_dir / scenario / "alerts"
    if alert_store_dir.exists() and any(alert_store_dir.glob("*.json")):
        print(f"  [skip] Alert cache already populated at {alert_store_dir}")
        return

    with alerts_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    grouper = build_grouper(grouping) if grouping is not None else None
    cache = TokenCache(cache_dir=cache_dir, scenario=scenario)
    ingestor = CacheIngestor(cache=cache)
    count = process_alert_batch(
        rows=payload,
        scenario=scenario,
        ingestor=ingestor,
        grouping_mode=grouping_mode,
        grouper=grouper,
    )
    print(f"  Processed {count} alerts into cache.")


def _load_transactions(
    scenario: str, cache_dir: Path, groups_cache_dir: Path | None = None
) -> list:
    out_dir = cache_dir / scenario / "transactions"
    out_path = out_dir / "transactions_raw.json"

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            serialized = json.load(f)
        if serialized:
            print(f"  [skip] Loading transactions from existing {out_path}")
            transactions = [
                Transaction(
                    transaction_id=t["transaction_id"],
                    group_id=t["group_id"],
                    method=t["method"],
                    start_ts=t["start_ts"],
                    end_ts=t["end_ts"],
                    n_alerts=t["n_alerts"],
                    alert_ids=t["alert_ids"],
                    abs_items=set(t["abs_items"]),
                    raw_items=set(t["raw_items"])
                    if t["raw_items"] is not None
                    else None,
                    sorted_items=[set(s) for s in t["sorted_items"]],
                    alert_ips=set(t["alert_ips"]),
                    tx_label=t["tx_label"],
                    alert_labels=set(t["alert_labels"])
                    if t["alert_labels"] is not None
                    else None,
                    weight=t["weight"],
                )
                for t in serialized
            ]
            print(f"  Loaded {len(transactions)} transactions from cache.")
            return transactions
        print(f"  [warn] {out_path} is empty, rebuilding transactions...")

    effective_groups_dir = (
        groups_cache_dir if groups_cache_dir is not None else cache_dir
    )
    cache = TokenCache(cache_dir=effective_groups_dir, scenario=scenario)
    snapshots = select_groups_from_cache(
        cache=cache,
        allowed_methods=None,
        limit=None,
        min_start_ts=None,
        max_end_ts=None,
        require_closed=True,
    )
    transactions = build_transactions(snapshots)

    out_dir.mkdir(parents=True, exist_ok=True)
    serialized = [
        {
            "transaction_id": t.transaction_id,
            "group_id": t.group_id,
            "method": t.method,
            "start_ts": t.start_ts,
            "end_ts": t.end_ts,
            "n_alerts": t.n_alerts,
            "alert_ids": t.alert_ids,
            "abs_items": sorted(list(t.abs_items)),
            "raw_items": sorted(list(t.raw_items)) if t.raw_items is not None else None,
            "sorted_items": [sorted(s) for s in t.sorted_items],
            "alert_ips": sorted(list(t.alert_ips)),
            "tx_label": t.tx_label,
            "alert_labels": (
                sorted(list(t.alert_labels)) if t.alert_labels is not None else None
            ),
            "weight": t.weight,
        }
        for t in transactions
    ]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)

    print(f"  Built {len(transactions)} transactions → {out_path}")
    return transactions


def _encode_transactions(
    scenario: str,
    transactions: list,
    schema_name: str,
    cache_dir: Path,
    feature_selection: FeatureSelectionConfig | None = None,
) -> tuple[pd.DataFrame, object]:
    safe_name = schema_name.replace("+", "_").replace("/", "_")
    out_path = (
        cache_dir / scenario / "transactions" / f"transactions_{safe_name}.parquet"
    )

    registry = FeatureSchemaRegistry(root_dir=_ROOT / "artifacts" / "features")
    schema = registry.load(
        scenario_name=scenario,
        schema_name=schema_name,
        schema_version=None,
    )

    if out_path.exists() and feature_selection is None:
        print(f"  [skip] Loading encoded transactions from existing {out_path}")
        df = pd.read_parquet(out_path)
        print(f"  Loaded {len(df)} transactions from parquet.")
        return df, schema

    if feature_selection is not None and (
        feature_selection.top_k is not None
        or feature_selection.min_utility_score is not None
    ):
        before = len(schema.symbolic.features) if schema.symbolic else 0
        schema = select_symbolic_features(schema, feature_selection)
        after = len(schema.symbolic.features) if schema.symbolic else 0
        print(f"  Feature selection: {before} → {after} symbolic features")

    print("Loaded schema. Encoding transaction data under schema...")
    feature_df = encode_transactions_for_schema(
        transactions=transactions,
        schema=schema,
        top_k=None,
    )
    meta_df = pd.DataFrame(
        [
            {
                "transaction_id": t.transaction_id,
                "tx_label": t.tx_label,
                "n_alerts": t.n_alerts,
                "weight": t.weight,
            }
            for t in transactions
        ]
    )
    df = pd.concat(
        [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1,
    )

    df.to_parquet(out_path, index=False)
    print(f"  Encoded {len(df)} transactions under schema '{schema_name}' → {out_path}")
    return df, schema


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_baseline_experiment(
    config: BaselineExperimentConfig,
) -> ExperimentResult:
    ensure_artifact_dirs()
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Baseline] Scenario: '{config.scenario}'")

    # 1. Convert alerts CSV → JSON
    print("[1/6] Converting alerts to JSON...")
    alerts_path = _convert_alerts_to_json(config.scenario)

    # 2. Tokenise + ingest into cache
    grouping_cache_dir = config.grouping_cache_dir
    print("[2/7] Processing alert batch...")
    _process_alert_batch(
        config.scenario,
        alerts_path,
        grouping_cache_dir if grouping_cache_dir is not None else config.cache_dir,
        grouping_mode=config.grouping.mode,
        grouping=config.grouping,
    )

    # 3. Ensure feature manifest exists (creates base + base+dynamic schemas if missing)
    print("[3/7] Checking feature manifest...")
    _ensure_feature_manifest(config.scenario)

    # 4. Build transactions from closed groups
    print("[4/7] Building transactions from cache...")
    transactions = _load_transactions(
        config.scenario, config.cache_dir, groups_cache_dir=grouping_cache_dir
    )

    # 5. Encode under baseline schema
    print(f"[5/7] Encoding transactions (schema='{config.schema_name}')...")
    df, schema = _encode_transactions(
        config.scenario, transactions, config.schema_name, config.cache_dir
    )

    # 6. Train model
    grouping_tag = config.grouping.mode.replace("-", "_")
    effective_version = (
        f"{config.model_version}_{config.schema_name.replace('+', '_')}_{grouping_tag}"
    )
    print(f"[6/7] Training '{config.model_name}' v{effective_version}...")
    y = df["tx_label"].map({"benign": 0, "attack": 1})
    X = df.drop(columns=["tx_label"])
    mask = y.notna()
    n_mixed = int((~mask).sum())
    if n_mixed:
        print(
            f"  [warn] Dropping {n_mixed} transactions with unlabelled/mixed tx_label"
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

    grouping_params = (
        config.grouping.alertbert.model_dump()
        if config.grouping.mode == ALERTBERT_METHOD
        else None
    )
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "baseline",
                "scenario": config.scenario,
                "timestamp": timestamp,
                "model_name": config.model_name,
                "model_version": summary.model_version,
                "schema_name": summary.schema_name,
                "schema_version": summary.schema_version,
                "grouping": {"mode": config.grouping.mode, "params": grouping_params},
                "n_transactions": len(df),
                "n_mixed_dropped": n_mixed,
                "n_features": summary.n_features,
                "test_size": summary.test_size,
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
        n_transactions=len(df),
        n_mixed_dropped=n_mixed,
        n_features=summary.n_features,
        metrics=full_metrics,
        results_file=results_file,
        grouping_mode=config.grouping.mode,
    )
