"""
Symbolic experiment: full pipeline for a given scenario, including mining.

Steps:
  1. Convert raw alerts CSV to JSON
  2. Process alert batch (tokenise + ingest into cache)
  3. Ensure feature manifest (creates base schemas if missing)
  4. Build transactions from closed groups and save raw JSON
  5. Mine frequent itemsets and sequences, apply quality filters,
     and register the resulting symbolic feature schema
  6. Encode transactions under the base+symbolic schema
  7. Train logistic regression on the encoded features
  8. Write full metrics to artifacts/experiments/<scenario>/

The mining filter config is optional. Pass a path to a YAML file (see
src/thesis/configs/) to apply quality filters after mining; omit to use
all mined patterns as features.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.config import load_mining_filter_config
from thesis.experiments.baseline import (
    _EXPERIMENTS_DIR,
    _ROOT,
    _convert_alerts_to_json,
    _encode_transactions,
    _ensure_feature_manifest,
    _load_transactions,
    _process_alert_batch,
)
from thesis.features.service import build_persist_and_register_symbolic_schema
from thesis.mining.itemset_mining_job import run_transaction_eclat_job
from thesis.mining.sequence_mining_job import run_transaction_prefixspan_job
from thesis.mining.util import filter_mined_itemsets, filter_mined_sequences
from thesis.paths import CACHE_DIR, ensure_artifact_dirs
from thesis.schemas.mining import FeatureSelectionConfig
from thesis.registry.models import get_model_path, resolve_model_paths
from thesis.training.service import train_model_for_schema
from thesis.utils.runs import create_run_dir


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------


@dataclass
class SymbolicExperimentConfig:
    scenario: str
    # mining
    min_support: float = 0.05
    max_itemset_size: int = 3
    max_seq_len: int = 5
    target_label: str = "benign"
    filter_config: Path | None = None
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    # training
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base+symbolic"
    test_frac: float = 0.3
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)


@dataclass
class SymbolicExperimentResult:
    scenario: str
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    symbolic_schema_path: Path
    auc: float
    n_transactions: int
    n_features: int
    metrics: dict
    results_file: Path


# ---------------------------------------------------------------------------
# Private step helpers
# ---------------------------------------------------------------------------


def _mine_and_register_symbolic_schema(
    scenario: str,
    transactions_path: Path,
    run_name: str,
    min_support: float,
    max_itemset_size: int,
    max_seq_len: int,
    target_label: str,
    filter_config: Path | None,
) -> Path:
    run_dir = create_run_dir(run_name)

    print("  Running Eclat itemset mining...")
    eclat_result = run_transaction_eclat_job(
        transactions_path=transactions_path,
        scenario_name=scenario,
        run_name=run_name,
        min_support=min_support,
        max_len=max_itemset_size,
        target_label=target_label,
        run_dir=run_dir,
    )

    print("  Running PrefixSpan sequence mining...")
    item_seq_result = run_transaction_prefixspan_job(
        transactions_path=transactions_path,
        scenario_name=scenario,
        run_name=run_name,
        min_support=min_support,
        max_len=max_seq_len,
        target_label=target_label,
        run_dir=run_dir,
    )

    eclat_df = eclat_result.mined_df.copy()
    item_seq_df = item_seq_result.mined_df.copy()

    if filter_config is not None:
        if not filter_config.is_absolute():
            filter_config = _ROOT / filter_config
        mining_filters = load_mining_filter_config(filter_config)

        f = mining_filters.itemsets
        eclat_df = filter_mined_itemsets(
            eclat_df,
            min_k=f.min_k,
            max_k=f.max_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            remove_subsumed=f.remove_subsumed,
        )

        f = mining_filters.item_sequences
        item_seq_df = filter_mined_sequences(
            item_seq_df,
            min_k=f.min_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            min_lift=f.min_lift,
            remove_subsumed=f.remove_subsumed,
        )
        print(
            f"  After filtering: {len(eclat_df)} itemsets, "
            f"{len(item_seq_df)} sequences"
        )

    eclat_df["mining_type"] = "itemset"
    item_seq_df = item_seq_df.rename(columns={"sequence": "itemset"})
    item_seq_df["mining_type"] = "item_sequence"

    cols_to_keep = [
        "itemset",
        "mining_type",
        "support",
        "confidence_attack",
        "confidence_benign",
    ]
    eclat_df = eclat_df[[c for c in cols_to_keep if c in eclat_df.columns]]
    item_seq_df = item_seq_df[[c for c in cols_to_keep if c in item_seq_df.columns]]

    combined_df = pd.concat([eclat_df, item_seq_df], axis=0, ignore_index=True)
    combined_df.to_csv(os.path.join(run_dir, "combined_mining_df.csv"), index=False)
    print(
        f"  Combined {len(eclat_df)} itemsets + {len(item_seq_df)} sequences "
        f"= {len(combined_df)} candidate features"
    )

    schema_path = build_persist_and_register_symbolic_schema(
        df=combined_df,
        scenario_name=scenario,
        source_label=target_label,
        schema_name="symbolic",
        root_dir=_ROOT / "artifacts" / "features",
    )
    print(f"  Symbolic schema registered → {schema_path}")
    return schema_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_symbolic_experiment(
    config: SymbolicExperimentConfig,
) -> SymbolicExperimentResult:
    ensure_artifact_dirs()
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Symbolic] Scenario: '{config.scenario}'")

    # Resolve feature selection from filter config file if not already set
    feature_selection = config.feature_selection
    if config.filter_config is not None:
        resolved_filter_config = config.filter_config
        if not resolved_filter_config.is_absolute():
            resolved_filter_config = _ROOT / resolved_filter_config
        mining_filters = load_mining_filter_config(resolved_filter_config)
        if (
            mining_filters.feature_selection.top_k is not None
            or mining_filters.feature_selection.min_utility_score is not None
        ):
            feature_selection = mining_filters.feature_selection

    # 1. Convert alerts CSV → JSON
    print("[1/8] Converting alerts to JSON...")
    alerts_path = _convert_alerts_to_json(config.scenario)

    # 2. Tokenise + ingest into cache
    print("[2/8] Processing alert batch...")
    _process_alert_batch(config.scenario, alerts_path, config.cache_dir)

    # 3. Ensure feature manifest
    print("[3/8] Checking feature manifest...")
    _ensure_feature_manifest(config.scenario)

    # 4. Build transactions from closed groups
    print("[4/8] Building transactions from cache...")
    transactions = _load_transactions(config.scenario, config.cache_dir)
    transactions_path = (
        config.cache_dir / config.scenario / "transactions" / "transactions_raw.json"
    )

    # 5. Mine and register symbolic schema
    print("[5/8] Mining transactions...")
    run_name = f"symbolic_{config.scenario}"
    symbolic_schema_path = _mine_and_register_symbolic_schema(
        scenario=config.scenario,
        transactions_path=transactions_path,
        run_name=run_name,
        min_support=config.min_support,
        max_itemset_size=config.max_itemset_size,
        max_seq_len=config.max_seq_len,
        target_label=config.target_label,
        filter_config=config.filter_config,
    )

    # 6. Encode under base+symbolic schema
    print(f"[6/8] Encoding transactions (schema='{config.schema_name}')...")
    df, schema = _encode_transactions(
        config.scenario,
        transactions,
        config.schema_name,
        config.cache_dir,
        feature_selection=feature_selection,
    )

    # 7. Train model
    print(f"[7/8] Training '{config.model_name}' v{config.model_version}...")
    y = df["tx_label"].map({"benign": 0, "attack": 1})
    X = df.drop(columns=["tx_label"])
    output_dir = get_model_path(config.model_name, config.model_version)

    summary = train_model_for_schema(
        X=X,
        y=y,
        schema=schema,
        model_name=config.model_name,
        model_version=config.model_version,
        output_dir=output_dir,
        test_frac=config.test_frac,
    )

    # 8. Load full metrics and write results file
    print("[8/8] Saving experiment results...")
    _, metadata_path, _ = resolve_model_paths(config.model_name, config.model_version)
    with metadata_path.open("r", encoding="utf-8") as f:
        full_metrics = json.load(f).get("metrics", {})

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = _EXPERIMENTS_DIR / config.scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"symbolic_{timestamp}.json"

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "symbolic",
                "scenario": config.scenario,
                "timestamp": timestamp,
                "model_name": config.model_name,
                "model_version": config.model_version,
                "schema_name": summary.schema_name,
                "schema_version": summary.schema_version,
                "symbolic_schema_path": str(symbolic_schema_path),
                "mining": {
                    "min_support": config.min_support,
                    "max_itemset_size": config.max_itemset_size,
                    "max_seq_len": config.max_seq_len,
                    "target_label": config.target_label,
                    "filter_config": (
                        str(config.filter_config)
                        if config.filter_config is not None
                        else None
                    ),
                },
                "n_transactions": len(df),
                "n_features": summary.n_features,
                "test_size": summary.test_size,
                "metrics": full_metrics,
            },
            f,
            indent=2,
        )

    print(f"  AUC: {summary.auc:.4f}")
    print(f"  Results → {results_file}")

    return SymbolicExperimentResult(
        scenario=config.scenario,
        model_name=config.model_name,
        model_version=config.model_version,
        schema_name=summary.schema_name,
        schema_version=summary.schema_version,
        symbolic_schema_path=symbolic_schema_path,
        auc=summary.auc,
        n_transactions=len(df),
        n_features=summary.n_features,
        metrics=full_metrics,
        results_file=results_file,
    )
