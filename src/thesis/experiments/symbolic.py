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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.config import load_mining_filter_config
from thesis.experiments.baseline import (
    ALERTBERT_METHOD,
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
from thesis.mining.token_abstraction import (
    abstract_mined_df,
    abstract_or_clauses_df,
    load_abstraction_map,
)
from thesis.mining.util import filter_mined_itemsets, filter_mined_sequences
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.experiments import SymbolicExperimentConfig, ExperimentResult
from thesis.registry.models import get_model_path, resolve_model_paths
from thesis.training.service import train_model_for_schema
from thesis.utils.runs import create_run_dir


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
    jaccard_threshold: float = 0.98,
    abstraction_map_path: Path | None = None,
    abstraction_level: int = 0,
) -> Path:
    run_dir = create_run_dir(run_name)

    print("--- Running Eclat itemset mining (AND + AND/OR) --- ")
    eclat_result = run_transaction_eclat_job(
        transactions_path=transactions_path,
        scenario_name=scenario,
        run_name=run_name,
        min_support=min_support,
        max_len=max_itemset_size,
        target_label=target_label,
        run_dir=run_dir,
        jaccard_threshold=jaccard_threshold,
    )

    print("--- Running PrefixSpan sequence mining ---")
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

    if abstraction_map_path is not None:
        print("--- Applying token abstraction ---")
        abstraction_map = load_abstraction_map(abstraction_map_path)
        n_eclat_before, n_seq_before = len(eclat_df), len(item_seq_df)
        eclat_df = abstract_mined_df(eclat_df, abstraction_map, level=abstraction_level)
        item_seq_df = abstract_mined_df(
            item_seq_df, abstraction_map, level=abstraction_level, column="sequence"
        )
        print(
            f"  itemsets {n_eclat_before}→{len(eclat_df)}, "
            f"sequences {n_seq_before}→{len(item_seq_df)} "
            f"(level={abstraction_level})"
        )
        if eclat_result.or_df is not None and not eclat_result.or_df.empty:
            n_or_before = len(eclat_result.or_df)
            eclat_result.or_df = abstract_or_clauses_df(
                eclat_result.or_df, abstraction_map, level=abstraction_level
            )
            print(f"  OR patterns {n_or_before}→{len(eclat_result.or_df)}")

    print("--- Filtering mining results ---")
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
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )

        f = mining_filters.item_sequences
        item_seq_df = filter_mined_sequences(
            item_seq_df,
            min_k=f.min_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            min_lift=f.min_lift,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )
        print(
            f"  After filtering: {len(eclat_df)} itemsets, {len(item_seq_df)} sequences"
        )

    eclat_df["mining_type"] = "itemset"
    item_seq_df = item_seq_df.rename(columns={"sequence": "itemset"})
    item_seq_df["mining_type"] = "item_sequence"

    print("--- Constructing combined dataframe from mining results ---")
    cols_to_keep = [
        "itemset",
        "mining_type",
        "support",
        "confidence_attack",
        "confidence_benign",
    ]
    or_cols_to_keep = [
        "clauses",
        "mining_type",
        "confidence_attack",
        "confidence_benign",
    ]
    eclat_df = eclat_df[[c for c in cols_to_keep if c in eclat_df.columns]]
    item_seq_df = item_seq_df[[c for c in cols_to_keep if c in item_seq_df.columns]]

    dfs_to_concat = [eclat_df, item_seq_df]
    if eclat_result.or_df is not None and not eclat_result.or_df.empty:
        or_df = eclat_result.or_df[
            [c for c in or_cols_to_keep if c in eclat_result.or_df.columns]
        ]
        dfs_to_concat.append(or_df)

    combined_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
    combined_df.to_csv(
        os.path.join(run_dir, "final_combined_mining_df.csv"), index=False
    )
    n_or = len(eclat_result.or_df) if eclat_result.or_df is not None else 0
    print(
        f"  Combined {len(eclat_df)} itemsets + {len(item_seq_df)} sequences "
        f"+ {n_or} OR patterns = {len(combined_df)} candidate features"
    )

    print("--- Building and saving symbolic schema ---")
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
) -> ExperimentResult:
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
    grouping_cache_dir = config.grouping_cache_dir
    print("[2/8] Processing alert batch...")
    _process_alert_batch(
        config.scenario,
        alerts_path,
        grouping_cache_dir if grouping_cache_dir is not None else config.cache_dir,
        grouping_mode=config.grouping.mode,
        grouping=config.grouping,
    )

    # 3. Ensure feature manifest
    print("[3/8] Checking feature manifest...")
    _ensure_feature_manifest(config.scenario)

    # 4. Build transactions from closed groups
    print("[4/8] Building transactions from cache...")
    transactions = _load_transactions(
        config.scenario, config.cache_dir, groups_cache_dir=grouping_cache_dir
    )

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
        jaccard_threshold=config.jaccard_threshold,
        abstraction_map_path=config.abstraction_map_path,
        abstraction_level=config.abstraction_level,
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
    effective_version = f"{config.model_version}_{config.schema_name.replace('+', '_')}"
    print(f"[7/8] Training '{config.model_name}' v{effective_version}...")
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

    # 8. Load full metrics and write results file
    print("[8/8] Saving experiment results...")
    _, metadata_path, _ = resolve_model_paths(
        config.scenario, config.model_name, effective_version
    )
    with metadata_path.open("r", encoding="utf-8") as f:
        full_metrics = json.load(f).get("metrics", {})

    # Enrich feature importances with feature types
    if schema.symbolic is not None:
        feature_type_map = {
            f.feature_name: {
                "mining_type": f.mining_type,
                "source_label": f.source_label,
                "clauses": "AND/OR" if f.clauses else None,
            }
            for f in schema.symbolic.features
        }

        for importance_type in ["by_coefficient", "by_permutation"]:
            if importance_type in full_metrics.get("top_feature_importances", {}):
                enriched = {}
                for feat_name, importance_val in full_metrics[
                    "top_feature_importances"
                ][importance_type].items():
                    enriched[feat_name] = {
                        "importance": importance_val,
                        "feature_info": feature_type_map.get(feat_name, {}),
                    }
                full_metrics["top_feature_importances"][importance_type] = enriched

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = _EXPERIMENTS_DIR / config.scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"symbolic_{timestamp}.json"

    grouping_params = (
        config.grouping.alertbert.model_dump()
        if config.grouping.mode == ALERTBERT_METHOD
        else None
    )
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "symbolic",
                "scenario": config.scenario,
                "timestamp": timestamp,
                "model_name": config.model_name,
                "model_version": summary.model_version,
                "schema_name": summary.schema_name,
                "schema_version": summary.schema_version,
                "grouping": {"mode": config.grouping.mode, "params": grouping_params},
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
        symbolic_schema_path=symbolic_schema_path,
        auc=summary.auc,
        n_transactions=len(df),
        n_mixed_dropped=n_mixed,
        n_features=summary.n_features,
        metrics=full_metrics,
        results_file=results_file,
        grouping_mode=config.grouping.mode,
    )
