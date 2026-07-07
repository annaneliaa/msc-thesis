"""
Symbolic experiment: full pipeline for a given scenario, including mining.

Steps:
  1. Convert raw alerts CSV to JSON format
  2. Process alert batch (tokenise + ingest into cache)
  3. Ensure feature manifest (creates base schemas if missing)
  4. Build alert_groups from closed groups and save raw JSON
  5. Mine frequent itemsets and sequences, apply quality filters,
     and register the resulting symbolic feature schema
  6. Encode alert_groups under the base+symbolic schema
  7. Train logistic regression on the encoded features
  8. Write full metrics to artifacts/experiments/<scenario>/

The mining filter config is optional. Pass a path to a YAML file (see
src/thesis/configs/) to apply quality filters after mining; omit to use
all mined patterns as features.

Mining scope (mine_frac) and train/test overlap
------------------------------------------------
AlertGroups are sorted chronologically before any split is applied.
The mine_frac and no_overlap fields on SymbolicExperimentConfig (exposed
as --mine-frac and --no-overlap in the run scripts) control which
alert_groups feed the miner and which feed training:

  mine_frac=1.0 (default)
    Mine: [0%, 100%)   Train: [0%, 70%)   Test: [70%, 100%)
    The miner sees all data; training and testing use the standard
    temporal holdout.

  mine_frac=0.7, no_overlap=False (default)
    Mine: [0%, 70%)    Train: [0%, 70%)   Test: [70%, 100%)
    Mining and training cover the same window, ensuring no test-period
    data leaks into the mined patterns.

  mine_frac=0.5, no_overlap=False (default)
    Mine: [0%, 50%)    Train: [0%, 70%)   Test: [70%, 100%)
    Mining uses only the earliest 50%; training still uses the full
    [0%, 70%) window.

  mine_frac=0.3, no_overlap=True
    Mine: [0%, 30%)    Train: [30%, 70%)  Test: [70%, 100%)
    Mining and training are strictly disjoint; patterns are discovered
    on a held-out prefix and evaluated on subsequent unseen data.

  mine_frac=0.5, no_overlap=True
    Mine: [0%, 50%)    Train: [50%, 70%)  Test: [70%, 100%)
    Same disjoint setup with a larger mining window.

In all cases test_frac (default 0.3) controls the size of the test set
and the encoding/training always operates on the full alert_group list.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from thesis.config import load_mining_filter_config
from thesis.configs import dataset_for_scenario
from thesis.experiments.baseline import (
    _EXPERIMENTS_DIR,
    _ROOT,
)
from thesis.pipeline.pipeline import (
    convert_ait_alerts_to_json,
    encode_and_cache_alert_groups,
    ensure_feature_manifest,
    ingest_ait_alert_batch,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
    resolve_mining_alert_groups_path,
)
from thesis.pipeline.pipeline import is_single_class_split as _is_single_class_split
from thesis.mining.attribute_schema_cache import mine_or_reuse_attribute_schema
from thesis.mining.cooccurrence_schema import mine_and_register_cooccurrence_schema
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.experiments import SymbolicExperimentConfig, ExperimentResult
from thesis.registry.models import get_model_path, resolve_model_paths
from thesis.training.service import train_model_for_schema
from thesis.training.util import effective_train_start


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
            or mining_filters.feature_selection.filter_cross_host_or
        ):
            feature_selection = mining_filters.feature_selection

    # 1-2. Ingest raw data into alert_groups_raw.json under config.cache_dir
    if dataset_for_scenario(config.scenario) == "cscas":
        print("[1-2/8] Ingesting CSCAS scenario...")
        ingest_cscas_scenario(cache_dir=config.cache_dir)
    else:
        print("[1/8] Converting alerts to JSON...")
        alerts_path = convert_ait_alerts_to_json(
            config.scenario, config.alerts_json_path
        )

        print("[2/8] Processing alert batch...")
        ingest_ait_alert_batch(
            config.scenario,
            alerts_path,
            config.cache_dir,
            grouping_mode=config.grouping.mode,
            grouping=config.grouping,
        )

    # 3. Ensure feature manifest
    print("[3/8] Checking feature manifest...")
    ensure_feature_manifest(config.scenario)

    # 4. Build alert_groups from closed groups
    print("[4/8] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(config.scenario, config.cache_dir)
    alert_groups_path = config.cache_dir / "alert_groups" / "alert_groups_raw.json"

    # Sort chronologically so that positional train/test split = temporal split.
    # This also ensures the mining subset and training set are drawn from the
    # same ordering regardless of how the cache stored the groups.
    alert_groups.sort(key=lambda t: t.start_ts or "")

    if config.random_split:
        import random as _random

        rng = _random.Random(config.random_seed)
        rng.shuffle(alert_groups)
        print(
            f"  [random-split] Shuffled {len(alert_groups)} alert_groups (seed={config.random_seed})"
        )

    n_total = len(alert_groups)
    n_mine = int(config.mine_frac * n_total) if config.mine_frac < 1.0 else n_total

    # When --no-overlap is set, training starts after the mine window so mining
    # and training data are strictly disjoint.  Default (overlap) always trains
    # from 0, meaning mine_frac only controls what the miner sees.
    train_start = n_mine if (config.no_overlap and config.mine_frac < 1.0) else 0

    if _is_single_class_split(
        alert_groups,
        config.test_frac,
        train_start,
        random_split=config.random_split,
        random_seed=config.random_seed,
        train_frac=config.train_frac,
    ):
        n_train = int((1 - config.test_frac) * n_total) - train_start
        n_test = n_total - int((1 - config.test_frac) * n_total)
        print(
            f"  [skip] Single-class split detected for '{config.scenario}' "
            f"({n_train} train / {n_test} test, train_start={train_start}) — skipping symbolic."
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir = (
            config.results_dir
            if config.results_dir is not None
            else _EXPERIMENTS_DIR / config.scenario
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"symbolic_{timestamp}.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": "symbolic",
                    "scenario": config.scenario,
                    "timestamp": timestamp,
                    "skipped": True,
                    "mine_frac": config.mine_frac,
                    "no_overlap": config.no_overlap,
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
            n_alert_groups=n_total,
            n_features=0,
            metrics={"single_class_split": True},
            results_file=results_file,
            grouping_mode=config.grouping.mode,
        )

    mining_alert_groups_path = resolve_mining_alert_groups_path(
        alert_groups,
        alert_groups_path,
        mine_frac=config.mine_frac,
        random_split=config.random_split,
        random_seed=config.random_seed,
    )
    if config.mine_frac < 1.0:
        split_label = "random" if config.random_split else "first"
        print(
            f"  Mining on {split_label} {n_mine}/{n_total} alert_groups (mine_frac={config.mine_frac:.2f})"
        )
        if train_start > 0:
            print(
                f"  Training on alert_groups [{train_start}, {int((1.0 - config.test_frac) * n_total)}) — no overlap with mining window"
            )

    # Invalidate the cached parquet: the symbolic schema is re-mined every run
    # and alert_groups are now in sorted order, so any existing parquet is stale.
    stale_parquet = (
        config.cache_dir
        / "alert_groups"
        / f"alert_groups_{config.schema_name.replace('+', '_')}.parquet"
    )
    if stale_parquet.exists():
        stale_parquet.unlink()
        print(f"  Removed stale encoded parquet: {stale_parquet.name}")

    # 5. Mine and register symbolic schema
    print("[5/8] Mining alert_groups for symbolic schema...")
    mining_run_dir: Path | None = None
    if config.prebuilt_symbolic_schema_path is not None:
        print(
            f"  [skip] Reusing prebuilt schema: {config.prebuilt_symbolic_schema_path}"
        )
        symbolic_schema_path = config.prebuilt_symbolic_schema_path
        mining_stats: dict = {"prebuilt": True}
    else:
        run_name = f"symbolic_{config.scenario}"
        if config.mining_strategy == "attribute":
            print(
                "  [mining_strategy=attribute] Using per-alert-group attribute mining."
            )
            symbolic_schema_path, mining_run_dir, mining_stats = (
                mine_or_reuse_attribute_schema(
                    scenario=config.scenario,
                    alert_groups_path=mining_alert_groups_path,
                    run_name=run_name,
                    attribute_mining_config=config.attribute_mining_config,
                )
            )
        elif config.mining_strategy == "cooccurrence":
            n_attack_in_mine = sum(
                1 for t in alert_groups[:n_mine] if t.group_label == "attack"
            )
            if n_attack_in_mine == 0:
                print(
                    f"  [info] No attack alert_groups in mine window ({n_mine}/{n_total}) — skipping attack mining pass."
                )
            has_sequence_data = any(t.sorted_items for t in alert_groups[:n_mine])
            symbolic_schema_path, mining_run_dir, mining_stats = (
                mine_and_register_cooccurrence_schema(
                    scenario=config.scenario,
                    alert_groups_path=mining_alert_groups_path,
                    run_name=run_name,
                    min_support=config.min_support,
                    max_itemset_size=config.max_itemset_size,
                    max_seq_len=config.max_seq_len,
                    target_label=config.target_label,
                    filter_config=config.filter_config,
                    jaccard_threshold=config.jaccard_threshold,
                    abstraction_map_path=config.abstraction_map_path,
                    abstraction_level=config.abstraction_level,
                    run_attack_pass=n_attack_in_mine > 0,
                    has_sequence_data=has_sequence_data,
                )
            )
        else:
            raise ValueError(
                f"Unsupported mining_strategy: {config.mining_strategy!r}. "
                "Expected 'cooccurrence' or 'attribute'."
            )

    # 6. Encode under base+symbolic schema
    print(f"[6/8] Encoding alert_groups (schema='{config.schema_name}')...")
    df, schema = encode_and_cache_alert_groups(
        config.scenario,
        alert_groups,
        config.schema_name,
        config.cache_dir,
        feature_selection=feature_selection,
    )

    # 7. Train model
    grouping_tag = config.grouping.mode.replace("-", "_")
    effective_version = (
        f"{config.model_version}_{config.schema_name.replace('+', '_')}_{grouping_tag}"
    )
    print(f"[7/8] Training '{config.model_name}' v{effective_version}...")
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
        train_start=train_start,
        train_frac=config.train_frac,
        random_split=config.random_split,
        random_seed=config.random_seed,
    )

    # 8. Load full metrics and write results file
    print("[8/8] Saving experiment results...")
    _, metadata_path, _ = resolve_model_paths(
        config.scenario, config.model_name, effective_version
    )
    if summary.single_class_split:
        full_metrics = {"single_class_split": True}
    else:
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
    results_dir = (
        config.results_dir
        if config.results_dir is not None
        else _EXPERIMENTS_DIR / config.scenario
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"symbolic_{timestamp}.json"

    grouping_params = None
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "symbolic",
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
                "symbolic_schema_path": str(symbolic_schema_path),
                "mining": {
                    "min_support": config.min_support,
                    "max_itemset_size": config.max_itemset_size,
                    "max_seq_len": config.max_seq_len,
                    "target_label": config.target_label,
                    "mine_frac": config.mine_frac,
                    "no_overlap": config.no_overlap,
                    "train_start": train_start,
                    "filter_config": (
                        str(config.filter_config)
                        if config.filter_config is not None
                        else None
                    ),
                    **mining_stats,
                },
                "n_alert_groups": len(df),
                "n_mixed_dropped": n_mixed,
                "n_features": summary.n_features,
                "test_frac": config.test_frac,
                "train_frac": config.train_frac
                if config.train_frac is not None
                else 1.0 - config.test_frac,
                "n_train": summary.test_idx_start
                - effective_train_start(
                    train_start, config.train_frac, n_total, summary.test_idx_start
                ),
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
        symbolic_schema_path=symbolic_schema_path,
        mining_run_dir=mining_run_dir,
        auc=summary.auc,
        n_alert_groups=len(df),
        n_mixed_dropped=n_mixed,
        n_features=summary.n_features,
        metrics=full_metrics,
        results_file=results_file,
        grouping_mode=config.grouping.mode,
    )
