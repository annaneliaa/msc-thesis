from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.paths import ensure_artifact_dirs
from thesis.schemas.features import AttributePredicate
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import AttributeMiningConfig, MiningMetadata
from thesis.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    log_params,
    set_tags,
    start_run,
)
from thesis.utils.runs import create_run_dir, save_dataframe_artifact, write_manifest

from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    fit_and_extract_rules,
)


@dataclass(slots=True)
class AttributeMiningJobResult:
    run_dir: Path
    mined_df: pd.DataFrame
    scenario_name: str
    predicates: list[AttributePredicate] = field(default_factory=list)


def _load_labeled_alert_groups(alert_groups_path: str | Path) -> list[AlertGroup]:
    from thesis.pipeline.pipeline import load_alert_groups_json

    alert_groups = load_alert_groups_json(Path(alert_groups_path))
    labeled = [tx for tx in alert_groups if tx.group_label in ("benign", "attack")]
    n_dropped = len(alert_groups) - len(labeled)
    if n_dropped:
        print(
            f"  [warn] Dropping {n_dropped} unlabeled/mixed alert_groups "
            "before attribute mining"
        )
    return labeled


def run_alert_group_attribute_mining_job(
    alert_groups_path: str | Path,
    scenario_name: str,
    run_name: str = "debug",
    config: AttributeMiningConfig | None = None,
    run_dir: Path | None = None,
) -> AttributeMiningJobResult:
    """
    Two-stage per-alert-group attribute mining:
      Step 1: brute-force contrast-set stats over categorical predicates
              (singles + pairs), filtered down to discriminative survivors.
      Step 2: a shallow decision tree fit on those survivors + numeric base
              features jointly, with leaf paths extracted as rules.

    Unlike the itemset/sequence co-occurrence jobs, this operates directly on
    AlertGroup records (one per signature x external-IP group) -- there is no
    cross-group basket to construct.
    """
    ensure_artifact_dirs()
    config = config or AttributeMiningConfig()

    print("Starting alert_group attribute mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        if run_dir is None:
            run_dir = create_run_dir(run_name)
        run_dir = run_dir / "attribute_mining"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_tags(
            {
                "stage": "mining",
                "component": "alert_group-attribute-mining",
                "algorithm": "contrast_set+decision_tree",
                "run_name": run_name,
                "scenario_name": scenario_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "alert_group_attribute_mining",
                "scenario_name": scenario_name,
                "min_attack_coverage": config.contrast.min_attack_coverage,
                "min_benign_coverage": config.contrast.min_benign_coverage,
                "min_growth_rate": config.contrast.min_growth_rate,
                "min_growth_rate_attack": config.contrast.min_growth_rate_attack,
                "max_p_value": config.contrast.max_p_value,
                "max_depth": config.tree.max_depth,
                "max_depth_attack": config.tree.max_depth_attack,
                "min_samples_leaf": config.tree.min_samples_leaf,
                "class_weight": config.tree.class_weight,
                "input_type": "AlertGroup",
                "output_format": "csv",
            }
        )

        alert_groups = _load_labeled_alert_groups(alert_groups_path)
        print(
            f"  Loaded {len(alert_groups)} labeled alert_groups for attribute mining."
        )

        # --- Step 1: contrast-set mining over categorical predicates ---
        # Single per-group pass: X_cat, X_num, and column_predicate_map are
        # reused as-is in Step 2 below, so compute_candidate_attribute_features
        # is never re-derived from the raw alert_groups a second time.
        X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(
            alert_groups
        )
        contrast_stats_df = compute_predicate_contrast_stats(
            X_cat, y, column_predicate_map
        )
        save_dataframe_artifact(contrast_stats_df, run_dir, "contrast_stats_all")

        survivors_df = filter_contrast_survivors(
            contrast_stats_df,
            min_attack_coverage=config.contrast.min_attack_coverage,
            min_benign_coverage=config.contrast.min_benign_coverage,
            min_growth_rate=config.contrast.min_growth_rate,
            min_growth_rate_attack=config.contrast.min_growth_rate_attack,
            max_p_value=config.contrast.max_p_value,
        )
        save_dataframe_artifact(survivors_df, run_dir, "contrast_survivors")
        print(
            f"  Step 1: {len(survivors_df)}/{len(contrast_stats_df)} categorical "
            "predicates/pairs survived the contrast-set filter."
        )

        surviving_cols = surviving_single_columns(survivors_df)

        # --- Step 2: decision tree over survivors + numeric base features ---
        X_train, kept_predicate_map = build_training_matrix(
            X_cat, X_num, column_predicate_map, surviving_cols
        )
        leaf_rules_df, predicates = fit_and_extract_rules(
            X_train, y, kept_predicate_map, config.tree
        )
        save_dataframe_artifact(leaf_rules_df, run_dir, "decision_tree_rules")
        tree_mode = (
            "two-tree" if config.tree.max_depth_attack is not None else "single-tree"
        )
        print(
            f"  Step 2: extracted {len(leaf_rules_df)} leaf rules ({tree_mode} mode)."
        )

        # Step 1's survivors_df still needs its own source_label -- fit_and_
        # extract_rules already tags leaf_rules_df's (it needs the tag
        # internally in two-tree mode, to know which leaves to keep). Unlike
        # the cooccurrence pipeline (separate benign-mining and attack-mining
        # passes, each explicitly tagged before concatenation), Step 1 mines
        # both classes jointly in one pass -- a survivor can lean either way,
        # tagged from its own confidence_attack vs confidence_benign rather
        # than assuming everything is attack-leaning (a past version of this
        # caller passed a single hardcoded source_label="attack" for the
        # whole combined set, mislabeling every benign-leaning row).
        if not survivors_df.empty:
            survivors_df["source_label"] = np.where(
                survivors_df["confidence_attack"] > survivors_df["confidence_benign"],
                "attack",
                "benign",
            )

        mined_df = pd.concat(
            [survivors_df, leaf_rules_df], ignore_index=True, sort=False
        )
        save_dataframe_artifact(mined_df, run_dir, "mined_attribute_features")

        runtime_sec = time.perf_counter() - t0

        meta = MiningMetadata(
            run_name=run_name,
            timestamp=datetime.now(timezone.utc),
            scenario_name=scenario_name,
            n_candidates=len(mined_df),
            run_id=run_dir.name,
            artifact_path=str(run_dir),
            n_alert_groups=len(alert_groups),
        )

        write_manifest(
            run_dir,
            config={
                "run_name": run_name,
                "scenario_name": scenario_name,
                "contrast": config.contrast.model_dump(),
                "tree": config.tree.model_dump(),
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_alert_groups": len(alert_groups),
                "n_contrast_candidates": len(contrast_stats_df),
                "n_contrast_survivors": len(survivors_df),
                "n_leaf_rules": len(leaf_rules_df),
                "n_predicates": len(predicates),
                "runtime_sec": runtime_sec,
            }
        )

        log_artifact(str(run_dir))

        print(f"Finished attribute mining job. Saved artifacts to {run_dir}")

        return AttributeMiningJobResult(
            run_dir=run_dir,
            mined_df=mined_df,
            scenario_name=scenario_name,
            predicates=predicates,
        )
