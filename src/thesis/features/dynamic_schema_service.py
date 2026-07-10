from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from thesis.features.dynamic_schema_builder import build_dynamic_schema
from thesis.features.dynamic_schema_registry import DynamicSchemaRegistry
from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    extract_leaf_rules,
    fit_rule_tree,
)
from thesis.paths import FEATURE_DIR
from thesis.pipeline.pipeline import load_alert_groups_json
from thesis.schemas.mining import AttributeMiningConfig


def _load_sorted_labeled_alert_groups(alert_groups_path: str | Path) -> list:
    alert_groups = load_alert_groups_json(Path(alert_groups_path))
    labeled = [tx for tx in alert_groups if tx.group_label in ("benign", "attack")]
    labeled.sort(key=lambda tx: tx.start_ts or 0)
    return labeled


def mine_and_deploy_dynamic_schema(
    alert_groups_path: str | Path,
    scenario_name: str,
    win_start_idx: int,
    win_end_idx: int,
    config: AttributeMiningConfig | None = None,
    root_dir: Path = FEATURE_DIR,
) -> Path:
    """
    Mine a window of chronologically-sorted, labeled alert groups
    [win_start_idx:win_end_idx) and deploy the result as the next Vk for
    `scenario_name`. A deliberate, explicit "deploy this as the new
    production schema" action -- unlike attribute_mining_job.py (called at
    high frequency by sweeps/screening/walk-forward experiments purely for
    evaluation), this is not wired into any existing mining/experiment call
    graph, so routine evaluation mining never floods the deployment history.

    Runs the same two-stage mining attribute_mining_job.py orchestrates, but
    calls the building blocks directly to keep the pre-concatenation frames
    (contrast_stats_df, leaf_rules_df) and X_num/y that build_dynamic_schema
    needs -- attribute_mining_job.py's AttributeMiningJobResult only keeps
    the post-concatenation, post-tagging mined_df.
    """
    config = config or AttributeMiningConfig()

    alert_groups = _load_sorted_labeled_alert_groups(alert_groups_path)
    window = alert_groups[win_start_idx:win_end_idx]
    if not window:
        raise ValueError(
            f"Mining window [{win_start_idx}:{win_end_idx}) is empty "
            f"(scenario has {len(alert_groups)} labeled alert groups)."
        )
    mining_window_start = datetime.fromtimestamp(window[0].start_ts, tz=timezone.utc)
    window_end_ts = window[-1].end_ts or window[-1].start_ts
    mining_window_end = datetime.fromtimestamp(window_end_ts, tz=timezone.utc)

    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(window)

    contrast_stats_df = compute_predicate_contrast_stats(X_cat, y, column_predicate_map)
    survivors_df = filter_contrast_survivors(
        contrast_stats_df,
        min_attack_coverage=config.contrast.min_attack_coverage,
        min_benign_coverage=config.contrast.min_benign_coverage,
        min_growth_rate=config.contrast.min_growth_rate,
        max_p_value=config.contrast.max_p_value,
    )
    surviving_cols = surviving_single_columns(survivors_df)

    X_train, kept_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_cols
    )
    tree = fit_rule_tree(
        X_train,
        y,
        max_depth=config.tree.max_depth,
        min_samples_leaf=config.tree.min_samples_leaf,
        class_weight=config.tree.class_weight,
        random_state=config.tree.random_state,
        min_impurity_decrease=config.tree.min_impurity_decrease,
    )
    leaf_rules_df, predicate_alphabet = extract_leaf_rules(
        tree, X_train, y, kept_predicate_map
    )

    mined_at = datetime.now(timezone.utc)

    def _build(version: int):
        return build_dynamic_schema(
            contrast_stats_df=survivors_df,
            leaf_rules_df=leaf_rules_df,
            predicate_alphabet=predicate_alphabet,
            column_predicate_map=column_predicate_map,
            X_num=X_num,
            y=y,
            version=version,
            mining_window_start=mining_window_start,
            mining_window_end=mining_window_end,
            mined_at=mined_at,
        )

    registry = DynamicSchemaRegistry(root_dir=root_dir)
    return registry.deploy(scenario_name, _build)
