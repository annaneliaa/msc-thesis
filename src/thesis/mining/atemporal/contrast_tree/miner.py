"""Contrast sets + decision tree (the thesis method) as a comparison miner.

Runs the same two steps as attribute_mining_job.run_alert_group_attribute_mining_job
(Step 1 contrast-set survivors, Step 2 tree leaves over survivors + numerics),
but on a prebuilt ItemTable and without artifacts, and returns the shared
pattern frame (common/patterns.py). Both survivors and leaves count against
the feature budget K, because both become schema features in the thesis
pipeline.
"""

from __future__ import annotations

import pandas as pd

from thesis.mining.atemporal.contrast_tree.attribute_contrast_mining import (
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.atemporal.contrast_tree.decision_tree_rule_mining import (
    build_training_matrix,
    fit_and_extract_rules,
)
from thesis.mining.common.item_table import ItemTable
from thesis.mining.common.patterns import PATTERN_COLUMNS
from thesis.schemas.mining import AttributeMiningConfig

METHOD = "contrast_tree"


def mine_contrast_tree(
    table: ItemTable, config: AttributeMiningConfig | None = None
) -> pd.DataFrame:
    config = config or AttributeMiningConfig()
    stats = compute_predicate_contrast_stats(
        table.X_cat, table.y, table.column_predicate_map
    )
    survivors = filter_contrast_survivors(
        stats,
        min_attack_coverage=config.contrast.min_attack_coverage,
        min_benign_coverage=config.contrast.min_benign_coverage,
        min_growth_rate=config.contrast.min_growth_rate,
        min_growth_rate_attack=config.contrast.min_growth_rate_attack,
        max_p_value=config.contrast.max_p_value,
    )
    rows = []
    for r in survivors.itertuples(index=False):
        items = tuple(r.itemset)
        rows.append(
            {
                "items": items,
                "predicates": table.predicates_for(items),
                "mining_type": "contrast_categorical",
                # Same leaning rule attribute_mining_job uses for survivors.
                "source_label": (
                    "attack" if r.confidence_attack > r.confidence_benign else "benign"
                ),
                "n_attack": int(r.n_attack),
                "n_benign": int(r.n_benign),
            }
        )

    X_train, kept_map = build_training_matrix(
        table.X_cat,
        table.X_num,
        table.column_predicate_map,
        surviving_single_columns(survivors),
    )
    leaves, alphabet = fit_and_extract_rules(X_train, table.y, kept_map, config.tree)
    by_token = {p.token: p for p in alphabet}
    for r in leaves.itertuples(index=False):
        tokens = tuple(r.itemset)
        if not tokens:
            continue  # a root-only "leaf" (no split) is not a rule
        rows.append(
            {
                "items": tokens,
                "predicates": tuple(by_token[t] for t in tokens),
                "mining_type": "decision_tree_rule",
                "source_label": r.source_label,
                "n_attack": int(r.n_attack),
                "n_benign": int(r.n_benign),
            }
        )
    return pd.DataFrame(rows, columns=[*PATTERN_COLUMNS, "predicates"])
