"""Eclat + filters (label used *after* the search) as a comparison miner.

Vaarandi-style baseline: mine frequent itemsets per class on the shared item
table (one item per field value, numerics as quantile bins, never tokenised
signature words), then use the label afterwards: cross-class supports, the
direction filter (own-class support minus other-class support) and the
existing itemset quality filters (ItemsetFilterConfig). Returns the shared
pattern frame. No OR-disjunction step and no mail-host abstraction: neither
is part of the comparison.
"""

from __future__ import annotations

import pandas as pd

from thesis.mining.atemporal.eclat.eclat_mining import run_eclat
from thesis.mining.common.item_table import ItemTable
from thesis.mining.common.patterns import PATTERN_COLUMNS
from thesis.mining.common.util import (
    add_confidence_scores,
    add_cross_label_supports,
    filter_mined_itemsets,
)
from thesis.schemas.mining import ItemsetFilterConfig

METHOD = "eclat"


def mine_eclat(
    table: ItemTable,
    min_support: float = 0.05,
    max_len: int | None = 3,
    min_support_diff: float = 0.05,
    filters: ItemsetFilterConfig | None = None,
) -> pd.DataFrame:
    """min_support is relative to the class being mined (each class has its
    own pass, as in the co-occurrence pipeline), so attack patterns aren't
    drowned by the ~98% benign majority."""
    filters = filters or ItemsetFilterConfig()
    baskets = table.baskets()
    y = table.y.to_numpy()
    by_label = {
        "attack": [b for b, v in zip(baskets, y) if v == 1],
        "benign": [b for b, v in zip(baskets, y) if v == 0],
    }
    frames = []
    for label, other in (("attack", "benign"), ("benign", "attack")):
        target = by_label[label]
        if not target:
            continue
        mined = run_eclat(target, min_support=min_support, max_len=max_len)
        if mined.empty:
            continue
        mined = add_cross_label_supports(
            mined, target, by_label[other], target_label=label, other_label=other
        )
        mined = add_confidence_scores(mined)
        mined = mined[mined["support_diff"] >= min_support_diff]
        mined = filter_mined_itemsets(
            mined.reset_index(drop=True),
            min_k=filters.min_k,
            max_k=filters.max_k,
            min_support_count=filters.min_support_count,
            min_abs_support_diff=filters.min_abs_support_diff,
            min_confidence_attack=filters.min_confidence_attack,
            max_confidence_attack=filters.max_confidence_attack,
            min_confidence_benign=filters.min_confidence_benign,
            max_overlap=filters.max_overlap,
            remove_subsumed=filters.remove_subsumed,
        )
        if mined.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "items": mined["itemset"].map(tuple),
                    "mining_type": "itemset",
                    "source_label": label,
                    "n_attack": mined["count_attack"].astype(int),
                    "n_benign": mined["count_benign"].astype(int),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.concat(frames, ignore_index=True)
