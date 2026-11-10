"""PrefixSpan + filters (temporal, label used *after* the search) as a
comparison miner.

Mines frequent item sequences per class on the per-host look-back sequences
(AlertGroup.context_steps, temporal/lookback.py), then uses the label
afterwards: cross-class supports, the direction filter and the existing
sequence quality filters (SequenceFilterConfig). Returns the shared pattern
frame; the encoder checks these features in order on context_steps.
No top_k_per_pass frequency cut: the shared budget K replaces it.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from thesis.mining.common.patterns import PATTERN_COLUMNS
from thesis.mining.common.util import (
    add_confidence_scores,
    add_cross_label_sequence_supports,
    filter_mined_sequences,
)
from thesis.mining.temporal.prefixspan.prefixspan_mining import run_prefixspan
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import SequenceFilterConfig

METHOD = "prefixspan"


def labelled_sequences(
    alert_groups: Sequence[AlertGroup],
) -> dict[str, list[list[set[str]]]]:
    """context_steps per class; groups must have look-back steps attached."""
    out: dict[str, list[list[set[str]]]] = {"attack": [], "benign": []}
    for tx in alert_groups:
        if tx.group_label not in out:
            continue
        if tx.context_steps is None:
            raise ValueError(
                f"group {tx.alert_group_id} has no context_steps; call "
                "temporal.lookback.attach_lookback_steps first"
            )
        out[tx.group_label].append(tx.context_steps)
    return out


def mine_prefixspan(
    alert_groups: Sequence[AlertGroup],
    min_support: float = 0.05,
    max_len: int | None = 3,
    min_support_diff: float = 0.05,
    filters: SequenceFilterConfig | None = None,
) -> pd.DataFrame:
    filters = filters or SequenceFilterConfig()
    seqs = labelled_sequences(alert_groups)
    frames = []
    for label, other in (("attack", "benign"), ("benign", "attack")):
        target = seqs[label]
        if not target:
            continue
        mined = run_prefixspan(target, min_support=min_support, max_len=max_len)
        if mined.empty:
            continue
        mined = add_cross_label_sequence_supports(
            mined, target, seqs[other], target_label=label, other_label=other
        )
        mined = add_confidence_scores(mined)
        mined = mined[mined["support_diff"] >= min_support_diff]
        mined = filter_mined_sequences(
            mined.reset_index(drop=True),
            min_k=filters.min_k,
            min_support_count=filters.min_support_count,
            min_abs_support_diff=filters.min_abs_support_diff,
            min_confidence_attack=filters.min_confidence_attack,
            max_confidence_attack=filters.max_confidence_attack,
            min_confidence_benign=filters.min_confidence_benign,
            min_lift=filters.min_lift,
            max_overlap=filters.max_overlap,
            remove_subsumed=filters.remove_subsumed,
        )
        if mined.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "items": mined["sequence"].map(tuple),
                    "mining_type": "item_sequence",
                    "source_label": label,
                    "n_attack": mined["count_attack"].astype(int),
                    "n_benign": mined["count_benign"].astype(int),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.concat(frames, ignore_index=True)
