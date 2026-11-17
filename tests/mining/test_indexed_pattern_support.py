"""The indexed, single-process pattern counting (which replaced a process pool
that copied the sequences into every worker) and the column-wise encoder must
give exactly what the brute-force scans gave."""

import random

import pandas as pd
import pytest

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.common.util import (
    add_cross_label_itemset_sequence_supports,
    add_cross_label_sequence_supports,
    indexed_pattern_counts,
    itemset_sequence_contains_pattern,
    sequence_contains_pattern,
)
from thesis.schemas.features import SymbolicFeature, SymbolicFeatureSchema
from thesis.schemas.groups import AlertGroup

TOKENS = [f"short:{c}" for c in "ABCDEFGH"]


def _sequences(n, seed):
    rng = random.Random(seed)
    return [
        [set(rng.sample(TOKENS, rng.randint(1, 3))) for _ in range(rng.randint(0, 6))]
        for _ in range(n)
    ]


def _item_patterns(seed):
    rng = random.Random(seed)
    pats = [
        tuple(rng.choice(TOKENS) for _ in range(rng.randint(1, 3))) for _ in range(60)
    ]
    return pats + [(), ("short:Z",)]  # empty pattern, unknown token


def _itemset_patterns(seed):
    rng = random.Random(seed)
    pats = [
        tuple(
            frozenset(rng.sample(TOKENS, rng.randint(1, 2)))
            for _ in range(rng.randint(1, 3))
        )
        for _ in range(60)
    ]
    return pats + [(), (frozenset(),), (frozenset({"short:Z"}),)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_indexed_counts_equal_brute_force(seed):
    seqs = _sequences(300, seed)
    pats = _item_patterns(seed)
    brute = [sum(sequence_contains_pattern(s, p) for s in seqs) for p in pats]
    assert indexed_pattern_counts(seqs, pats, sequence_contains_pattern, set) == brute

    ipats = _itemset_patterns(seed)
    brute = [sum(itemset_sequence_contains_pattern(s, p) for s in seqs) for p in ipats]
    got = indexed_pattern_counts(
        seqs, ipats, itemset_sequence_contains_pattern, lambda p: set().union(*p)
    )
    assert got == brute


def test_cross_label_supports_match_brute_force():
    target, other = _sequences(100, 5), _sequences(250, 6)
    pats = _item_patterns(5)[:20]
    mined = pd.DataFrame(
        {
            "sequence": pats,
            "support_count": [
                sum(sequence_contains_pattern(s, p) for s in target) for p in pats
            ],
        }
    )
    out = add_cross_label_sequence_supports(mined, target, other, "attack", "benign")
    expect = [sum(sequence_contains_pattern(s, p) for s in other) for p in pats]
    assert out["count_benign"].tolist() == expect
    assert out["support_benign"].tolist() == pytest.approx([c / 250 for c in expect])

    ipats = _itemset_patterns(5)[:20]
    mined = pd.DataFrame({"sequence": ipats, "support_count": [1] * len(ipats)})
    out = add_cross_label_itemset_sequence_supports(
        mined, target, other, "attack", "benign"
    )
    expect = [
        sum(itemset_sequence_contains_pattern(s, p) for s in other) for p in ipats
    ]
    assert out["count_benign"].tolist() == expect


def test_encoder_transform_equals_row_by_row():
    seqs = _sequences(200, 9)
    rng = random.Random(9)
    groups = [
        AlertGroup(
            alert_group_id=f"g{i}",
            group_id=f"g{i}",
            method="cscas_grouping",
            start_ts=i,
            end_ts=i,
            n_alerts=1,
            raw_items=set(rng.sample(TOKENS, 2)),
            context_steps=s,
        )
        for i, s in enumerate(seqs)
    ]
    feats = [
        SymbolicFeature(
            feature_name=f"seq{k}",
            itemset=tuple(p),
            mining_type="item_sequence",
            source_label="attack",
        )
        for k, p in enumerate(_item_patterns(3)[:15])
        if p
    ] + [
        SymbolicFeature(
            feature_name=f"set{k}",
            itemset=tuple(rng.sample(TOKENS, 2)),
            mining_type="itemset",
            source_label="benign",
        )
        for k in range(10)
    ]
    enc = SymbolicFeatureEncoder(
        SymbolicFeatureSchema(schema_name="t", schema_version="1", features=feats)
    )
    fast = enc.transform(groups)
    slow = pd.DataFrame([enc._row(tx) for tx in groups])
    pd.testing.assert_frame_equal(fast, slow)
    assert list(fast.columns) == [f.feature_name for f in feats]
