"""Shared item table, pattern frame, budget K and schema writer.

The round-trip tests are the important ones: a pattern mined on the item
table, written into the schema and encoded on the same groups must fire on
exactly the groups the miner counted. Otherwise mined support/precision and
what the model sees would disagree (the bug class behind the old dead
contrast columns and the order-blind sequence features)."""

import random

import numpy as np
import pandas as pd
import pytest

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.atemporal.eclat.eclat_mining import run_eclat
from thesis.mining.common.item_table import build_item_table, quantile_bin_edges
from thesis.mining.common.patterns import (
    add_pattern_stats,
    patterns_to_schema,
    select_budget,
)
from thesis.schemas.groups import AlertGroup

CATEGORIES = ["EXPLOIT", "SCAN", "POLICY"]


def _group(i, rng):
    category = rng.choice(CATEGORIES)
    attack = category == "EXPLOIT" and rng.random() < 0.6
    return AlertGroup(
        alert_group_id=f"g{i}",
        group_id=f"g{i}",
        method="cscas_pregrouped",
        start_ts=1_642_636_800 + i,
        end_ts=1_642_636_800 + i,
        n_alerts=rng.randint(1, 50),
        group_label="attack" if attack else "benign",
        category=category,
        ruleset=rng.choice(["ET", "GPL"]),
        proto=rng.choice([6, 17]),
        cve_refs={"CVE-2020-1234"} if rng.random() < 0.3 else set(),
        qualifiers=set(),
        signature_matches_per_day=float(rng.randint(1, 5000)),
        attr_similarities={},
        int_ip_is_multiple=rng.random() < 0.2,
        ext_port_is_multiple=False,
        int_port_is_multiple=False,
    )


@pytest.fixture
def groups():
    rng = random.Random(0)
    return [_group(i, rng) for i in range(300)]


def test_item_table_has_one_item_per_field_value_and_bins(groups):
    table = build_item_table(groups, n_bins=4)
    items = table.items()
    assert {"category=EXPLOIT", "category=SCAN", "ruleset=ET", "cve_present"} <= set(
        items.columns
    )
    # Single-valued fields: exactly one category item per row.
    cat_cols = [c for c in items.columns if c.startswith("category=")]
    assert (items[cat_cols].sum(axis=1) == 1).all()
    # Numeric bins partition the rows: exactly one bin item per row per field.
    bins = [c for c in table.X_bin.columns if "signature_matches_per_day" in c]
    assert 2 <= len(bins) <= 4
    assert (table.X_bin[bins].sum(axis=1) == 1).all()
    # No tokenised signature words, only field items.
    assert not any(c.startswith(("sig:", "cat:")) for c in items.columns)
    assert len(table.baskets()) == len(groups)


def test_quantile_edges_collapse_ties_and_skip_constants():
    assert quantile_bin_edges(np.ones(10), 5).size == 0
    edges = quantile_bin_edges(np.array([0] * 90 + list(range(1, 11))), 5)
    assert edges.size == 0 or (edges > 0).all() or edges[0] == 0
    assert np.all(np.diff(edges) > 0)


@pytest.mark.parametrize("seed", range(3))
def test_atemporal_round_trip_fires_exactly_where_counted(seed):
    rng = random.Random(seed)
    groups = [_group(i, rng) for i in range(250)]
    table = build_item_table(groups, n_bins=3)
    baskets = table.baskets()
    mined = run_eclat(baskets, min_support=0.1, max_len=2)
    y = table.y.to_numpy()
    rows = []
    for items in mined["itemset"]:
        fires = np.array([set(items) <= b for b in baskets])
        rows.append(
            {
                "items": items,
                "mining_type": "itemset",
                "source_label": "attack",
                "n_attack": int((fires & (y == 1)).sum()),
                "n_benign": int((fires & (y == 0)).sum()),
            }
        )
    frame = add_pattern_stats(pd.DataFrame(rows), table.n_attack, table.n_benign)
    schema = patterns_to_schema(frame, "s", "1", item_table=table)
    X = SymbolicFeatureEncoder(schema).transform(groups)
    by_name = {f.feature_name: f for f in schema.features}
    assert len(schema.features) == len(frame)  # no name collisions
    for f, row in zip(schema.features, frame.itertuples()):
        col = X[f.feature_name].to_numpy()
        assert int((col & (y == 1)).sum()) == row.n_attack, f.feature_name
        assert int((col & (y == 0)).sum()) == row.n_benign, f.feature_name
        assert by_name[f.feature_name].confidence_attack == pytest.approx(
            row.support_attack
        )


def test_pattern_stats_precision_and_score():
    frame = pd.DataFrame(
        {
            "items": [("a",), ("b",), ("c",)],
            "mining_type": "itemset",
            "source_label": ["attack", "benign", "attack"],
            "n_attack": [9, 1, 0],
            "n_benign": [1, 99, 0],
        }
    )
    out = add_pattern_stats(frame, n_attack_total=10, n_benign_total=990)
    assert out["precision_attack"].tolist() == pytest.approx([0.9, 0.01, 0.0])
    assert out["lean_precision"].tolist() == pytest.approx([0.9, 0.99, 0.0])
    assert out["support"].tolist() == pytest.approx([0.01, 0.1, 0.0])
    assert out["support_attack"].tolist() == pytest.approx([0.9, 0.1, 0.0])
    assert out["score"].tolist() == pytest.approx([0.009, 0.099, 0.0])


def _frame(n_attack_patterns, n_benign_patterns):
    rows = [
        {
            "items": (f"a{i}",),
            "mining_type": "itemset",
            "source_label": "attack",
            "n_attack": 10 - i,
            "n_benign": 1,
        }
        for i in range(n_attack_patterns)
    ] + [
        {
            "items": (f"b{i}",),
            "mining_type": "itemset",
            "source_label": "benign",
            "n_attack": 0,
            "n_benign": 200 - i,
        }
        for i in range(n_benign_patterns)
    ]
    return add_pattern_stats(pd.DataFrame(rows), 20, 1000)


def test_budget_pooled_keeps_top_k_by_score():
    out = select_budget(_frame(5, 5), k=4)
    assert len(out) == 4
    assert out["score"].is_monotonic_decreasing
    # Benign patterns have far higher support, so a pooled budget is all benign.
    assert set(out["source_label"]) == {"benign"}


def test_budget_per_class_splits_and_fills():
    out = select_budget(_frame(5, 5), k=4, per_class=True)
    assert out["source_label"].value_counts().to_dict() == {"attack": 2, "benign": 2}
    # Only one attack pattern: benign fills the rest so k is still met.
    out = select_budget(_frame(1, 5), k=4, per_class=True)
    assert out["source_label"].value_counts().to_dict() == {"attack": 1, "benign": 3}
    assert len(select_budget(_frame(1, 1), k=4, per_class=True)) == 2


def test_budget_requires_stats():
    with pytest.raises(ValueError):
        select_budget(pd.DataFrame({"items": [("a",)]}), k=1)


def test_constant_items_are_dropped(groups):
    # Every synthetic group has qualifiers=set() and ext_port_is_multiple=False:
    # those items never fire; drop them (and anything that always fires).
    table = build_item_table(groups)
    X = table.items()
    assert "multi_ext_port" not in X.columns
    assert (X.nunique() > 1).all()
