"""Comparison miners (Eclat, contrast tree, RIPPER, PrefixSpan) on synthetic
data: each must return the shared pattern frame, and every pattern, written
into the schema and encoded on the same groups, must fire on exactly the
groups the miner counted (n_attack / n_benign)."""

import random

import numpy as np
import pandas as pd
import pytest

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.atemporal.contrast_tree.miner import mine_contrast_tree
from thesis.mining.atemporal.eclat.miner import mine_eclat
from thesis.mining.atemporal.ripper.miner import mine_ripper, ripper_input
from thesis.mining.common.item_table import build_item_table
from thesis.mining.common.patterns import (
    PATTERN_COLUMNS,
    add_pattern_stats,
    patterns_to_schema,
)
from thesis.mining.temporal.lookback import attach_lookback_steps
from thesis.mining.temporal.prefixspan.miner import mine_prefixspan
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import (
    AttributeMiningConfig,
    DecisionTreeRuleConfig,
    SequenceFilterConfig,
)


def _cscas_group(i, rng):
    category = rng.choice(["EXPLOIT", "SCAN", "POLICY", "DNS"])
    rate = float(rng.randint(1, 5000))
    attack = (category == "EXPLOIT" and rate < 2000) or rng.random() < 0.02
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
        signature_matches_per_day=rate,
        attr_similarities={},
        int_ip_is_multiple=rng.random() < 0.2,
        ext_port_is_multiple=False,
        int_port_is_multiple=False,
    )


@pytest.fixture(scope="module")
def cscas_like():
    rng = random.Random(0)
    groups = [_cscas_group(i, rng) for i in range(600)]
    return groups, build_item_table(groups, n_bins=4)


def _assert_round_trip(frame, groups, table):
    assert list(frame.columns[: len(PATTERN_COLUMNS)]) == PATTERN_COLUMNS
    assert len(frame) > 0
    assert set(frame["source_label"]) <= {"attack", "benign"}
    stats = add_pattern_stats(frame, table.n_attack, table.n_benign)
    schema = patterns_to_schema(stats, "s", "1", item_table=table)
    X = SymbolicFeatureEncoder(schema).transform(groups)
    y = np.array([g.group_label == "attack" for g in groups])
    # patterns_to_schema keeps the first of identical conditions; compare in order.
    for f in schema.features:
        col = X[f.feature_name].to_numpy().astype(bool)
        assert col.any(), f"{f.feature_name} never fires"
    counted = {
        tuple(sorted(p.token for p in preds))
        if isinstance(preds, tuple) and preds
        else None: (a, b)
        for preds, a, b in zip(
            stats.get("predicates", pd.Series([None] * len(stats))),
            stats["n_attack"],
            stats["n_benign"],
        )
    }
    for f in schema.features:
        col = X[f.feature_name].to_numpy().astype(bool)
        got = (int((col & y).sum()), int((col & ~y).sum()))
        key = tuple(sorted(f.itemset))
        assert key in counted, f.feature_name
        assert got == counted[key], f.feature_name


def test_eclat_miner_round_trip(cscas_like):
    groups, table = cscas_like
    frame = mine_eclat(table, min_support=0.1, max_len=2)
    stats = add_pattern_stats(frame, table.n_attack, table.n_benign)
    schema = patterns_to_schema(stats, "s", "1", item_table=table)
    X = SymbolicFeatureEncoder(schema).transform(groups)
    y = np.array([g.group_label == "attack" for g in groups])
    for f, row in zip(schema.features, stats.itertuples()):
        col = X[f.feature_name].to_numpy().astype(bool)
        assert (int((col & y).sum()), int((col & ~y).sum())) == (
            row.n_attack,
            row.n_benign,
        )
    # Label used after the search: every kept pattern leans its own way.
    lean = stats["support_attack"] > stats["support_benign"]
    assert (lean == (stats["source_label"] == "attack")).all()


def test_contrast_tree_miner_round_trip(cscas_like):
    groups, table = cscas_like
    config = AttributeMiningConfig(
        tree=DecisionTreeRuleConfig(
            max_depth=1, max_depth_attack=3, min_samples_leaf=10
        )
    )
    frame = mine_contrast_tree(table, config)
    assert {"contrast_categorical", "decision_tree_rule"} <= set(frame["mining_type"])
    _assert_round_trip(frame, groups, table)


def test_ripper_miner_round_trip(cscas_like):
    groups, table = cscas_like
    frame = mine_ripper(table, random_state=0)
    assert set(frame["source_label"]) == {"attack", "benign"}
    _assert_round_trip(frame, groups, table)
    # The planted attack rule (EXPLOIT and a low rate) is found.
    attack_items = {
        i for items in frame.loc[frame.source_label == "attack", "items"] for i in items
    }
    assert "category=EXPLOIT" in attack_items


def test_ripper_input_one_bin_column_per_numeric_field(cscas_like):
    _, table = cscas_like
    frame = ripper_input(table)
    assert "signature_matches_per_day" in frame.columns
    assert set(frame["signature_matches_per_day"]) <= set(table.X_bin.columns)
    assert not any(c in frame.columns for c in table.X_bin.columns)


def test_ripper_is_deterministic_given_seed(cscas_like):
    _, table = cscas_like
    a = mine_ripper(table, random_state=3)
    b = mine_ripper(table, random_state=3)
    assert a["items"].tolist() == b["items"].tolist()


def test_prefixspan_miner_round_trip():
    rng = random.Random(1)
    t0 = 1_642_204_800
    alerts, groups = [], []
    for i in range(80):
        host = f"h{i}"
        start = t0 + i * 10_000
        attack = i % 4 == 0
        chain = ["SCAN", "LOGIN", "EXPLOIT"] if attack else ["NOISE", "LOGIN", "NOISE"]
        for j, short in enumerate(chain):
            alerts.append((start - 300 + 120 * j, host, short))
        if rng.random() < 0.5:
            alerts.append((start - 200, host, "NOISE"))
        groups.append(
            AlertGroup(
                alert_group_id=f"g{i}",
                group_id=f"g{i}",
                method="cscas_grouping",
                start_ts=start,
                end_ts=start,
                n_alerts=1,
                group_label="attack" if attack else "benign",
                raw_items={f"host:{host}"},
            )
        )
    attach_lookback_steps(
        groups, pd.DataFrame(alerts, columns=["time", "host", "short"])
    )
    frame = mine_prefixspan(
        groups,
        min_support=0.3,
        max_len=3,
        filters=SequenceFilterConfig(min_k=2, min_support_count=3),
    )
    assert "short:SCAN" in {i for items in frame["items"] for i in items}
    stats = add_pattern_stats(frame, 20, 60)
    schema = patterns_to_schema(stats, "s", "1")
    X = SymbolicFeatureEncoder(schema).transform(groups)
    y = np.array([g.group_label == "attack" for g in groups])
    for f, row in zip(schema.features, stats.itertuples()):
        col = X[f.feature_name].to_numpy().astype(bool)
        assert (int((col & y).sum()), int((col & ~y).sum())) == (
            row.n_attack,
            row.n_benign,
        ), f.feature_name


def test_prefixspan_miner_requires_lookback():
    g = AlertGroup(
        alert_group_id="g",
        group_id="g",
        method="m",
        start_ts=0,
        end_ts=0,
        n_alerts=1,
        group_label="attack",
    )
    with pytest.raises(ValueError, match="context_steps"):
        mine_prefixspan([g])


def _planted_lookback_groups(n=80, seed=1):
    rng = random.Random(seed)
    t0 = 1_642_204_800
    alerts, groups = [], []
    for i in range(n):
        host = f"h{i}"
        start = t0 + i * 10_000
        attack = i % 4 == 0
        chain = ["SCAN", "LOGIN", "EXPLOIT"] if attack else ["NOISE", "LOGIN", "NOISE"]
        for j, short in enumerate(chain):
            alerts.append((start - 300 + 120 * j, host, short))
        if rng.random() < 0.5:
            alerts.append((start - 300, host, "NOISE"))  # same bucket as step 0
        groups.append(
            AlertGroup(
                alert_group_id=f"g{i}",
                group_id=f"g{i}",
                method="cscas_grouping",
                start_ts=start,
                end_ts=start,
                n_alerts=1,
                group_label="attack" if attack else "benign",
                raw_items={f"host:{host}"},
            )
        )
    attach_lookback_steps(
        groups, pd.DataFrame(alerts, columns=["time", "host", "short"])
    )
    return groups


@pytest.fixture(scope="module")
def mctsextent():
    from thesis.mining.temporal.mctsextent import miner, vendor

    try:
        vendor.load_launch_mcts(fetch=False)
    except FileNotFoundError:
        pytest.skip("MCTSExtent checkout missing (vendor.ensure_mctsextent())")
    return miner.mine_mctsextent


def test_mctsextent_miner_round_trip_and_planted_pattern(mctsextent):
    groups = _planted_lookback_groups()
    frame = mctsextent(groups, top_k=5, iterations=300, random_state=0)
    assert set(frame["mining_type"]) == {"itemset_sequence"}
    assert set(frame["source_label"]) == {"attack", "benign"}
    attack_items = {
        i for items in frame.loc[frame.source_label == "attack", "items"] for i in items
    }
    assert any("short:SCAN" in step or "short:EXPLOIT" in step for step in attack_items)
    stats = add_pattern_stats(frame, 20, 60)
    schema = patterns_to_schema(stats, "s", "1")
    X = SymbolicFeatureEncoder(schema).transform(groups)
    y = np.array([g.group_label == "attack" for g in groups])
    for f, row in zip(schema.features, stats.drop_duplicates("items").itertuples()):
        col = X[f.feature_name].to_numpy().astype(bool)
        assert (int((col & y).sum()), int((col & ~y).sum())) == (
            row.n_attack,
            row.n_benign,
        ), f.feature_name


def test_mctsextent_is_reproducible_given_seed(mctsextent):
    groups = _planted_lookback_groups()
    a = mctsextent(groups, top_k=5, iterations=300, random_state=7)
    b = mctsextent(groups, top_k=5, iterations=300, random_state=7)
    assert a["items"].tolist() == b["items"].tolist()
