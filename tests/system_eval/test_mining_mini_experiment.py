"""Phase 4 mini experiment on synthetic windows: every method runs end to end,
temporal methods are n/a without raw alerts, planted signal passes, and a
signature-restating feature is flagged."""

import random

import numpy as np
import pandas as pd
import pytest

from thesis.mining.temporal.lookback import attach_lookback_steps
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.mining_mini_experiment import (
    METHODS,
    MiniConfig,
    MiniWindow,
    run_mini,
    single_signature_share,
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
        qualifiers=set(),
        attr_similarities={},
        cve_refs={"CVE-2020-1"} if rng.random() < 0.3 else set(),
        signature_matches_per_day=rate,
        int_ip_is_multiple=rng.random() < 0.2,
        ext_port_is_multiple=False,
        int_port_is_multiple=False,
    )


def _ait_groups(n=160, seed=1):
    rng = random.Random(seed)
    t0 = 1_642_204_800
    alerts, groups = [], []
    for i in range(n):
        host, start, attack = f"h{i}", t0 + i * 10_000, i % 4 == 0
        chain = ["SCAN", "LOGIN", "EXPLOIT"] if attack else ["NOISE", "LOGIN", "NOISE"]
        for j, short in enumerate(chain):
            alerts.append((start - 300 + 120 * j, host, short))
        if rng.random() < 0.5:
            alerts.append((start - 300, host, "NOISE"))
        groups.append(
            AlertGroup(
                alert_group_id=f"g{i}",
                group_id=f"cscas_grouping:{host}:{chain[-1]} alert:{i:040x}",
                method="cscas_grouping",
                start_ts=start,
                end_ts=start,
                n_alerts=1,
                group_label="attack" if attack else "benign",
                raw_items={
                    f"host:{host}",
                    f"short:{chain[-1]}",
                    f"sig:{chain[-1].lower()}",
                },
            )
        )
    attach_lookback_steps(
        groups, pd.DataFrame(alerts, columns=["time", "host", "short"])
    )
    return groups


def _methods():
    from thesis.mining.temporal.mctsextent import vendor

    try:
        vendor.load_launch_mcts(fetch=False)
        return METHODS
    except FileNotFoundError:
        return tuple(m for m in METHODS if m != "mctsextent")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    rng = random.Random(0)
    cscas = [_cscas_group(i, rng) for i in range(900)]
    ait = _ait_groups()
    windows = [
        MiniWindow(
            "cscas",
            "synthetic cscas",
            cscas[:600],
            cscas[600:],
            signature_of=lambda g: (g.category, g.ruleset, g.proto),
        ),
        MiniWindow(
            "ait_ads",
            "synthetic ait",
            ait[:110],
            ait[110:],
            signature_of=lambda g: g.group_id.split(":")[2],
            has_raw_alerts=True,
        ),
    ]
    config = MiniConfig(
        k=10,
        n_bins=4,
        mctsextent_iterations=150,
        mctsextent_top_k=5,
        prefixspan_min_support=0.2,
        methods=_methods(),
    )
    out = tmp_path_factory.mktemp("mini")
    summary, feats = run_mini(windows, config, out)
    assert (out / "summary.csv").exists() and (out / "features.csv").exists()
    return summary.set_index(["dataset", "method"]), feats


def test_temporal_methods_are_na_on_cscas(results):
    summary, _ = results
    for m in ("prefixspan", "mctsextent"):
        if ("cscas", m) in summary.index:
            assert summary.loc[("cscas", m), "status"].startswith("n/a")


def test_atemporal_methods_pass_on_cscas_like_data(results):
    summary, _ = results
    for m in ("eclat", "contrast_tree", "ripper"):
        row = summary.loc[("cscas", m)]
        assert row["status"] == "PASS", (m, row["status"])
        assert row["dead_on_train"] == 0
        assert row["n_budget"] <= 10


def test_temporal_methods_run_on_ait_like_data(results):
    summary, _ = results
    for m in ("prefixspan", "mctsextent"):
        if ("ait_ads", m) in summary.index:
            row = summary.loc[("ait_ads", m)]
            assert not row["status"].startswith("ERROR"), row["status"]
            assert row["n_raw"] > 0
            assert row["dead_on_train"] == 0


def test_feature_table_has_one_row_per_budgeted_feature(results):
    summary, feats = results
    counts = feats.groupby(["dataset", "method"]).size()
    for key, n in counts.items():
        assert n == summary.loc[key, "n_budget"]


def test_single_signature_flag():
    groups = [
        AlertGroup(
            alert_group_id=str(i),
            group_id=str(i),
            method="m",
            start_ts=i,
            end_ts=i,
            n_alerts=1,
        )
        for i in range(4)
    ]
    X = pd.DataFrame({"one_sig": [1, 1, 0, 0], "two_sigs": [1, 0, 1, 0], "never": 0})
    flags = single_signature_share(X, groups, lambda g: "A" if g.start_ts < 2 else "B")
    assert flags.to_dict() == {"one_sig": True, "two_sigs": False, "never": False}


def test_benign_gate_factor_defaults_to_g_and_can_be_relaxed():
    from thesis.system_eval.rule_precision_gate import passes_gate

    labels = pd.Series(["benign", "benign", "attack", "attack"])
    rates = pd.Series([0.015, 0.008, 0.05, 0.03])
    # pi = 2%: strict benign gate needs <= 1%, attack gate needs >= 4%.
    assert passes_gate(labels, rates, 0.02, 2.0).tolist() == [False, True, True, False]
    # benign_g = 1: benign rules only need <= pi; attack side unchanged.
    assert passes_gate(labels, rates, 0.02, 2.0, benign_g=1.0).tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_restatement_generalises_to_signature_sets():
    from thesis.system_eval.mining_mini_experiment import restatement_flags

    groups = [
        AlertGroup(
            alert_group_id=str(i),
            group_id=str(i),
            method="m",
            start_ts=i,
            end_ts=i,
            n_alerts=1,
        )
        for i in range(4)
    ]
    sigs = [{"A", "B"}, {"A", "C"}, {"B"}, {"C"}]
    X = pd.DataFrame(
        {
            "always_with_A": [1, 1, 0, 0],  # {A,B} & {A,C} = {A} -> restatement
            "no_common": [1, 0, 1, 1],  # {A,B} & {B} & {C} = {} -> not
            "never": [0, 0, 0, 0],
        }
    )
    flags = restatement_flags(X, groups, lambda g: sigs[g.start_ts])
    assert flags.to_dict() == {
        "always_with_A": True,
        "no_common": False,
        "never": False,
    }


def test_relative_eclat_floor_and_token_arms():
    from thesis.mining.common.item_table import build_item_table
    from thesis.system_eval.mining_mini_experiment import TOKEN_ARMS, eclat_count_floor

    rng = random.Random(0)
    table = build_item_table([_cscas_group(i, rng) for i in range(300)])
    assert eclat_count_floor(MiniConfig(), table) == 10
    rel = eclat_count_floor(MiniConfig(eclat_min_support_count="relative"), table)
    assert rel == max(3, -(-table.n_attack * 5 // 100))
    assert TOKEN_ARMS["T3"][1] == {"short", "sig"}
    assert TOKEN_ARMS["T0"] == (set(), set())


def test_unseen_attack_masks_any_vs_all():
    from thesis.system_eval.mining_mini_experiment import unseen_attack_masks

    def g(i, codes):
        return AlertGroup(
            alert_group_id=str(i),
            group_id=str(i),
            method="m",
            start_ts=i,
            end_ts=i,
            n_alerts=1,
            raw_items={f"short:{c}" for c in codes},
        )

    train = [g(0, {"A"}), g(1, {"B"})]
    test = [g(2, {"A", "X"}), g(3, {"Y"}), g(4, {"A"}), g(5, {"Z"})]
    window = MiniWindow(
        "ait_ads",
        "w",
        train,
        test,
        signature_of=lambda t: frozenset(i[6:] for i in t.raw_items),
    )
    y = np.array([True, True, True, False])
    any_, all_ = unseen_attack_masks(window, y)
    assert any_.tolist() == [True, True, False, False]  # benign Z never counts
    assert all_.tolist() == [False, True, False, False]


def test_equivalence_vs_confined_restatement():
    from thesis.system_eval.mining_mini_experiment import (
        equivalence_flags,
        restatement_flags,
    )

    # BG is a background code in every group; X occurs in groups 0-4.
    groups = [
        AlertGroup(
            alert_group_id=str(i),
            group_id=str(i),
            method="m",
            start_ts=i,
            end_ts=i,
            n_alerts=1,
        )
        for i in range(10)
    ]
    sigs = [{"BG", "X"} if i < 5 else {"BG"} for i in range(10)]
    X = pd.DataFrame(
        {
            "is_x": [1] * 5 + [0] * 5,  # = X exactly
            "slice_of_x": [1, 1, 0, 0, 0] + [0] * 5,  # inside X, but 2/5 of it
            "host_rule": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # only shares BG
        }
    )
    sig_of = lambda g: sigs[g.start_ts]  # noqa: E731
    confined = restatement_flags(X, groups, sig_of)
    eq = equivalence_flags(X, groups, sig_of, jaccard=0.9)
    # Confined flags all three (every group contains BG); equivalence only
    # flags the feature that IS a signature.
    assert confined.to_dict() == {"is_x": True, "slice_of_x": True, "host_rule": True}
    assert eq["equivalent"].to_dict() == {
        "is_x": True,
        "slice_of_x": False,
        "host_rule": False,
    }
    assert eq.loc["is_x", "equiv_signature"] == "X"
    assert eq.loc["slice_of_x", "equiv_jaccard"] == 0.4


def test_cell_artifacts_are_written_and_consistent(tmp_path):
    import json

    from thesis.encoders.symbolic import SymbolicFeatureEncoder
    from thesis.features.persistence import load_symbolic_feature_schema

    rng = random.Random(0)
    cscas = [_cscas_group(i, rng) for i in range(900)]
    window = MiniWindow(
        "cscas",
        "synthetic cscas",
        cscas[:600],
        cscas[600:],
        signature_of=lambda g: (g.category, g.ruleset, g.proto),
    )
    config = MiniConfig(k=10, n_bins=4, methods=("eclat", "contrast_tree", "ripper"))
    summary, feats = run_mini([window], config, tmp_path)
    for method in config.methods:
        cell = tmp_path / "cells" / "synthetic" / "cscas_pregrouped" / "T0" / method
        patterns = pd.read_csv(cell / "patterns.csv")
        meta = json.loads((cell / "mining.json").read_text())
        row = summary.set_index("method").loc[method]
        # Stage counts agree with the summary row.
        assert len(patterns) == row["n_raw"]
        assert (patterns["stage"] == "kept").sum() == row["n_budget"]
        assert (patterns["stage"] != "dropped_gate").sum() >= row["n_gated_conditions"]
        assert set(patterns["stage"]) <= {
            "dropped_gate",
            "dropped_duplicate_condition",
            "dropped_duplicate_firing",
            "dropped_budget",
            "kept",
        }
        # Kept patterns carry their feature name; settings are recorded.
        kept = patterns[patterns["stage"] == "kept"]
        assert (kept["feature"].fillna("") != "").all()
        assert meta["method"] == method and meta["method_settings"]
        assert meta["versions"]["repo_commit"]
        # The saved schema reloads and encodes exactly like the run did.
        schema = load_symbolic_feature_schema(cell / "schema.json")
        X = SymbolicFeatureEncoder(schema).transform(window.train)
        mine = feats[feats["method"] == method]
        assert list(X.columns) == list(mine["feature"])
        assert X.sum().tolist() == mine["train_fires"].tolist()
