"""Phase 4 mini experiment on synthetic windows: every method runs end to end,
temporal methods are n/a without raw alerts, planted signal passes, and a
signature-restating feature is flagged."""

import random

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
