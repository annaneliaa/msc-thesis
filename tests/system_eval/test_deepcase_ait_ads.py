import numpy as np
import pandas as pd
import pytest

from thesis.schemas.groups import AlertGroup
from thesis.system_eval.deepcase_ait_ads import (
    AitAdsH2HConfig,
    ScenarioData,
    alert_labels,
    alert_metric_rows,
    attach_alerts,
    budget_mask,
    build_pool,
    cluster_bootstrap,
    group_level,
    group_metric_row,
    per_group_counts,
    split_groups,
    tune_halves,
)
from thesis.system_eval.mining_comparison import scenario_split
from thesis.system_eval.mining_mini_experiment import labelled


def _g(i, label, ts, alert_ids=None):
    return AlertGroup(
        alert_group_id=f"ag{i}",
        group_id=f"g{i}",
        method="cscas_grouping",
        start_ts=ts,
        end_ts=ts,
        n_alerts=len(alert_ids or [1]),
        group_label=label,
        alert_ids=alert_ids,
    )


def test_split_groups_matches_scenario_split_and_keeps_late_mixed():
    groups = [_g(i, "attack" if i % 4 == 0 else "benign", 10 * i) for i in range(10)]
    groups.insert(3, _g(100, "mixed", 25))  # before the cut: context only
    groups.append(_g(101, "mixed", 1000))  # after the cut: scored
    tr, te, cut = split_groups(groups)
    ref_tr, ref_te = scenario_split(labelled(sorted(groups, key=lambda g: g.start_ts)))
    assert [g.group_id for g in tr] == [g.group_id for g in ref_tr]
    assert cut == ref_te[0].start_ts
    ids = [g.group_id for g in te]
    assert set(g.group_id for g in ref_te) < set(ids)
    assert "g101" in ids and "g100" not in ids
    assert [g.start_ts for g in te] == sorted(g.start_ts for g in te)


def test_attach_alerts_matches_duplicate_ids_copy_by_copy():
    # "d" is an exact duplicate alert present twice, both in group 0 (test);
    # "x" belongs to no group (training-side context).
    alerts = pd.DataFrame(
        {
            "alert_id": ["a", "d", "d", "b", "x"],
            "ts": [1, 2, 2, 3, 4],
            "host": ["h"] * 5,
            "short": ["S1", "S2", "S2", "S1", "S3"],
            "signature": ["n1", "n2", "n2", "n1", "n3"],
            "label_str": ["false_positive", "dirb", "dirb", None, "false_positive"],
        }
    )
    train = [_g(0, "benign", 1, ["a"])]
    test = [_g(1, "attack", 2, ["d", "d"]), _g(2, "benign", 3, ["b"])]
    df = attach_alerts(alerts, train, test, "signature")
    assert df["test_group"].tolist() == [-1, 0, 0, 1, -1]
    assert df["train_group"].tolist() == [0, -1, -1, -1, -1]
    assert df["is_test"].tolist() == [False, True, True, True, False]
    assert df["label"].tolist() == [0, 1, 1, 0, 0]
    assert df["event"].tolist() == ["n1", "n2", "n2", "n1", "n3"]


def test_attach_alerts_rejects_unknown_members():
    alerts = pd.DataFrame(
        {
            "alert_id": ["a"],
            "ts": [1],
            "host": ["h"],
            "short": ["S"],
            "signature": ["n"],
            "label_str": ["false_positive"],
        }
    )
    with pytest.raises(ValueError, match="not in the alert table"):
        attach_alerts(alerts, [_g(0, "benign", 1, ["a", "a"])], [], "signature")


def test_alert_labels_missing_and_benign_are_zero():
    s = pd.Series(["false_positive", None, "dirb", "wpscan"])
    assert alert_labels(s).tolist() == [0, 0, 1, 1]


def _scenario(name, n, test_from, events, labels):
    alerts = pd.DataFrame(
        {
            "alert_id": [f"{name}{i}" for i in range(n)],
            "ts": np.arange(n, dtype=np.int64) * 10,
            "host": ["h1", "h2"] * (n // 2),
            "event": events,
            "label": np.asarray(labels, dtype=np.int8),
            "is_test": np.arange(n) >= test_from,
            "test_group": np.where(np.arange(n) >= test_from, 0, -1),
            "train_group": np.full(n, -1),
        }
    )
    return ScenarioData(name, [], [], alerts, test_from * 10)


def test_build_pool_prefixes_machines_and_shares_event_codes():
    a = _scenario("fox", 4, 2, ["x", "y", "x", "z"], [0, 1, 0, 1])
    b = _scenario("shaw", 4, 3, ["y", "y", "w", "x"], [0, 0, 0, 1])
    pool = build_pool({"fox": a, "shaw": b})
    assert pool.machine[0] == "fox:h1" and pool.machine[4] == "shaw:h1"
    names = pool.event_names[pool.event]
    assert names.tolist() == ["x", "y", "x", "z", "y", "y", "w", "x"]
    assert pool.is_test.tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    assert pool.train_idx.tolist() == [0, 1, 4, 5, 6]
    first, second = tune_halves(pool, pool.train_idx)
    # fox: 2 training alerts -> 1 | 1; shaw: 3 -> 1 | 2
    assert first.tolist() == [0, 4] and second.tolist() == [1, 5, 6]


def test_budget_mask_variants():
    groups = [_g(i, "attack" if i < 3 else "benign", i) for i in range(30)]
    assert budget_mask("full", groups, 5, set(), 0).all()
    assert budget_mask("stratified_b0", groups, 100, set(), 0).all()
    m = budget_mask("stratified_b0", groups, 10, set(), 0)
    assert m.sum() == 10
    assert m[:3].sum() == 1  # 10 * 3/30 attack, rounded
    m = budget_mask("deepcase_items", groups, 10, {0, 7}, 0)
    assert np.flatnonzero(m).tolist() == [0, 7]
    with pytest.raises(ValueError):
        budget_mask("nope", groups, 1, set(), 0)


def test_alert_metric_rows_e2e_counts_manual_queue_as_escalated():
    label = np.array([1, 1, 0, 0, 1, 0])
    codes = np.array([1.0, -2.0, -1.0, 0.0, 0.0, 1.0])
    seen = np.array([True, False, True, True, True, True])
    rows = {r["subset"]: r for r in alert_metric_rows(label, codes, seen, 1.0)}
    a = rows["all"]
    assert a["e2e_tp"] == 2 and a["e2e_fp"] == 2 and a["missed_attacks"] == 1
    assert a["analyst_items"] == 4 and a["n_manual"] == 2
    assert a["flag_all_precision"] == pytest.approx(0.5)
    assert a["flag_all_f1"] == pytest.approx(2 / 3)
    assert np.isnan(a["pr_auc"])  # no score
    assert rows["unseen"]["n"] == 1 and rows["unseen"]["e2e_recall"] == 1.0


def test_alert_metric_rows_pr_auc_from_group_scores():
    label = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.1, 0.8, 0.2])
    codes = (score >= 0.5).astype(int)
    rows = alert_metric_rows(label, codes, np.ones(4, bool), 1.0, score)
    assert rows[0]["pr_auc"] == pytest.approx(1.0)
    assert rows[0]["e2e_f1"] == pytest.approx(1.0)


def test_group_level_any_alert_rule():
    tg = np.array([0, 0, 1, 1, 2])
    label = np.array([0, 1, 0, 0, 1])
    esc = np.array([1, 0, 0, 0, 0])
    gl, ge = group_level(tg, label, esc, 3)
    assert gl.tolist() == [True, False, True]
    assert ge.tolist() == [True, False, False]
    r = group_metric_row(gl, ge)
    assert (r["tp"], r["fp"], r["missed_attacks"]) == (1, 0, 1)
    assert r["recall"] == 0.5 and r["precision"] == 1.0


def test_per_group_counts_and_bootstrap_point_delta():
    tg = np.array([0, 0, 1, 2, 2, 2])
    label = np.array([1, 1, 0, 1, 0, 0])
    esc_a = np.array([1, 1, 0, 1, 0, 0])  # perfect
    esc_b = np.array([1, 1, 1, 1, 1, 1])  # flag everything
    ca = per_group_counts(tg, label, esc_a, 3)
    cb = per_group_counts(tg, label, esc_b, 3)
    assert ca.tolist() == [[2, 0, 0], [0, 0, 0], [1, 0, 0]]
    assert cb.tolist() == [[2, 0, 0], [0, 1, 0], [1, 2, 0]]
    strata = np.array([True, False, True])
    res = cluster_bootstrap(ca[None], cb[None], strata, 200, np.random.default_rng(0))
    point, d = res["f1"]
    assert point == pytest.approx(1.0 - 2 * 3 / (2 * 3 + 3))
    assert d.shape == (200,)
    assert (d >= -1e-12).all()  # a never worse than flag-everything
    assert res["recall"][0] == pytest.approx(0.0)


def test_config_validates():
    with pytest.raises(ValueError):
        AitAdsH2HConfig(event_field="host")
    with pytest.raises(ValueError):
        AitAdsH2HConfig(scenarios=("fox",), eval_scenarios=("shaw",))
    with pytest.raises(ValueError):
        AitAdsH2HConfig(budget_variants=("everything",))
    assert AitAdsH2HConfig(scenarios=("fox", "shaw")).evaluated == ("fox", "shaw")
