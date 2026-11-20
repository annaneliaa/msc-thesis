"""Group-property measures on hand-built groups."""

import pytest

from thesis.schemas.groups import AlertGroup
from thesis.system_eval.grouping_properties import group_properties, properties_table


def _g(i, label, codes, hosts=("h1",), alert_labels=None, n=1, dur=0, steps=1):
    return AlertGroup(
        alert_group_id=f"g{i}",
        group_id=f"g{i}",
        method="m",
        start_ts=100,
        end_ts=100 + dur,
        n_alerts=n,
        group_label=label,
        raw_items={f"short:{c}" for c in codes} | {f"host:{h}" for h in hosts},
        alert_labels=set(
            alert_labels or (["dirb"] if label == "attack" else ["false_positive"])
        ),
        sorted_items=[set() for _ in range(steps)],
    )


def test_properties_on_small_example():
    groups = [
        _g(0, "benign", ["A"]),
        _g(1, "benign", ["B", "C"], hosts=("h1", "h2"), n=3, dur=10, steps=3),
        _g(2, "attack", ["A"]),  # ambiguous: A also benign
        _g(
            3,
            "attack",
            ["A", "X"],
            alert_labels=["dirb", "false_positive"],
            n=2,
            dur=4,
            steps=2,
        ),
        _g(4, "mixed", ["Z"]),  # not labelled: ignored
    ]
    r = group_properties(groups, other_attack_codes={"A"})
    assert r["n_groups"] == 4 and r["n_attack"] == 2 and r["base_rate"] == 0.5
    assert r["mixed_label_share"] == 0.25
    assert r["ambiguous_attack_share"] == 0.5
    assert r["single_code_share"] == 0.5
    assert r["multi_host_share"] == 0.25
    assert r["ordered_share"] == 0.5
    assert r["codes_per_group_mean"] == pytest.approx(6 / 4)
    assert r["unseen_attack_share"] == 0.5  # X never an attack elsewhere


def test_unseen_compares_against_other_scenarios_same_grouping():
    a = [_g(0, "attack", ["A"]), _g(1, "benign", ["B"])]
    b = [_g(2, "attack", ["A"]), _g(3, "attack", ["Y"])]
    other_grouping = [_g(4, "attack", ["Y"])]
    t = properties_table({("s1", "m"): a, ("s2", "m"): b, ("s1", "m2"): other_grouping})
    t = t.set_index(["scenario", "grouping"])
    assert t.loc[("s1", "m"), "unseen_attack_share"] == 0.0
    assert t.loc[("s2", "m"), "unseen_attack_share"] == 0.5  # Y not an attack in s1/m
    assert (
        t.loc[("s1", "m2"), "unseen_attack_share"]
        != t.loc[("s1", "m2"), "unseen_attack_share"]
    )  # NaN: no other scenario
