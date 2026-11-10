import numpy as np
import pandas as pd

from thesis.schemas.features import (
    BaseFeatureSchema,
    FeatureSchema,
    SymbolicFeature,
    SymbolicFeatureSchema,
)
from thesis.system_eval.rule_precision_gate import (
    gated_feature_names,
    passes_gate,
    precision_gate,
    rule_table,
)


def _feat(name: str, label: str) -> SymbolicFeature:
    return SymbolicFeature(
        feature_name=name,
        itemset=(name,),
        source_label=label,
        mining_type="decision_tree_rule",
    )


def _setup():
    # 2000 rows, 40 attacks -> pi = 2%.
    n, n_atk = 2000, 40
    y = np.zeros(n, dtype=int)
    y[:n_atk] = 1
    X = pd.DataFrame({"base": np.arange(n)})
    # strong attack rule: fires on 40 attacks + 40 benign -> 50% (25x pi)
    X["atk_strong"] = 0
    X.loc[: n_atk - 1, "atk_strong"] = 1
    X.loc[n_atk : 2 * n_atk - 1, "atk_strong"] = 1
    # weak attack rule (the "3 in 100" case): 3 attacks in 100 rows -> 3% (1.5x pi)
    X["atk_weak"] = 0
    X.loc[:2, "atk_weak"] = 1
    X.loc[100:196, "atk_weak"] = 1
    # clean benign rule: 1000 benign rows, no attacks -> 0%
    X["ben_clean"] = 0
    X.loc[1000:1999, "ben_clean"] = 1
    # weak benign rule: 20 attacks in 1020 rows -> ~1.96% (~pi)
    X["ben_weak"] = 0
    X.loc[:19, "ben_weak"] = 1
    X.loc[500:1499, "ben_weak"] = 1
    feats = [
        _feat(c, "attack" if c.startswith("atk") else "benign")
        for c in X.columns
        if c != "base"
    ]
    schema = FeatureSchema(
        schema_name="t",
        schema_version="0",
        base=BaseFeatureSchema(["base"]),
        symbolic=SymbolicFeatureSchema(
            schema_name="s", schema_version="0", features=feats
        ),
    )
    return schema, X, y


def test_gate_drops_weak_rules_at_g2():
    schema, X, y = _setup()
    assert set(gated_feature_names(schema, X, y, g=2.0)) == {"atk_strong", "ben_clean"}


def test_weak_attack_rule_passes_at_g1_5_only():
    schema, X, y = _setup()
    assert "atk_weak" in gated_feature_names(schema, X, y, g=1.5)
    assert "atk_weak" not in gated_feature_names(schema, X, y, g=2.0)


def test_precision_gate_keeps_base_and_returns_schema():
    schema, X, y = _setup()
    gated = precision_gate(schema, X, y, g=2.0)
    assert gated.base.features == ["base"]
    assert [f.feature_name for f in gated.symbolic.features] == [
        "atk_strong",
        "ben_clean",
    ]


def test_never_firing_rule_fails_and_mixed_is_kept():
    labels = pd.Series(["attack", "benign", "mixed"])
    rates = pd.Series([np.nan, np.nan, np.nan])
    assert passes_gate(labels, rates, 0.02, 2.0).tolist() == [False, False, True]


def test_rule_table_next_window_and_same_side():
    schema, X, y = _setup()
    # Next window: atk_strong stops working (fires only on benign).
    X_next = X.copy()
    X_next["atk_strong"] = 0
    X_next.loc[1000:1079, "atk_strong"] = 1
    t = rule_table(schema, X, y, X_next, y).set_index("feature")
    assert t.loc["atk_strong", "attack_rate"] == 0.5
    assert t.loc["atk_strong", "attack_rate_next"] == 0.0
    assert t.loc["atk_strong", "same_side_next"] == 0.0
    assert t.loc["ben_clean", "same_side_next"] == 1.0
    assert bool(t.loc["atk_strong", "passes_g2"]) and not bool(
        t.loc["atk_weak", "passes_g2"]
    )
