"""
Paper Check 2: do weak rules get in, and should every method get a precision gate?

Step 1 (code reading, recorded here so the paper text can cite it):

  * decision_tree_rule_mining.extract_leaf_rules counts each leaf's groups
    from tree.apply(X) -- RAW counts, not tree_.value's class-weighted sums.
  * A leaf's polarity (source_label) is "attack" iff
    confidence_attack > confidence_benign, and both are per-class RECALL
    (n_class_in_leaf / n_class_total). recall_attack > recall_benign is
    exactly "attack rate in the leaf > the window's base rate pi" (growth
    rate > 1): a class-balanced majority, not a plain one. A 3-in-100 leaf
    at pi = 1.5% is therefore "attack-leaning". The thesis text ("whichever
    class is more common decides") doesn't match.
  * Contrast-set items need growth rate >= 3 and diverse-round leaves
    (round >= 2) need round_min_growth_rate >= 3, so the only route for a
    weak rule is a round-1 tree leaf, which has no floor.
  * The stored confidence_* are recall everywhere (mined_attribute_features,
    SymbolicFeature, DynamicSchema support_*). The drift monitor's mined rule
    confidence used to be max(recall_attack, recall_benign) while signal 2
    observes precision -- fixed in features/dynamic_schema_builder.py
    (mined confidence = leaf precision in its predicted class).

The gate (shared output format, applied after mining, same for every method):
  attack-leaning rule: attack rate in the rule >= g * pi
  benign-leaning rule: attack rate in the rule <= pi / g
with pi and the rule's attack rate measured on the window it was mined from.
Main g = 2, sensitivity g in {1.5, 3}. Base columns are never gated.

The attack rate is read off the encoded mining window (the feature's own
column in X_fit), not from the miner's stored statistics, so the gate works
for any method's schema. `precision_gate` has the PostFilter signature of
mining_setting_selection.run_w0_tuning, so the new methods' W0 tuning can
apply the same gate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from thesis.schemas.features import FeatureSchema

GATE_GS: tuple[float, ...] = (1.5, 2.0, 3.0)
MAIN_G = 2.0


def rule_attack_rates(
    schema: FeatureSchema, X: pd.DataFrame, y: np.ndarray
) -> pd.DataFrame:
    """One row per symbolic feature: leaning class, mining type, how often it
    fires on (X, y) and the attack rate among the rows it fires on."""
    if schema.symbolic is None:
        return pd.DataFrame(
            columns=[
                "feature",
                "source_label",
                "mining_type",
                "n_fire",
                "n_attack_fire",
                "support",
                "attack_rate",
            ]
        )
    y = np.asarray(y)
    rows = []
    for f in schema.symbolic.features:
        fires = (
            X[f.feature_name].to_numpy() > 0
            if f.feature_name in X
            else np.zeros(len(y), bool)
        )
        n_fire = int(fires.sum())
        n_atk = int((fires & (y == 1)).sum())
        rows.append(
            {
                "feature": f.feature_name,
                "source_label": f.source_label,
                "mining_type": f.mining_type,
                "n_fire": n_fire,
                "n_attack_fire": n_atk,
                "support": n_fire / len(y) if len(y) else np.nan,
                "attack_rate": n_atk / n_fire if n_fire else np.nan,
            }
        )
    return pd.DataFrame(rows)


def passes_gate(
    source_label: pd.Series,
    attack_rate: pd.Series,
    base_rate: float,
    g: float,
    benign_g: float | None = None,
) -> pd.Series:
    """Gate per rule. A rule that never fires fails; "mixed" rules (no
    leaning class) are kept, since the gate is defined per leaning class.

    benign_g (default: g) sets the benign side separately: benign-leaning
    rules need attack rate <= pi / benign_g. The mining comparison uses
    benign_g = 1 on AIT-ADS, whose time-window labels mark background alerts
    during attack phases as attacks, so a routine benign pattern can't reach
    pi / 2 (see PLAN.md, "AIT-ADS label noise")."""
    benign_g = g if benign_g is None else benign_g
    rate = attack_rate.astype(float)
    attack_ok = (source_label == "attack") & (rate >= g * base_rate)
    benign_ok = (source_label == "benign") & (rate <= base_rate / benign_g)
    mixed = ~source_label.isin(["attack", "benign"])
    return attack_ok | benign_ok | mixed  # NaN rate (never fires) compares False


def gated_feature_names(
    schema: FeatureSchema, X: pd.DataFrame, y: np.ndarray, g: float = MAIN_G
) -> list[str]:
    """Names of the symbolic features that pass the gate on (X, y)."""
    rates = rule_attack_rates(schema, X, y)
    if rates.empty:
        return []
    keep = passes_gate(
        rates["source_label"], rates["attack_rate"], float(np.mean(y)), g
    )
    return rates.loc[keep, "feature"].tolist()


def precision_gate(
    schema: FeatureSchema, X: pd.DataFrame, y: np.ndarray, g: float = MAIN_G
) -> FeatureSchema:
    """`schema` with every symbolic feature that fails the gate on its mining
    window (X, y) removed. Base columns untouched."""
    if schema.symbolic is None:
        return schema
    keep = set(gated_feature_names(schema, X, y, g))
    symbolic = replace(
        schema.symbolic,
        features=[f for f in schema.symbolic.features if f.feature_name in keep],
    )
    return replace(schema, symbolic=symbolic)


def rule_table(
    schema: FeatureSchema,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_next: pd.DataFrame,
    y_next: np.ndarray,
    gs: tuple[float, ...] = GATE_GS,
) -> pd.DataFrame:
    """Rule log for one walk-forward step: each symbolic feature's attack rate
    on its mining window Wi and on the next window Wi+1, gate pass/fail per g
    (on Wi), and whether it stays on the same side of the base rate on Wi+1
    (each window against its own pi) -- the stability the monitor's
    calibration signal watches."""
    tr = rule_attack_rates(schema, X_fit, y_fit)
    if tr.empty:
        return tr
    nx = rule_attack_rates(schema, X_next, y_next)[
        ["feature", "n_fire", "n_attack_fire", "attack_rate"]
    ]
    out = tr.merge(nx, on="feature", suffixes=("", "_next"), how="left")
    pi, pi_next = (
        float(np.mean(y_fit)),
        float(np.mean(y_next)) if len(y_next) else np.nan,
    )
    out["base_rate"], out["base_rate_next"] = pi, pi_next
    out["lift"] = out["attack_rate"] / pi if pi else np.nan
    out["lift_next"] = out["attack_rate_next"] / pi_next if pi_next else np.nan
    for g in gs:
        out[f"passes_g{g:g}"] = passes_gate(
            out["source_label"], out["attack_rate"], pi, g
        )
    above, above_next = out["attack_rate"] > pi, out["attack_rate_next"] > pi_next
    out["same_side_next"] = np.where(
        out["attack_rate_next"].isna(), np.nan, (above == above_next).astype(float)
    )
    return out
