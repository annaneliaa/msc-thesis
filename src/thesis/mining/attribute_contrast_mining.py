from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np
import pandas as pd

from thesis.mining.attribute_features import (
    BINARY_CATEGORICAL_FIELDS,
    MULTI_VALUED_CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    compute_candidate_attribute_features,
)
from thesis.schemas.groups import AlertGroup

_EPS = 1e-9

_CONTRAST_STATS_COLUMNS = [
    "itemset",
    "support",
    "support_count",
    "confidence_attack",
    "confidence_benign",
    "growth_rate",
    "precision_attack",
    "lift",
    "p_value",
    "mining_type",
    "n_attack",
    "n_benign",
]


def build_categorical_predicate_matrix(
    alert_groups: Sequence[AlertGroup],
    exclude_fields: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, tuple[str, Any]]]:
    """
    Single per-group pass building everything downstream mining needs:
    - X_cat: one binary (0/1) column per candidate single categorical
      predicate (Step 1's input, and Step 2's categorical columns)
    - X_num: the numeric base-feature columns (Step 2's numeric columns) --
      built here too so compute_candidate_attribute_features() is only
      ever called once per alert group, not re-derived later
    - y: the binary Label column (1 = attack, 0 = benign)
    - column_predicate_map: column -> (attribute, expected_value), describing
      how to reconstruct each X_cat column as a condition against
      compute_candidate_attribute_features() output

    exclude_fields, if given, drops the named candidate fields (matching
    MULTI_VALUED_CATEGORICAL_FIELDS / BINARY_CATEGORICAL_FIELDS /
    NUMERIC_FIELDS entries) from the candidate space entirely, for both
    Step 1 (contrast-set) and Step 2 (decision tree) -- e.g. to keep fields
    that a real deployment couldn't compute out of the mined schema. None
    (default) preserves the full candidate space, unchanged.
    """
    exclude_fields = exclude_fields or set()
    multi_valued_fields = [
        f for f in MULTI_VALUED_CATEGORICAL_FIELDS if f not in exclude_fields
    ]
    binary_fields = [f for f in BINARY_CATEGORICAL_FIELDS if f not in exclude_fields]
    numeric_fields = [f for f in NUMERIC_FIELDS if f not in exclude_fields]

    cat_rows: list[dict[str, int]] = []
    num_rows: list[dict[str, float]] = []
    labels: list[int] = []
    column_predicate_map: dict[str, tuple[str, Any]] = {}

    for tx in alert_groups:
        feats = compute_candidate_attribute_features(tx)

        cat_row: dict[str, int] = {}
        for field_name in multi_valued_fields:
            value = feats[field_name]
            col = f"{field_name}={value}"
            cat_row[col] = 1
            column_predicate_map.setdefault(col, (field_name, value))
        for field_name in binary_fields:
            cat_row[field_name] = int(bool(feats[field_name]))
            column_predicate_map.setdefault(field_name, (field_name, True))
        cat_rows.append(cat_row)

        num_rows.append(
            {field_name: feats[field_name] for field_name in numeric_fields}
        )

        labels.append(1 if tx.group_label == "attack" else 0)

    # int8, not the .astype(int) default of int64 -- every value here is a
    # 0/1 indicator, and every consumer (compute_predicate_contrast_stats's
    # .astype(bool), sklearn's tree builder in decision_tree_rule_mining)
    # already casts on use, so this is a straight ~8x memory cut with no
    # behavior change. Matters at CSCAS's ~1.4M-row scale (int64 was ~970MB).
    X_cat = pd.DataFrame(cat_rows).fillna(0).astype(np.int8)
    X_num = pd.DataFrame(num_rows)
    y = pd.Series(labels, name="Label")
    return X_cat, X_num, y, column_predicate_map


def _chi_square_p_value(
    n_fires_attack: int, n_attack: int, n_fires_benign: int, n_benign: int
) -> float | None:
    if n_attack == 0 or n_benign == 0:
        return None
    table = [
        [n_fires_attack, n_attack - n_fires_attack],
        [n_fires_benign, n_benign - n_fires_benign],
    ]
    try:
        from scipy.stats import chi2_contingency

        _, p_value, _, _ = chi2_contingency(table)
        return float(p_value)
    except Exception:
        return None


def predicate_support_stats(
    fires: np.ndarray,
    attack_mask: np.ndarray,
    benign_mask: np.ndarray,
    n_attack: int,
    n_benign: int,
) -> dict[str, Any]:
    """
    Attack/benign support, growth_rate, precision_attack, and chi-square
    p-value for one boolean "does this predicate fire" mask -- the same
    per-candidate arithmetic compute_predicate_contrast_stats uses for its
    categorical columns (lines below), factored out so
    features.dynamic_schema_builder can compute the identical statistics for
    numeric-threshold predicates (which this function never mines directly,
    since it only ever sees one-hot categorical columns).

    attack_support/benign_support (aliased confidence_attack/
    confidence_benign downstream) are per-class recall: P(fires | class).
    precision_attack is the classical association-rule confidence instead:
    P(class=attack | fires) -- how trustworthy a firing is, not how much of
    a class it covers. It's diagnostic only; filter_contrast_survivors and
    the dynamic schema still gate on growth_rate/coverage, not on this.
    """
    n_fires_attack = int((fires & attack_mask).sum())
    n_fires_benign = int((fires & benign_mask).sum())

    attack_support = n_fires_attack / n_attack if n_attack else 0.0
    benign_support = n_fires_benign / n_benign if n_benign else 0.0
    growth_rate = attack_support / (benign_support + _EPS)

    n_fires_total = n_fires_attack + n_fires_benign
    precision_attack = n_fires_attack / n_fires_total if n_fires_total else 0.0

    p_value = _chi_square_p_value(n_fires_attack, n_attack, n_fires_benign, n_benign)

    return {
        "n_attack": n_fires_attack,
        "n_benign": n_fires_benign,
        "attack_support": attack_support,
        "benign_support": benign_support,
        "growth_rate": growth_rate,
        "precision_attack": precision_attack,
        "p_value": p_value,
    }


def compute_predicate_contrast_stats(
    X: pd.DataFrame,
    y: pd.Series,
    column_predicate_map: dict[str, tuple[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    For every single categorical predicate column, and every pairwise AND
    combination (enumerated exhaustively -- the candidate space is a few
    dozen columns, so this is cheap and needs no mining library/pruning),
    compute attack_support, benign_support, growth_rate, lift, and the
    chi-square p-value on the 2x2 contingency table.

    Three structurally-uninformative candidate classes are pruned before any
    stats are computed, since neither can ever survive the downstream
    contrast-set filter regardless of the data:
      - Columns that are constant (all-fire or never-fire) within this
        population have zero variance, so growth_rate/coverage are
        degenerate for them -- there's nothing to discriminate.
      - Same-field cross-value pairs, e.g. category=EXPLOIT AND
        category=WEB_SERVER. These come from one-hot expanding a
        MULTI_VALUED_CATEGORICAL_FIELDS column (attribute_features.py), and
        an alert group has exactly one value for that field, so two
        different values of it can never co-fire -- the pair is always
        empty. Detected via column_predicate_map (column -> (attribute,
        value)): two columns sharing the same attribute but a different
        value are mutually exclusive by construction. Pass None to skip this
        check (falls back to enumerating every pair, as before).
      - Pairwise ANDs of two individually-non-constant columns that just
        never co-occur in this population (fires.sum() == 0) -- unlike the
        single-column case, this can't be detected from column_predicate_map
        alone, so it's checked per-combo right after computing `fires`,
        before predicate_support_stats()/chi-square are ever called on it.
        On the CSCAS cscas_pregrouped population this is ~59% of all
        single+pairwise candidates, so skipping the stats arithmetic for
        them (rather than computing and then discarding) is a real cost
        saving, not just a smaller output table.
    """
    base_rate = float(y.mean()) if len(y) else 0.0
    y_arr = y.to_numpy()
    attack_mask = y_arr == 1
    benign_mask = y_arr == 0
    n_attack = int(attack_mask.sum())
    n_benign = int(benign_mask.sum())
    n_total = len(X)

    columns = [c for c in X.columns if X[c].nunique(dropna=False) > 1]

    def _mutually_exclusive(a: str, b: str) -> bool:
        if column_predicate_map is None:
            return False
        pa = column_predicate_map.get(a)
        pb = column_predicate_map.get(b)
        return pa is not None and pb is not None and pa[0] == pb[0] and pa[1] != pb[1]

    candidates: list[tuple[str, ...]] = [(c,) for c in columns]
    candidates += [
        combo
        for combo in itertools.combinations(columns, 2)
        if not _mutually_exclusive(combo[0], combo[1])
    ]

    rows = []
    n_dead = 0
    for combo in candidates:
        if len(combo) == 1:
            fires = X[combo[0]].to_numpy().astype(bool)
        else:
            fires = X[combo[0]].to_numpy().astype(bool) & X[combo[1]].to_numpy().astype(
                bool
            )

        if not fires.any():
            # Never fires on either class -- attack_support and
            # benign_support are both 0, so growth_rate is degenerate (0)
            # and it can never clear either survivor direction in
            # filter_contrast_survivors. Drop it here rather than computing
            # (and later discarding) support stats and a chi-square test for it.
            n_dead += 1
            continue

        stats = predicate_support_stats(
            fires, attack_mask, benign_mask, n_attack, n_benign
        )
        attack_support = stats["attack_support"]
        benign_support = stats["benign_support"]
        lift = attack_support / (base_rate + _EPS) if base_rate else 0.0

        support_count = stats["n_attack"] + stats["n_benign"]
        support = support_count / n_total if n_total else 0.0

        rows.append(
            {
                "itemset": combo,
                "support": support,
                "support_count": support_count,
                "confidence_attack": attack_support,
                "confidence_benign": benign_support,
                "growth_rate": stats["growth_rate"],
                "precision_attack": stats["precision_attack"],
                "lift": lift,
                "p_value": stats["p_value"],
                "mining_type": "contrast_categorical",
                "n_attack": stats["n_attack"],
                "n_benign": stats["n_benign"],
            }
        )

    if n_dead:
        print(
            f"  compute_predicate_contrast_stats: {n_dead}/{len(candidates)} "
            "candidates never fire on either class (dropped before stats)"
        )

    # Explicit columns so a window with zero surviving candidates (e.g. every
    # categorical column happens to be constant in a small enough slice)
    # still returns a DataFrame with an "itemset" column -- pd.DataFrame([])
    # on an empty row list otherwise has no columns at all, which crashes
    # every downstream consumer that indexes by column name (e.g.
    # surviving_single_columns).
    result = pd.DataFrame(rows, columns=_CONTRAST_STATS_COLUMNS)
    # .attrs (not a real column) so EDA/diagnostics -- e.g. a funnel plot of
    # candidates -> non-dead -> survivors -- can report the pre-drop total
    # without re-deriving the enumeration logic above; ordinary consumers
    # that only care about the stats columns are unaffected.
    result.attrs["n_candidates_total"] = len(candidates)
    result.attrs["n_dead_dropped"] = n_dead
    return result


def filter_contrast_survivors(
    stats_df: pd.DataFrame,
    min_attack_coverage: float = 0.05,
    min_benign_coverage: float = 0.05,
    min_growth_rate: float = 3.0,
    min_growth_rate_attack: float | None = None,
    max_p_value: float | None = None,
) -> pd.DataFrame:
    """
    Keep a predicate/pair only if it is both meaningfully discriminative
    (growth_rate, or its reciprocal direction, clears the relevant growth-rate
    threshold) and has enough coverage on the class it discriminates toward --
    this is what stops a predicate that fires on a handful of attack groups
    out of tens of thousands from surviving purely because those few inflate
    its growth rate. An optional chi-square significance gate can be layered
    on top, most useful for pairwise predicates where attack-side counts get
    small fast.

    min_growth_rate gates the benign-leaning direction (via its reciprocal);
    min_growth_rate_attack, if set, gates the attack-leaning direction
    independently -- None (default) reuses min_growth_rate for both, exactly
    today's single-threshold behavior. Split thresholds exist because the two
    directions don't need the same bar: e.g. attacks are rarer, so a looser
    attack-side threshold may be needed to keep enough attack-leaning
    candidates, independent of how strict the benign side is.
    """
    if stats_df.empty:
        return stats_df

    attack_threshold = (
        min_growth_rate_attack
        if min_growth_rate_attack is not None
        else min_growth_rate
    )
    inv_threshold = 1.0 / min_growth_rate if min_growth_rate > 0 else float("inf")

    def _keep(row: pd.Series) -> bool:
        growth_rate = row["growth_rate"]
        attack_leaning = growth_rate >= attack_threshold
        benign_leaning = row["confidence_benign"] > 0 and growth_rate <= inv_threshold

        if not (
            attack_leaning or benign_leaning
        ):  # Drop dead candidates that don't discriminate either way
            return False
        if (
            attack_leaning and row["confidence_attack"] < min_attack_coverage
        ):  # Drop candidates that lean toward attack but don't cover enough attack instances
            return False
        if (
            benign_leaning and row["confidence_benign"] < min_benign_coverage
        ):  # Drop candidates that lean toward benign but don't cover enough benign instances
            return False
        if (
            max_p_value is not None
        ):  # Drop candidates that don't meet the chi-square significance threshold
            p_value = row["p_value"]
            if p_value is None or p_value >= max_p_value:
                return False
        return True

    mask = stats_df.apply(_keep, axis=1)
    return stats_df[mask].reset_index(drop=True)


def surviving_single_columns(stats_df: pd.DataFrame) -> list[str]:
    """
    Flatten survivor itemsets (singles and pairs) into the underlying single
    column names Step 2 needs in its training matrix -- if only a pair
    survives, both constituent columns must still be present for the tree to
    be able to recover that combination via nested splits.
    """
    cols: set[str] = set()
    for itemset in stats_df["itemset"]:
        cols.update(itemset)
    return sorted(cols)
