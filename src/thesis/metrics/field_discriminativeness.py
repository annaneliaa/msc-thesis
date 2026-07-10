"""Raw-field discriminativeness statistics, computed directly on the dataset.

These are dataset-level statistics, not tied to any mining run: given a raw
field and a binary label, how strongly does that one field on its own
separate the two classes? They complement `growth_rate`
(`thesis.mining.attribute_contrast_mining.compute_predicate_contrast_stats`,
computed per *predicate value* after mining has already enumerated
candidates) with a per-*field* view computed before any mining happens --
"which raw fields are likely to carry useful signal" ahead of running the
attribute mining pipeline, and a reference to revisit later against the
mining pipeline's actual output.

- Categorical fields: `local_attack_rates` (per-value attack rate vs. the
  dataset base rate) and `cramers_v` (a single chi-square-derived
  association score per field, 0-1, comparable across fields regardless of
  how many values each takes).
- Numeric fields: `point_biserial_score` (Pearson correlation between the
  field and the binary label) and `auc_separability` (the field's raw value
  used directly as a one-dimensional classifier score).
- Both: `mutual_information_score`, an encoding-agnostic measure that
  captures any statistical dependency, not only the monotonic/single-
  threshold relationships the other measures are each individually
  sensitive to.

`field_discriminativeness_table` runs all of the above over a set of
categorical and numeric fields and lays them out in one ranked table.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


def local_attack_rates(
    df: pd.DataFrame, field_col: str, label_col: str
) -> pd.DataFrame:
    """Per-value attack rate for `field_col`, alongside the dataset base rate.

    One row per distinct value `v` of `field_col`: its support count,
    `attack_rate` (`P(label=1 | field=v)`), the dataset's overall
    `base_rate`, and `deviation` (`attack_rate - base_rate`) -- the
    "how far from uninformative" number to rank values by. Sorted by
    `abs(deviation)` descending, so the most discriminative values (in
    either direction) come first.
    """
    base_rate = df[label_col].mean()
    grouped = df.groupby(field_col, dropna=False)[label_col].agg(["mean", "count"])
    out = grouped.rename(
        columns={"mean": "attack_rate", "count": "support"}
    ).reset_index()
    out = out.rename(columns={field_col: "value"})
    out["base_rate"] = base_rate
    out["deviation"] = out["attack_rate"] - base_rate
    return out.reindex(
        out["deviation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association between a categorical field and the label.

    0 (no association) to 1 (perfect association), from the chi-square
    statistic of `x`'s value x `y`'s value contingency table -- summarizes
    an entire field's discriminativeness in one number, comparable across
    fields with different numbers of distinct values (unlike raw chi-square).
    `NaN` if `x` or `y` has fewer than 2 distinct values (no contingency
    table to test).
    """
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    return float(association(table.to_numpy(), method="cramer"))


def _valid_pairs(
    x: pd.Series, y: pd.Series, sentinel: float | None = None
) -> tuple[pd.Series, pd.Series]:
    mask = x.notna() & y.notna()
    if sentinel is not None:
        mask &= x != sentinel
    return x[mask], y[mask]


def point_biserial_score(
    x: pd.Series, y: pd.Series, sentinel: float | None = None
) -> tuple[float, float]:
    """Point-biserial correlation between a numeric field and the binary label.

    Pearson correlation between `x` and `y` (treating `y`'s 0/1 label as
    numeric) -- captures the strength and direction of an (approximately)
    monotonic relationship. Rows where `x == sentinel` (e.g. `-1` for a
    "not applicable" numeric field, this codebase's convention -- see
    `thesis.mining.attribute_features._NOT_APPLICABLE`) are dropped first if
    `sentinel` is given, so a placeholder value doesn't get treated as real
    data. Returns `(r, p_value)`; `(nan, nan)` if fewer than 2 valid rows
    remain, or if `x` or `y` is constant among the valid rows (correlation
    is undefined either way -- e.g. a field whose only non-sentinel rows all
    happen to share one label).
    """
    x_valid, y_valid = _valid_pairs(x, y, sentinel)
    if len(x_valid) < 2 or x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return float("nan"), float("nan")
    result = pointbiserialr(x_valid, y_valid)
    return float(result.correlation), float(result.pvalue)


def auc_separability(
    x: pd.Series, y: pd.Series, sentinel: float | None = None
) -> float:
    """AUC from using a numeric field's raw value as a 1-D classifier score.

    `roc_auc_score(y, x)`, reported as `max(auc, 1-auc)` so a field that
    separates the classes but points the "wrong" way (low values = attack)
    scores the same as one that points the "right" way -- this is meant as
    a separability measure, not a directional one (see
    `point_biserial_score` for direction). `0.5` = no separative power
    (random guessing), `1.0` = perfect separation. Rows where `x ==
    sentinel` are dropped first if given. `NaN` if fewer than 2 valid rows
    remain, or `y` has only one class among them.
    """
    x_valid, y_valid = _valid_pairs(x, y, sentinel)
    if len(x_valid) < 2 or y_valid.nunique() < 2:
        return float("nan")
    auc = roc_auc_score(y_valid, x_valid)
    return float(max(auc, 1 - auc))


def mutual_information_score(
    x: pd.Series,
    y: pd.Series,
    discrete: bool,
    sentinel: float | None = None,
    random_state: int = 0,
) -> float:
    """Mutual information `I(field; label)`, categorical or numeric alike.

    Unifies `cramers_v`/`point_biserial_score`/`auc_separability` under one
    encoding-agnostic scale: non-negative, `0` iff independent, with no
    assumption of a monotonic or single-threshold relationship. Categorical
    `x` is label-encoded first (`pd.factorize`) since
    `sklearn.feature_selection.mutual_info_classif` expects numeric input;
    `discrete=True` for categorical fields, `False` for numeric ones. Rows
    where `x == sentinel` are dropped first if given.
    """
    x_valid, y_valid = _valid_pairs(x, y, sentinel)
    if len(x_valid) < 2 or x_valid.nunique() < 2:
        return float("nan")
    values = pd.factorize(x_valid)[0] if discrete else x_valid.to_numpy()
    mi = mutual_info_classif(
        values.reshape(-1, 1),
        y_valid.to_numpy(),
        discrete_features=[discrete],
        random_state=random_state,
    )
    return float(mi[0])


def field_discriminativeness_table(
    df: pd.DataFrame,
    categorical_fields: list[str],
    numeric_fields: list[str],
    label_col: str,
    sentinels: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One ranked row per field, combining every measure above.

    `cramers_v` is populated for `categorical_fields` only,
    `point_biserial_r`/`auc_separability` for `numeric_fields` only (`NaN`
    where a measure doesn't apply to that field's type), and
    `mutual_information` for every field -- the one column comparable across
    both types, and what the table is sorted by (descending). `sentinels`
    is an optional `{field: sentinel_value}` map (only meaningful for
    numeric fields) forwarded to `point_biserial_score`/`auc_separability`/
    `mutual_information_score`.
    """
    sentinels = sentinels or {}
    rows = []
    for field in categorical_fields:
        rows.append(
            {
                "field": field,
                "field_type": "categorical",
                "cramers_v": cramers_v(df[field], df[label_col]),
                "point_biserial_r": float("nan"),
                "auc_separability": float("nan"),
                "mutual_information": mutual_information_score(
                    df[field], df[label_col], discrete=True
                ),
            }
        )
    for field in numeric_fields:
        sentinel = sentinels.get(field)
        r, _ = point_biserial_score(df[field], df[label_col], sentinel=sentinel)
        rows.append(
            {
                "field": field,
                "field_type": "numeric",
                "cramers_v": float("nan"),
                "point_biserial_r": r,
                "auc_separability": auc_separability(
                    df[field], df[label_col], sentinel=sentinel
                ),
                "mutual_information": mutual_information_score(
                    df[field], df[label_col], discrete=False, sentinel=sentinel
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        "mutual_information", ascending=False, na_position="last"
    ).reset_index(drop=True)
