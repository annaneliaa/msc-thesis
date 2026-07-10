from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.metrics.field_discriminativeness import (
    auc_separability,
    cramers_v,
    field_discriminativeness_table,
    local_attack_rates,
    mutual_information_score,
    point_biserial_score,
)


def test_local_attack_rates_reports_deviation_from_base_rate():
    df = pd.DataFrame(
        {
            "field": ["a"] * 10 + ["b"] * 10,
            "label": [1] * 10 + [0] * 10,  # base rate 0.5
        }
    )
    out = local_attack_rates(df, "field", "label")

    row_a = out[out["value"] == "a"].iloc[0]
    row_b = out[out["value"] == "b"].iloc[0]
    assert row_a["attack_rate"] == pytest.approx(1.0)
    assert row_a["support"] == 10
    assert row_a["base_rate"] == pytest.approx(0.5)
    assert row_a["deviation"] == pytest.approx(0.5)
    assert row_b["attack_rate"] == pytest.approx(0.0)
    assert row_b["deviation"] == pytest.approx(-0.5)
    # sorted by |deviation| descending -- both tie here, but both must lead
    assert set(out["value"].tolist()) == {"a", "b"}


def test_local_attack_rates_ranks_most_discriminative_value_first():
    df = pd.DataFrame(
        {
            "field": ["rare"] * 2 + ["common"] * 18,
            "label": [1, 1] + [1] * 9 + [0] * 9,  # rare: 100% attack, common: 50%
        }
    )
    out = local_attack_rates(df, "field", "label")
    assert out.iloc[0]["value"] == "rare"


def _perfectly_associated_categorical(n_per_group: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": ["A"] * n_per_group + ["B"] * n_per_group,
            "label": [1] * n_per_group + [0] * n_per_group,
        }
    )


def test_cramers_v_perfect_association_is_one():
    df = _perfectly_associated_categorical()
    v = cramers_v(df["field"], df["label"])
    assert v == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_independent_field_is_near_zero():
    # Balanced 2x2 table with no association: each field value has the same
    # attack rate as the other.
    df = pd.DataFrame(
        {
            "field": ["A", "A", "A", "A", "B", "B", "B", "B"] * 20,
            "label": [1, 1, 0, 0, 1, 1, 0, 0] * 20,
        }
    )
    v = cramers_v(df["field"], df["label"])
    assert v == pytest.approx(0.0, abs=1e-9)


def test_cramers_v_returns_nan_for_constant_field():
    df = pd.DataFrame({"field": ["A"] * 10, "label": [1, 0] * 5})
    assert np.isnan(cramers_v(df["field"], df["label"]))


def test_point_biserial_score_perfect_correlation():
    x = pd.Series([0.0, 0.0, 1.0, 1.0] * 10)
    y = pd.Series([0, 0, 1, 1] * 10)
    r, p = point_biserial_score(x, y)
    assert r == pytest.approx(1.0, abs=1e-9)
    assert p < 0.01


def test_point_biserial_score_drops_sentinel_rows():
    # Without dropping the sentinel (-1) rows, these corrupt an otherwise
    # perfect correlation.
    x = pd.Series([0.0, 0.0, 1.0, 1.0] * 10 + [-1.0] * 20)
    y = pd.Series([0, 0, 1, 1] * 10 + [1, 0] * 10)

    r_with_sentinel, _ = point_biserial_score(x, y)
    r_dropped, _ = point_biserial_score(x, y, sentinel=-1.0)

    assert r_dropped == pytest.approx(1.0, abs=1e-9)
    assert r_dropped > r_with_sentinel


def test_point_biserial_score_nan_when_x_constant():
    x = pd.Series([1.0] * 10)
    y = pd.Series([1, 0] * 5)
    r, p = point_biserial_score(x, y)
    assert np.isnan(r)
    assert np.isnan(p)


def test_point_biserial_score_nan_when_y_constant_among_valid_rows():
    # After dropping the sentinel, every remaining row shares the same label
    # -- correlation is undefined (not just "x is constant").
    x = pd.Series([0.0, 1.0, 2.0] + [-1.0] * 10)
    y = pd.Series([1, 1, 1] + [0] * 10)
    r, p = point_biserial_score(x, y, sentinel=-1.0)
    assert np.isnan(r)
    assert np.isnan(p)


def test_auc_separability_perfect_separation_either_direction():
    x_ascending = pd.Series([0.0, 0.0, 1.0, 1.0] * 10)
    y = pd.Series([0, 0, 1, 1] * 10)
    assert auc_separability(x_ascending, y) == pytest.approx(1.0)

    # Field points the "wrong" way (low value = attack) -- still perfectly
    # separable, so still scores 1.0 thanks to max(auc, 1-auc).
    x_descending = pd.Series([1.0, 1.0, 0.0, 0.0] * 10)
    assert auc_separability(x_descending, y) == pytest.approx(1.0)


def test_auc_separability_no_separation_is_half():
    # Every field value co-occurs equally with both labels.
    x = pd.Series([0.0, 1.0] * 20)
    y = pd.Series([0, 1, 1, 0] * 10)
    assert auc_separability(x, y) == pytest.approx(0.5, abs=1e-9)


def test_mutual_information_score_categorical_perfect_vs_independent():
    perfect = _perfectly_associated_categorical()
    mi_perfect = mutual_information_score(
        perfect["field"], perfect["label"], discrete=True
    )

    independent = pd.DataFrame(
        {
            "field": ["A", "A", "A", "A", "B", "B", "B", "B"] * 20,
            "label": [1, 1, 0, 0, 1, 1, 0, 0] * 20,
        }
    )
    mi_independent = mutual_information_score(
        independent["field"], independent["label"], discrete=True
    )

    assert mi_perfect > 0.1
    assert mi_independent < mi_perfect
    assert mi_independent == pytest.approx(0.0, abs=0.05)


def test_mutual_information_score_drops_sentinel_rows():
    x = pd.Series([0.0, 0.0, 1.0, 1.0] * 10 + [-1.0] * 20)
    y = pd.Series([0, 0, 1, 1] * 10 + [1, 0] * 10)

    mi_dropped = mutual_information_score(x, y, discrete=False, sentinel=-1.0)
    assert mi_dropped > 0.0


def test_field_discriminativeness_table_combines_categorical_and_numeric():
    df = pd.DataFrame(
        {
            "cat_field": ["A"] * 20 + ["B"] * 20,
            "num_field": [1.0] * 20 + [0.0] * 20,
            "label": [1] * 20 + [0] * 20,
        }
    )
    out = field_discriminativeness_table(
        df,
        categorical_fields=["cat_field"],
        numeric_fields=["num_field"],
        label_col="label",
    )

    assert set(out["field"]) == {"cat_field", "num_field"}
    cat_row = out[out["field"] == "cat_field"].iloc[0]
    num_row = out[out["field"] == "num_field"].iloc[0]

    assert cat_row["field_type"] == "categorical"
    assert cat_row["cramers_v"] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(cat_row["point_biserial_r"])
    assert np.isnan(cat_row["auc_separability"])

    assert num_row["field_type"] == "numeric"
    assert np.isnan(num_row["cramers_v"])
    assert num_row["point_biserial_r"] == pytest.approx(1.0, abs=1e-9)
    assert num_row["auc_separability"] == pytest.approx(1.0)

    # both fields perfectly discriminate here, so both get high MI
    assert (out["mutual_information"] > 0.1).all()
    # sorted descending by mutual_information
    assert out["mutual_information"].is_monotonic_decreasing


def test_field_discriminativeness_table_respects_sentinels():
    df = pd.DataFrame(
        {
            "num_field": [0.0, 0.0, 1.0, 1.0] * 10 + [-1.0] * 20,
            "label": [0, 0, 1, 1] * 10 + [1, 0] * 10,
        }
    )
    out = field_discriminativeness_table(
        df,
        categorical_fields=[],
        numeric_fields=["num_field"],
        label_col="label",
        sentinels={"num_field": -1.0},
    )
    row = out.iloc[0]
    assert row["point_biserial_r"] == pytest.approx(1.0, abs=1e-9)
