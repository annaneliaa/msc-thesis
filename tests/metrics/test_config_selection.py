from __future__ import annotations

import pandas as pd
import pytest

from thesis.metrics.config_selection import (
    apply_floor_check,
    rank_configs,
    select_top_k,
    summarize_configs,
)


def _per_window_df() -> pd.DataFrame:
    # Two configs x three windows. config "a" has a higher mean AUC but one
    # bad window; config "b" is lower but rock-steady.
    rows = [
        {
            "feature_set": "symbolic",
            "mining_setting": "a",
            "granularity": 0.5,
            "n_features": 10,
            "auc": 0.99,
        },
        {
            "feature_set": "symbolic",
            "mining_setting": "a",
            "granularity": 0.5,
            "n_features": 10,
            "auc": 0.99,
        },
        {
            "feature_set": "symbolic",
            "mining_setting": "a",
            "granularity": 0.5,
            "n_features": 10,
            "auc": 0.90,
        },
        {
            "feature_set": "symbolic",
            "mining_setting": "b",
            "granularity": 0.5,
            "n_features": 4,
            "auc": 0.95,
        },
        {
            "feature_set": "symbolic",
            "mining_setting": "b",
            "granularity": 0.5,
            "n_features": 4,
            "auc": 0.95,
        },
        {
            "feature_set": "symbolic",
            "mining_setting": "b",
            "granularity": 0.5,
            "n_features": 4,
            "auc": 0.95,
        },
    ]
    return pd.DataFrame(rows)


def test_summarize_configs_groups_and_aggregates():
    df = _per_window_df()
    summary = summarize_configs(
        df, score_col="auc", group_cols=["feature_set", "mining_setting", "granularity"]
    )

    assert set(summary["mining_setting"]) == {"a", "b"}
    row_a = summary[summary["mining_setting"] == "a"].iloc[0]
    row_b = summary[summary["mining_setting"] == "b"].iloc[0]

    assert row_a["n_windows"] == 3
    assert row_a["score_min"] == pytest.approx(0.90)
    assert row_a["score_mean"] == pytest.approx((0.99 + 0.99 + 0.90) / 3)
    assert row_b["score_std"] == pytest.approx(0.0)
    assert row_a["n_features_mean"] == pytest.approx(10)
    assert row_b["n_features_mean"] == pytest.approx(4)


def test_summarize_configs_ignores_missing_group_cols():
    df = _per_window_df()
    # "model" isn't a column in this fixture -- should be silently dropped
    # rather than erroring, so the same call works across scenarios.
    summary = summarize_configs(
        df,
        score_col="auc",
        group_cols=["feature_set", "mining_setting", "granularity", "model"],
    )
    assert "model" not in summary.columns


def test_summarize_configs_raises_when_no_group_cols_present():
    df = _per_window_df()
    with pytest.raises(ValueError):
        summarize_configs(df, score_col="auc", group_cols=["nonexistent"])


def test_rank_configs_penalizes_instability():
    df = _per_window_df()
    summary = summarize_configs(
        df, score_col="auc", group_cols=["feature_set", "mining_setting", "granularity"]
    )

    # config "a" has the higher raw mean...
    a_mean = summary.loc[summary["mining_setting"] == "a", "score_mean"].iloc[0]
    b_mean = summary.loc[summary["mining_setting"] == "b", "score_mean"].iloc[0]
    assert a_mean > b_mean

    # ...but a strong-enough std penalty should flip the ranking.
    ranked = rank_configs(summary, lam=0.5)
    assert ranked.iloc[0]["mining_setting"] == "b"

    # lam=0 (no penalty) should rank by raw mean instead.
    ranked_no_penalty = rank_configs(summary, lam=0.0)
    assert ranked_no_penalty.iloc[0]["mining_setting"] == "a"


def test_rank_configs_tiebreak_prefers_fewer_features_within_tolerance():
    summary = pd.DataFrame(
        [
            {
                "mining_setting": "big",
                "score_mean": 0.9001,
                "score_std": 0.0,
                "n_features_mean": 100,
            },
            {
                "mining_setting": "small",
                "score_mean": 0.9000,
                "score_std": 0.0,
                "n_features_mean": 10,
            },
        ]
    )
    # Scores are within tiebreak_tol of each other -> smaller schema wins.
    ranked = rank_configs(summary, lam=0.0, tiebreak_tol=0.01)
    assert ranked.iloc[0]["mining_setting"] == "small"

    # Without a tiebreak tolerance, the (tiny) raw score difference decides.
    ranked_strict = rank_configs(summary, lam=0.0, tiebreak_tol=0.0)
    assert ranked_strict.iloc[0]["mining_setting"] == "big"


def test_select_top_k_and_floor_check():
    df = _per_window_df()
    summary = summarize_configs(
        df, score_col="auc", group_cols=["feature_set", "mining_setting", "granularity"]
    )
    ranked = rank_configs(summary, lam=0.5)
    top1 = select_top_k(ranked, k=1)
    assert len(top1) == 1

    checked = apply_floor_check(ranked, floor_threshold=0.92, floor_col="score_min")
    passes = dict(zip(checked["mining_setting"], checked["passes_floor"]))
    assert passes["b"] is True
    assert passes["a"] is False  # its worst window (0.90) is below the floor
