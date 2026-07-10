from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.metrics.sensitivity import (
    expand_mining_setting,
    main_effects,
    parameter_importance,
    parse_mining_setting,
)


def test_parse_mining_setting_extracts_thresholds():
    assert parse_mining_setting("gr3.0_md5") == {"min_growth_rate": 3.0, "max_depth": 5}
    assert parse_mining_setting("gr4.5_md12") == {
        "min_growth_rate": 4.5,
        "max_depth": 12,
    }


def test_parse_mining_setting_handles_missing_or_unmatched():
    assert parse_mining_setting(None) == {"min_growth_rate": None, "max_depth": None}
    assert parse_mining_setting(np.nan) == {"min_growth_rate": None, "max_depth": None}
    assert parse_mining_setting("baseline") == {
        "min_growth_rate": None,
        "max_depth": None,
    }


def test_expand_mining_setting_adds_columns_and_preserves_nulls():
    df = pd.DataFrame(
        {
            "feature_set": ["symbolic", "symbolic", "baseline"],
            "mining_setting": ["gr3.0_md4", "gr5.0_md3", np.nan],
        }
    )
    out = expand_mining_setting(df)
    assert out["min_growth_rate"].tolist()[:2] == [3.0, 5.0]
    assert pd.isna(out["min_growth_rate"].iloc[2])
    assert out["max_depth"].tolist()[:2] == [4, 3]
    assert pd.isna(out["max_depth"].iloc[2])
    # original columns untouched
    assert list(out["mining_setting"]) == list(df["mining_setting"])


def _config_level_df() -> pd.DataFrame:
    # granularity drives score_mean strongly; max_depth is noise.
    rows = []
    for granularity, base in [(0.1, 0.9), (0.5, 0.6), (1.0, 0.3)]:
        for max_depth in [3, 4, 5]:
            rows.append(
                {
                    "granularity": granularity,
                    "max_depth": max_depth,
                    "model": "logreg",  # constant -- only one level swept
                    "score_mean": base + 0.001 * max_depth,
                }
            )
    return pd.DataFrame(rows)


def test_parameter_importance_ranks_the_dominant_parameter_first():
    df = _config_level_df()
    result = parameter_importance(
        df, score_col="score_mean", param_cols=["granularity", "max_depth", "model"]
    )

    ranked_params = list(result["parameter"])
    assert ranked_params[0] == "granularity"
    granularity_row = result[result["parameter"] == "granularity"].iloc[0]
    max_depth_row = result[result["parameter"] == "max_depth"].iloc[0]
    assert granularity_row["eta_squared"] > 0.9
    assert granularity_row["eta_squared"] > max_depth_row["eta_squared"]
    assert granularity_row["p_value"] < 0.01


def test_parameter_importance_flags_single_level_parameter():
    df = _config_level_df()
    result = parameter_importance(df, score_col="score_mean", param_cols=["model"])
    row = result.iloc[0]
    assert row["n_levels"] == 1
    assert np.isnan(row["eta_squared"])
    assert np.isnan(row["p_value"])


def test_parameter_importance_skips_missing_columns():
    df = _config_level_df()
    result = parameter_importance(
        df, score_col="score_mean", param_cols=["granularity", "nonexistent"]
    )
    assert set(result["parameter"]) == {"granularity"}


def test_main_effects_marginalizes_over_other_parameters():
    df = _config_level_df()
    effects = main_effects(df, score_col="score_mean", param_cols=["granularity"])
    assert len(effects) == 3  # one row per granularity level

    row = effects[effects["level"] == 0.1].iloc[0]
    expected_mean = df.loc[df["granularity"] == 0.1, "score_mean"].mean()
    assert row["mean"] == pytest.approx(expected_mean)
    assert row["n"] == 3
