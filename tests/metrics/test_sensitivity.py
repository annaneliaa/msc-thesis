from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.metrics.sensitivity import (
    EFFECT_SIZE_THRESHOLDS,
    effect_size_label,
    expand_named_setting,
    main_effects,
    mining_setting_param_map,
    parameter_importance,
    restrict_to_fixed_levels,
)
from thesis.schemas.mining import (
    ContrastSetFilterConfig,
    DecisionTreeRuleConfig,
    MiningSettingSpec,
)


def test_effect_size_label_at_and_around_cohen_boundaries():
    assert effect_size_label(np.nan) == "n/a"
    assert effect_size_label(0.0) == "negligible"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["small"] - 0.001) == "negligible"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["small"]) == "small"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["medium"] - 0.001) == "small"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["medium"]) == "medium"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["large"] - 0.001) == "medium"
    assert effect_size_label(EFFECT_SIZE_THRESHOLDS["large"]) == "large"
    assert effect_size_label(0.5) == "large"


def test_expand_named_setting_adds_one_column_per_param_and_preserves_nulls():
    df = pd.DataFrame(
        {
            "feature_set": ["symbolic", "symbolic", "baseline"],
            "mining_setting": ["a", "b", np.nan],
        }
    )
    param_map = {
        "a": {"min_growth_rate": 3.0, "max_depth": 4},
        "b": {"min_growth_rate": 5.0, "max_depth": 3},
    }
    out = expand_named_setting(df, param_map, col="mining_setting")

    assert out["min_growth_rate"].tolist()[:2] == [3.0, 5.0]
    assert pd.isna(out["min_growth_rate"].iloc[2])
    assert out["max_depth"].tolist()[:2] == [4, 3]
    assert pd.isna(out["max_depth"].iloc[2])
    # original columns untouched
    assert list(out["mining_setting"]) == list(df["mining_setting"])


def test_expand_named_setting_works_for_an_unrelated_preset_axis():
    # Not mining-related at all -- any {name: {param: value}} map works.
    df = pd.DataFrame({"filter_preset": ["strict", "simple", "strict"]})
    param_map = {
        "strict": {"min_confidence_attack": 0.9, "min_k": 2},
        "simple": {"min_confidence_attack": 0.5, "min_k": 1},
    }
    out = expand_named_setting(df, param_map, col="filter_preset")
    assert out["min_confidence_attack"].tolist() == [0.9, 0.5, 0.9]
    assert out["min_k"].tolist() == [2, 1, 2]


def test_mining_setting_param_map_reads_spec_fields_including_fixed_ones():
    specs = [
        MiningSettingSpec(
            name="gr3.0_md4",
            contrast=ContrastSetFilterConfig(min_growth_rate=3.0),
            tree=DecisionTreeRuleConfig(max_depth=4),
        ),
        MiningSettingSpec(
            name="gr5.0_md3",
            contrast=ContrastSetFilterConfig(min_growth_rate=5.0),
            tree=DecisionTreeRuleConfig(max_depth=3),
        ),
    ]
    param_map = mining_setting_param_map(specs)

    assert param_map["gr3.0_md4"]["min_growth_rate"] == 3.0
    assert param_map["gr3.0_md4"]["max_depth"] == 4
    # min_samples_leaf isn't varied by either spec above (both use the
    # DecisionTreeRuleConfig default) but should still surface as a
    # same-valued column, not be silently dropped.
    assert (
        param_map["gr3.0_md4"]["min_samples_leaf"]
        == param_map["gr5.0_md3"]["min_samples_leaf"]
    )


def test_restrict_to_fixed_levels_drops_non_matching_rows_but_keeps_nulls():
    df = pd.DataFrame(
        {
            "feature_set": ["symbolic", "symbolic", "symbolic", "baseline"],
            "min_growth_rate": [3.0, 5.0, 3.0, np.nan],
            "max_depth": [3, 3, 5, np.nan],
        }
    )
    out = restrict_to_fixed_levels(df, {"min_growth_rate": 3.0})

    # rows at the winning level are kept, the off-level row is dropped, and
    # the baseline row (null -- nothing to match against) is always kept.
    assert out["min_growth_rate"].tolist()[:-1] == [3.0, 3.0]
    assert pd.isna(out["min_growth_rate"].iloc[-1])
    assert len(out) == 3
    # max_depth is untouched -- both 3 and 5 survive since it wasn't pinned
    assert set(out["max_depth"].dropna()) == {3, 5}


def test_restrict_to_fixed_levels_combines_multiple_pins_and_ignores_missing_columns():
    df = pd.DataFrame(
        {
            "min_growth_rate": [3.0, 3.0, 5.0],
            "max_depth": [3, 5, 3],
        }
    )
    out = restrict_to_fixed_levels(
        df, {"min_growth_rate": 3.0, "max_depth": 3, "nonexistent": "x"}
    )
    assert len(out) == 1
    assert out.iloc[0]["max_depth"] == 3


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
