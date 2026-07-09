from __future__ import annotations

import pandas as pd
import pytest

from thesis.metrics.baseline_comparison import pair_with_baseline, summarize_comparison


def _per_window_df() -> pd.DataFrame:
    # Two windows, one model. Each window has a baseline row plus two
    # symbolic rows (mining settings "a" and "b"). Setting "a" trades FPs for
    # a missed attack (higher AUC, fewer FPs, one more FN); setting "b" is
    # identical to baseline (delta == 0 everywhere).
    rows = [
        # window 0
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 0,
            "model": "logreg",
            "feature_set": "baseline",
            "mining_setting": None,
            "auc": 0.90,
            "precision": 0.70,
            "recall": 0.90,
            "tp": 9,
            "fp": 10,
            "tn": 90,
            "fn": 1,
        },
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 0,
            "model": "logreg",
            "feature_set": "symbolic",
            "mining_setting": "a",
            "auc": 0.95,
            "precision": 0.85,
            "recall": 0.80,
            "tp": 8,
            "fp": 4,
            "tn": 96,
            "fn": 2,
        },
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 0,
            "model": "logreg",
            "feature_set": "symbolic",
            "mining_setting": "b",
            "auc": 0.90,
            "precision": 0.70,
            "recall": 0.90,
            "tp": 9,
            "fp": 10,
            "tn": 90,
            "fn": 1,
        },
        # window 1
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 1,
            "model": "logreg",
            "feature_set": "baseline",
            "mining_setting": None,
            "auc": 0.88,
            "precision": 0.60,
            "recall": 0.85,
            "tp": 17,
            "fp": 20,
            "tn": 80,
            "fn": 3,
        },
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 1,
            "model": "logreg",
            "feature_set": "symbolic",
            "mining_setting": "a",
            "auc": 0.93,
            "precision": 0.80,
            "recall": 0.80,
            "tp": 16,
            "fp": 8,
            "tn": 92,
            "fn": 4,
        },
        {
            "scenario": "s",
            "granularity": 0.5,
            "window_id": 1,
            "model": "logreg",
            "feature_set": "symbolic",
            "mining_setting": "b",
            "auc": 0.88,
            "precision": 0.60,
            "recall": 0.85,
            "tp": 17,
            "fp": 20,
            "tn": 80,
            "fn": 3,
        },
    ]
    return pd.DataFrame(rows)


def test_pair_with_baseline_joins_same_window_only():
    paired = pair_with_baseline(_per_window_df())

    # 2 windows x 2 symbolic settings = 4 paired rows.
    assert len(paired) == 4
    # Baseline columns should never leak into unrelated windows/models.
    assert set(paired["window_id"]) == {0, 1}
    assert "auc_base" in paired.columns and "auc_sym" in paired.columns


def test_pair_with_baseline_computes_deltas():
    paired = pair_with_baseline(_per_window_df())
    row = paired[(paired["mining_setting"] == "a") & (paired["window_id"] == 0)].iloc[0]

    assert row["auc_delta"] == pytest.approx(0.95 - 0.90)
    assert row["precision_delta"] == pytest.approx(0.85 - 0.70)
    assert row["recall_delta"] == pytest.approx(0.80 - 0.90)
    assert row["fp_reduction"] == pytest.approx(10 - 4)  # baseline_fp - symbolic_fp
    assert row["fp_reduction_pct"] == pytest.approx((10 - 4) / 10)
    assert row["fn_increase"] == pytest.approx(2 - 1)  # symbolic_fn - baseline_fn

    row_b = paired[(paired["mining_setting"] == "b") & (paired["window_id"] == 0)].iloc[
        0
    ]
    assert row_b["auc_delta"] == pytest.approx(0.0)
    assert row_b["fp_reduction"] == pytest.approx(0.0)
    assert row_b["fn_increase"] == pytest.approx(0.0)


def test_summarize_comparison_averages_across_windows():
    paired = pair_with_baseline(_per_window_df())
    summary = summarize_comparison(paired)

    assert set(summary["mining_setting"]) == {"a", "b"}
    row_a = summary[summary["mining_setting"] == "a"].iloc[0]
    row_b = summary[summary["mining_setting"] == "b"].iloc[0]

    assert row_a["n_windows"] == 2
    assert row_a["fp_reduction_mean"] == pytest.approx(((10 - 4) + (20 - 8)) / 2)
    assert row_a["fn_increase_mean"] == pytest.approx(((2 - 1) + (4 - 3)) / 2)
    assert row_a["auc_delta_mean"] > 0
    assert row_b["auc_delta_mean"] == pytest.approx(0.0)
    assert row_b["fp_reduction_mean"] == pytest.approx(0.0)


def test_summarize_comparison_raises_when_no_group_cols_present():
    paired = pair_with_baseline(_per_window_df())
    with pytest.raises(ValueError):
        summarize_comparison(paired, group_cols=["nonexistent"])
