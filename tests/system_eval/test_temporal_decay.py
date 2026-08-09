from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.system_eval.temporal_decay import _build_decay_summary
from thesis.metrics.shortlist import load_shortlist


# ---- _build_decay_summary ----------------------------------------------------


def _horizon_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "feature_set": "symbolic",
        "mining_setting": "gr3.0_md4",
        "granularity": 0.1,
        "model": "logreg",
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_build_decay_summary_uses_h0_and_last_horizon_reached():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": 0.99, "fpr": 0.01},
            {"horizon_window_index": 1, "auc": 0.95, "fpr": 0.02},
            {"horizon_window_index": 3, "auc": 0.90, "fpr": 0.03},
        ]
    )
    summary = _build_decay_summary(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["h_max"] == 3
    assert row["auc_at_h0"] == pytest.approx(0.99)
    assert row["auc_at_h3"] == pytest.approx(0.90)
    assert row["decay_rate_auc"] == pytest.approx(0.09)
    assert row["fpr_at_h0"] == pytest.approx(0.01)
    assert row["fpr_at_h3"] == pytest.approx(0.03)
    assert row["fpr_drift"] == pytest.approx(0.02)


def test_build_decay_summary_one_row_per_config():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": 0.99, "fpr": 0.01, "model": "logreg"},
            {"horizon_window_index": 2, "auc": 0.91, "fpr": 0.03, "model": "logreg"},
            {"horizon_window_index": 0, "auc": 0.97, "fpr": 0.02, "model": "rf"},
            {"horizon_window_index": 2, "auc": 0.85, "fpr": 0.05, "model": "rf"},
        ]
    )
    summary = _build_decay_summary(df)
    assert len(summary) == 2
    assert set(summary["model"]) == {"logreg", "rf"}


def test_build_decay_summary_nan_when_either_end_is_nan():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": np.nan, "fpr": 0.01},
            {"horizon_window_index": 1, "auc": 0.90, "fpr": 0.02},
        ]
    )
    summary = _build_decay_summary(df)
    assert np.isnan(summary.iloc[0]["decay_rate_auc"])


def test_build_decay_summary_empty_input():
    assert _build_decay_summary(pd.DataFrame()).empty


# ---- load_shortlist ----------------------------------------------------------


def _write_csv(tmp_path, rows: list[dict]) -> Path:
    path = tmp_path / "shortlist.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_shortlist_valid_round_trip(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "logreg",
            },
            {
                "feature_set": "symbolic",
                "mining_setting": "gr3.0_md3",
                "granularity": 0.2,
                "model": "logreg",
            },
        ],
    )
    configs = load_shortlist(path)
    assert len(configs) == 2
    assert configs[0].feature_set == "baseline"
    assert configs[0].mining_setting is None
    assert configs[1].mining_setting == "gr3.0_md3"
    assert configs[1].granularity == pytest.approx(0.2)


def test_load_shortlist_missing_column_raises(tmp_path):
    path = tmp_path / "shortlist.csv"
    pd.DataFrame(
        [{"feature_set": "baseline", "granularity": 0.2, "model": "logreg"}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="mining_setting"):
        load_shortlist(path)


def test_load_shortlist_baseline_with_mining_setting_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "gr3.0_md3",
                "granularity": 0.2,
                "model": "logreg",
            }
        ],
    )
    with pytest.raises(ValueError):
        load_shortlist(path)


def test_load_shortlist_symbolic_without_mining_setting_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "symbolic",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "logreg",
            }
        ],
    )
    with pytest.raises(ValueError):
        load_shortlist(path)


def test_load_shortlist_unknown_model_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "not_a_real_model",
            }
        ],
    )
    with pytest.raises(ValueError, match="not_a_real_model"):
        load_shortlist(path)
