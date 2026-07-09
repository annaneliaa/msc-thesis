from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.experiments import rolling_walk_forward as rwf
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.schemas.experiments import RollingWalkForwardConfig


# ---- _build_walk_forward_summary --------------------------------------------


def _step_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "feature_set": "symbolic",
        "mining_setting": "gr3.0_md4",
        "granularity": 0.1,
        "model": "logreg",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_build_walk_forward_summary_empty_input():
    assert rwf._build_walk_forward_summary(pd.DataFrame()).empty


def test_build_walk_forward_summary_mean_std_and_n_steps():
    df = _step_df(
        [
            {"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1},
            {"step_i": 1, "auc": 0.8, "f1": 0.7, "fpr": 0.2},
            {"step_i": 2, "auc": 1.0, "f1": 0.9, "fpr": 0.0},
        ]
    )
    summary = rwf._build_walk_forward_summary(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["auc_mean"] == pytest.approx(0.9)
    assert row["f1_mean"] == pytest.approx(0.8)
    assert row["fpr_mean"] == pytest.approx(0.1)
    assert row["auc_std"] == pytest.approx(np.std([0.9, 0.8, 1.0], ddof=1))
    assert row["n_steps"] == 3


def test_build_walk_forward_summary_one_row_per_config():
    df = _step_df(
        [
            {"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1, "model": "logreg"},
            {"step_i": 1, "auc": 0.8, "f1": 0.7, "fpr": 0.2, "model": "logreg"},
            {"step_i": 0, "auc": 0.7, "f1": 0.6, "fpr": 0.3, "model": "rf"},
        ]
    )
    summary = rwf._build_walk_forward_summary(df)
    assert len(summary) == 2
    assert set(summary["model"]) == {"logreg", "rf"}


def test_build_walk_forward_summary_nan_std_for_single_step():
    df = _step_df([{"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1}])
    summary = rwf._build_walk_forward_summary(df)
    assert summary.iloc[0]["n_steps"] == 1
    assert np.isnan(summary.iloc[0]["auc_std"])


# ---- _run_one_config ----------------------------------------------------------


class _FakeModel:
    def __init__(self, proba_pos: np.ndarray):
        self._proba_pos = proba_pos

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba_pos, self._proba_pos])


def test_run_one_config_walks_every_step_and_skips_gracefully(monkeypatch):
    """n_total=100, gran=0.2 -> win_size=20, n_windows=5 -> steps i=0..3.
    Step 0: fit succeeds, target window has labeled rows -> real metrics.
    Step 1: fit succeeds, but target window (win_idx=2) has no labeled rows
        -> nan metrics, no crash.
    Step 2: fit_window itself returns None (e.g. single-class Wi) -> nan
        metrics, encode_target_window never called for this step.
    Step 3: fit succeeds, target window has labeled rows -> real metrics.
    Each step's fit is independent -- a skip at one step doesn't affect the
    next, since every step re-mines/retrains from scratch."""
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=0.2, model="logreg"
    )
    config = RollingWalkForwardConfig(
        scenario="test_scenario",
        shortlist_path=Path("unused.csv"),
    )

    def fake_fit_window(*, win_idx, **kwargs):
        if win_idx == 2:
            return None
        return rwf.WindowFit(
            schema=object(),
            model=_FakeModel(np.array([0.2, 0.8, 0.9])),
            threshold=0.5,
            feature_names=["f1"],
            cache_hit=True,
        )

    def fake_encode_target_window(alert_groups, n_total, gran, win_idx, schema):
        if win_idx == 2:
            return pd.DataFrame({"f1": []}), np.array([]), 0
        return pd.DataFrame({"f1": [0.1, 0.2, 0.3]}), np.array([0, 1, 1]), 3

    monkeypatch.setattr(rwf, "fit_window", fake_fit_window)
    monkeypatch.setattr(rwf, "encode_target_window", fake_encode_target_window)

    rows = rwf._run_one_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=[object()] * 100,
        alert_groups_path=Path("unused.json"),
        n_total=100,
        base_schema=object(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
    )

    assert [r["step_i"] for r in rows] == [0, 1, 2, 3]
    assert not np.isnan(rows[0]["auc"])  # step 0: normal
    assert np.isnan(rows[1]["auc"])  # step 1: empty target window
    assert np.isnan(rows[2]["auc"])  # step 2: fit_window returned None
    assert not np.isnan(rows[3]["auc"])  # step 3: normal

    # Every row reflects that step's own fresh fit -- no leakage of
    # mining_cache_hit/threshold from a neighboring step.
    assert rows[0]["mining_cache_hit"] is True
    assert rows[2]["mining_cache_hit"] is None
    assert np.isnan(rows[2]["threshold"])
    assert rows[1]["threshold"] == pytest.approx(0.5)
