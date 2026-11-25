from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.system_eval.deepcase_frozen_comparison import (
    find_our_runs,
    matched_workload_escalation,
    run_frozen_comparison,
    workload_curve,
)


def _fixture(tmp_path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.1).astype(int)
    rows = pd.DataFrame(
        {
            "group_id": [f"g{i}" for i in range(n)],
            "ts": 1_000_000 + np.arange(n) * 864,  # n rows over n/100 days
            "horizon": 1 + np.arange(n) * 2 // n,  # two horizons
            "label": label,
            "signature_id": rng.integers(0, 20, n),
            "seen": rng.random(n) < 0.8,
            "code": np.where(label == 1, 1, 0),
        }
    )
    dc = tmp_path / "deepcase"
    (dc / "seed0").mkdir(parents=True)
    rows.to_parquet(dc / "seed0" / "test_predictions.parquet")
    dc_metrics = []
    for h in ("all", 1, 2):
        for subset in ("all", "seen", "unseen"):
            dc_metrics.append(
                {
                    "seed": 0,
                    "horizon": h,
                    "subset": subset,
                    "e2e_f1": 0.2,
                    "e2e_precision": 0.1,
                    "e2e_recall": 1.0,
                    "missed_attacks": 0,
                    "analyst_items_per_day": 50.0,
                    "fp_per_day": 45.0,
                    "workload_reduction": 0.5,
                    "coverage": 0.8,
                    "n": n,
                    "n_attack": int(label.sum()),
                    "analyst_items": 120 if subset == "all" else 60,
                }
            )
    pd.DataFrame(dc_metrics).to_csv(dc / "horizon_metrics.csv", index=False)
    pd.DataFrame(
        {"seed": [0], "budget_b0": [100], "budget_clusters_only": [10]}
    ).to_csv(dc / "budget.csv", index=False)

    # Ours: a decent scorer; older and newer run, newer must be picked.
    score = np.clip(label * 0.7 + rng.random(n) * 0.5, 0, 1)
    preds = pd.DataFrame(
        {
            "model": "xgboost",
            "horizon_window_index": rows["horizon"],
            "group_id": rows["group_id"],
            "label": label.astype(float),
            "score": score,
            "escalated": score >= 0.5,
        }
    )
    ours = tmp_path / "ours"
    for ts, p in (
        ("20260101_000000", preds.assign(score=0.0)),
        ("20260102_000000", preds),
    ):
        d = ours / "stratified_b0" / "seed0" / "cscas" / ts
        d.mkdir(parents=True)
        p.to_parquet(d / "predictions.parquet")
    return dc, ours, rows, preds


def test_find_our_runs_picks_newest(tmp_path):
    _, ours, _, _ = _fixture(tmp_path)
    runs = find_our_runs(ours)
    assert [(r.variant, r.seed) for r in runs] == [("stratified_b0", 0)]
    assert "20260102_000000" in str(runs[0].path)


def test_matched_workload_and_curve():
    score = np.array([0.9, 0.1, 0.5, 0.7])
    assert matched_workload_escalation(score, 2).tolist() == [True, False, False, True]
    label = np.array([1, 0, 0, 1])
    c = workload_curve(label, score, days=1.0, items_per_day=[0, 1, 2, 4])
    assert c.recall.tolist() == [0.0, 0.5, 1.0, 1.0]
    assert c.recall.is_monotonic_increasing


def test_run_frozen_comparison_matches_hand_computed(tmp_path):
    dc, ours, rows, preds = _fixture(tmp_path)
    res = run_frozen_comparison(dc, ours, tmp_path / "out")
    ps = res["per_seed"]
    o = ps[(ps.system == "ours") & (ps.subset == "all")].set_index("operating_point")

    label = preds.label.to_numpy().astype(bool)
    esc = preds.escalated.to_numpy()
    tp, fp = (esc & label).sum(), (esc & ~label).sum()
    assert o.loc["native", "e2e_recall"] == pytest.approx(tp / label.sum())
    assert o.loc["native", "e2e_precision"] == pytest.approx(tp / (tp + fp))
    assert o.loc["native", "coverage"] == 1.0
    days = (rows.ts.max() - rows.ts.min()) / 86400
    assert o.loc["native", "analyst_items_per_day"] == pytest.approx(esc.sum() / days)
    # Matched workload escalates exactly DeepCASE's 120 analyst items.
    assert o.loc["matched_workload", "analyst_items_per_day"] == pytest.approx(
        120 / days
    )

    d = ps[(ps.system == "deepcase") & (ps.subset == "all")]
    assert d.e2e_recall.iloc[0] == 1.0 and d.budget_b0.iloc[0] == 100
    t = res["table"]
    assert set(t.system) == {"ours", "deepcase"}
    assert {"e2e_f1_mean", "e2e_f1_std", "n_seeds"} <= set(t.columns)
    ph = res["per_horizon"]
    assert set(ph[ph.system == "ours"].horizon) == {1, 2}
    for f in ("per_seed.csv", "table.csv", "workload_curve.csv", "per_horizon.csv"):
        assert (tmp_path / "out" / f).exists()


def test_missing_rows_raise(tmp_path):
    dc, ours, _, preds = _fixture(tmp_path)
    p = ours / "stratified_b0" / "seed0" / "cscas" / "20260102_000000"
    preds.iloc[:-5].to_parquet(p / "predictions.parquet")
    with pytest.raises(ValueError, match="missing from ours"):
        run_frozen_comparison(dc, ours, tmp_path / "out")
