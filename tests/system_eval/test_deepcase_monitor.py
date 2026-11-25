from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.system_eval.deepcase_monitor import (
    label_weights,
    monitor_seed,
    run_deepcase_monitor,
    summarise,
)


def _window(h, n=4000, broken=False, noisy_fp=False, seed=0):
    """Cluster 0 = benign cluster (code 0), cluster 1 = attack cluster
    (code 1), 5% manual queue (code -1). `broken`: 40% of cluster 0's
    events are attacks (attacks hiding in a benign cluster)."""
    rng = np.random.default_rng(seed + h)
    cluster = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    code = cluster.astype(float)
    manual = rng.random(n) < 0.05
    code[manual] = -1
    cluster = np.where(manual, -1, cluster)
    label = (code == 1).astype(int)
    label[manual] = (rng.random(manual.sum()) < 0.3).astype(int)
    if broken:
        label[(code == 0) & (rng.random(n) < 0.4)] = 1
    if noisy_fp:
        label[(code == 1) & (rng.random(n) < 0.5)] = 0
    return pd.DataFrame(
        {
            "group_id": [f"h{h}_{i}" for i in range(n)],
            "horizon": h,
            "label": label,
            "code": code,
            "cluster": cluster,
        }
    )


def test_label_weights_audit_and_oracle():
    code = np.array([1, -1, 0, 0])
    audited = np.array([False, False, True, False])
    assert label_weights(code, audited, "audit", 0.05).tolist() == [1, 1, 20, 0]
    assert label_weights(code, audited, "oracle", 0.05).tolist() == [1, 1, 1, 1]


def test_clean_windows_stay_quiet_and_reference_is_window_one():
    # (A 99% test still fires on ~1% of clean windows; seed 100 is not one.)
    rows = pd.concat([_window(h, seed=100) for h in (1, 2, 3)], ignore_index=True)
    sig = monitor_seed(rows, audit_rate=0.2).set_index("horizon")
    assert sig.loc[1, "is_reference"] and not sig.loc[2, "is_reference"]
    assert not sig.loc[[2, 3], "pv_audit_elevated"].any()
    assert not sig.loc[[2, 3], "fb_precision_elevated"].fillna(False).any()
    assert (sig.loc[[2, 3], "combined_action"] == "NO_ACTION").all()
    assert sig.loc[2, "n_audit_labels"] > 0


def test_attacks_in_benign_cluster_trigger_pattern_validity():
    rows = pd.concat(
        [_window(1), _window(2), _window(3, broken=True)], ignore_index=True
    )
    sig = monitor_seed(rows, audit_rate=0.2).set_index("horizon")
    for mode in ("oracle", "audit"):
        assert sig.loc[3, f"pv_{mode}_precision_elevated"]  # cluster 0 now wrong
        assert sig.loc[3, f"pv_{mode}_coverage_elevated"]  # attacks auto-benign
        assert "0" in sig.loc[3, f"pv_{mode}_cluster_prec_drop_ids"].split(";")
    assert sig.loc[3, "combined_action"] == "REMINE_AND_RETRAIN"


def test_false_alarms_in_attack_cluster_trigger_feedback():
    rows = pd.concat(
        [_window(1), _window(2), _window(3, noisy_fp=True)], ignore_index=True
    )
    sig = monitor_seed(rows, audit_rate=0.2).set_index("horizon")
    assert sig.loc[3, "fb_precision_elevated"]
    assert sig.loc[3, "auto_fb_precision_elevated"]


def test_without_cluster_column_precision_part_is_skipped(tmp_path):
    rows = pd.concat([_window(h) for h in (1, 2)], ignore_index=True)
    rows = rows.drop(columns="cluster")
    sig = monitor_seed(rows, audit_rate=0.2)
    assert (sig["pv_audit_n_clusters_checked"].dropna() == 0).all()


def test_run_and_summary(tmp_path):
    run = tmp_path / "dc"
    (run / "seed0").mkdir(parents=True)
    pd.concat([_window(h) for h in (1, 2, 3)], ignore_index=True).to_parquet(
        run / "seed0" / "test_predictions.parquet"
    )
    pd.DataFrame(
        [
            {
                "seed": 0,
                "horizon": h,
                "subset": s,
                "e2e_f1": 0.5 + 0.1 * (h == 2),
                "e2e_precision": 0.5,
                "e2e_recall": 1.0,
                "coverage": 0.9,
            }
            for h in ("1", "2", "3", "all")
            for s in ("all", "seen")
        ]
    ).to_csv(run / "horizon_metrics.csv", index=False)
    sig = run_deepcase_monitor(run, tmp_path / "out", audit_rate=0.2)
    assert (tmp_path / "out" / "per_window_signals.csv").exists()
    assert sig.e2e_f1.notna().all()
    text = summarise(sig)
    assert "Firing rate" in text and "pattern validity analogue, audit" in text


def _native_window(h, n=5000, extra_code=None, extra_share=0.0, seed=0):
    """Codes: 80% auto, 5% each of -1/-2/-3 (+ an optional extra share of
    one reason), spread evenly over 5 days."""
    rng = np.random.default_rng(seed + h)
    code = rng.choice([0, 1, -1, -2, -3], size=n, p=[0.7, 0.15, 0.05, 0.05, 0.05])
    if extra_code is not None:
        code[rng.random(n) < extra_share] = extra_code
    return pd.DataFrame(
        {
            "group_id": [f"n{h}_{i}" for i in range(n)],
            "horizon": h,
            "ts": h * 10 * 86400 + np.sort(rng.integers(0, 5 * 86400, n)),
            "label": (code == 1).astype(int),
            "code": code.astype(float),
        }
    )


def test_native_signals_quiet_on_clean_windows():
    rows = pd.concat([_native_window(h) for h in (1, 2, 3)], ignore_index=True)
    sig = monitor_seed(rows, audit_rate=0.2).set_index("horizon")
    assert not sig.loc[[2, 3], "native_elevated"].any()
    assert (sig.loc[[2, 3], "native_action"] == "NO_ACTION").all()
    assert sig.loc[2, "native_unknown_event_thr"] >= 0.01


def test_native_unknown_events_remine_too_far_retrains():
    rows = pd.concat(
        [
            _native_window(1),
            _native_window(2, extra_code=-2, extra_share=0.15),
            _native_window(3, extra_code=-3, extra_share=0.15),
        ],
        ignore_index=True,
    )
    sig = monitor_seed(rows, audit_rate=0.2).set_index("horizon")
    assert sig.loc[2, "native_unknown_event_elevated"]
    assert sig.loc[2, "native_action"] == "REMINE_AND_RETRAIN"
    assert sig.loc[3, "native_too_far_elevated"]
    assert not sig.loc[3, "native_unknown_event_elevated"]
    assert sig.loc[3, "native_action"] == "RETRAIN_ONLY"
