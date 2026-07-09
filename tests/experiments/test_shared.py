from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.experiments._shared import (
    decide_threshold,
    metrics_at_threshold,
    nan_metrics,
    sample_rows,
)


# ---- metrics_at_threshold ---------------------------------------------------


def test_metrics_at_threshold_uses_the_given_threshold_not_0_5():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.35, 0.4, 0.6, 0.9])

    at_0_5 = metrics_at_threshold(y_true, proba, 0.5)
    at_0_3 = metrics_at_threshold(y_true, proba, 0.3)

    # At 0.5: pred = [0,0,0,0,1,1] -> tp=2, fn=1, fp=0, tn=3
    assert (at_0_5["tp"], at_0_5["fp"], at_0_5["tn"], at_0_5["fn"]) == (2, 0, 3, 1)
    # At 0.3: pred = [0,0,1,1,1,1] -> tp=3, fn=0, fp=1, tn=2
    assert (at_0_3["tp"], at_0_3["fp"], at_0_3["tn"], at_0_3["fn"]) == (3, 1, 2, 0)
    assert at_0_3["recall"] > at_0_5["recall"]
    assert at_0_3["fpr"] > at_0_5["fpr"]
    # AUC is threshold-independent -- identical at both thresholds.
    assert at_0_5["auc"] == pytest.approx(at_0_3["auc"])


def test_metrics_at_threshold_single_class_gives_nan_auc():
    y_true = np.array([0, 0, 0])
    proba = np.array([0.1, 0.2, 0.9])
    result = metrics_at_threshold(y_true, proba, 0.5)
    assert np.isnan(result["auc"])


# ---- decide_threshold --------------------------------------------------------


def test_decide_threshold_fixed_mode_is_always_0_5():
    y_src = np.array([0, 1, 0, 1])
    proba_src = np.array([0.2, 0.8, 0.3, 0.9])
    assert decide_threshold(y_src, proba_src, "fixed", 0.90) == 0.5


def test_decide_threshold_calibrated_recall_matches_workload_helper():
    from thesis.training.workload import compute_workload_at_recall

    rng = np.random.default_rng(0)
    y_src = np.array([0] * 50 + [1] * 50)
    proba_src = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])

    threshold = decide_threshold(y_src, proba_src, "calibrated_recall", 0.90)
    expected = compute_workload_at_recall(y_src, proba_src, targets=(0.90,))["0.90"][
        "threshold"
    ]
    assert threshold == pytest.approx(expected)


def test_decide_threshold_calibrated_recall_falls_back_on_single_class():
    y_src = np.array([0, 0, 0, 0])
    proba_src = np.array([0.1, 0.2, 0.3, 0.4])
    assert decide_threshold(y_src, proba_src, "calibrated_recall", 0.90) == 0.5


# ---- nan_metrics / sample_rows ----------------------------------------------


def test_nan_metrics_has_the_same_keys_as_metrics_at_threshold():
    y_true = np.array([0, 1])
    proba = np.array([0.2, 0.8])
    real = metrics_at_threshold(y_true, proba, 0.5)
    assert set(nan_metrics().keys()) == set(real.keys())
    assert np.isnan(nan_metrics()["auc"])


def test_sample_rows_caps_at_available_rows():
    X = pd.DataFrame({"a": range(5)})
    assert len(sample_rows(X, 100, random_state=0)) == 5
    assert len(sample_rows(X, 3, random_state=0)) == 3


def test_sample_rows_empty_input_stays_empty():
    X = pd.DataFrame({"a": []})
    assert sample_rows(X, 10, random_state=0).empty
