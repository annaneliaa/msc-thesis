from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.experiments.instance_explain import (
    InstanceExplanation,
    explanations_to_long_dataframe,
    select_error_instances,
)


# ---- select_error_instances --------------------------------------------------


def test_select_error_instances_finds_false_positives():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.9, 0.6, 0.1, 0.8, 0.2])  # idx0,1 are FP; idx4 is FN
    threshold = 0.5

    fp = select_error_instances(y_true, proba, threshold, kind="fp", top_n=5)
    assert set(fp) == {0, 1}


def test_select_error_instances_finds_false_negatives():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.9, 0.6, 0.1, 0.8, 0.2])
    threshold = 0.5

    fn = select_error_instances(y_true, proba, threshold, kind="fn", top_n=5)
    assert set(fn) == {4}


def test_select_error_instances_ranks_by_confidence_descending():
    y_true = np.array([0, 0, 0])
    proba = np.array([0.55, 0.95, 0.7])  # all FP at threshold 0.5
    threshold = 0.5

    fp = select_error_instances(y_true, proba, threshold, kind="fp", top_n=3)
    # most confidently wrong (farthest above threshold) first
    assert fp == [1, 2, 0]


def test_select_error_instances_respects_top_n():
    y_true = np.array([0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.6])
    threshold = 0.5

    fp = select_error_instances(y_true, proba, threshold, kind="fp", top_n=2)
    assert len(fp) == 2
    assert fp == [0, 1]


def test_select_error_instances_both_combines_fp_and_fn():
    y_true = np.array([0, 1])
    proba = np.array([0.9, 0.1])  # idx0 FP, idx1 FN
    threshold = 0.5

    both = select_error_instances(y_true, proba, threshold, kind="both", top_n=5)
    assert set(both) == {0, 1}


def test_select_error_instances_no_errors_returns_empty():
    y_true = np.array([0, 1])
    proba = np.array([0.1, 0.9])
    threshold = 0.5
    assert select_error_instances(y_true, proba, threshold, kind="both", top_n=5) == []


# ---- explanations_to_long_dataframe -------------------------------------------


def test_explanations_to_long_dataframe_shape():
    explanations = [
        InstanceExplanation(
            horizon_window_index=3,
            row_index=7,
            error_kind="fp",
            y_true=0,
            proba=0.91,
            threshold=0.5,
            shap_importances={"a": 0.5, "b": -0.2},
            lime_importances={"a": 0.4},
            lime_fidelity=0.87,
            feature_values={"a": 1.2, "b": 3.4},
        )
    ]
    base_row = {"scenario": "cscas", "feature_set": "symbolic", "model": "logreg"}

    df = explanations_to_long_dataframe(explanations, base_row)

    assert len(df) == 3  # 2 shap rows + 1 lime row
    assert set(df["method"]) == {"shap", "lime"}
    assert (df["horizon_window_index"] == 3).all()
    assert (df["scenario"] == "cscas").all()
    shap_rows = df[df["method"] == "shap"].sort_values("rank")
    assert list(shap_rows["feature"]) == ["a", "b"]
    assert list(shap_rows["importance"]) == pytest.approx([0.5, -0.2])


def test_explanations_to_long_dataframe_empty_input():
    df = explanations_to_long_dataframe([], {"scenario": "cscas"})
    assert isinstance(df, pd.DataFrame)
    assert df.empty
