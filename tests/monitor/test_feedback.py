import math

import numpy as np

from thesis.monitor.feedback import FeedbackCounts, feedback_counts, feedback_signal


def test_counts_only_escalated_labelled_groups():
    flagged = np.array([1, 1, 1, 0, 1], dtype=np.float32)
    labeled = np.array([1, 1, 0, 1, 1], dtype=np.float32)
    attack = np.array([1, 0, 1, 1, 1], dtype=np.float32)
    c = feedback_counts(flagged, labeled, attack)
    assert (c.n_escalated, c.n_true_attack) == (3, 2)
    assert math.isclose(c.precision, 2 / 3)


def test_clear_precision_drop_is_elevated():
    ref = FeedbackCounts(1500, 1350)  # 0.90
    out = feedback_signal(FeedbackCounts(1500, 1200), ref)  # 0.80
    assert out["fb_precision_elevated"] is True
    assert out["fb_precision_thr"] < 0.90


def test_noise_sized_drop_is_not_elevated():
    ref = FeedbackCounts(1500, 1350)
    out = feedback_signal(FeedbackCounts(1500, 1335), ref)  # 0.89
    assert out["fb_precision_elevated"] is False


def test_same_drop_on_small_window_is_not_elevated():
    ref = FeedbackCounts(1500, 1350)
    out = feedback_signal(FeedbackCounts(40, 34), ref)  # 0.85, n=40
    assert out["fb_precision_elevated"] is False


def test_missing_inputs_give_empty_columns():
    assert all(v is None for v in feedback_signal(None, FeedbackCounts(10, 5)).values())
    out = feedback_signal(FeedbackCounts(0, 0), FeedbackCounts(10, 5))
    assert out["fb_precision_elevated"] is None
