"""
Analyst-feedback precision (observe-only signal).

In a SOC, analysts review every alert group the system escalates, so the
labels of escalated groups exist anyway. Per window:

    precision = true attacks among escalated (score >= threshold) labelled
                groups / escalated labelled groups

compared with the same precision on the deployment's reference rows (W_src's
held-out split, or the warm-up window after a redeploy). Elevated when the
window's precision is significantly LOWER than the reference: a one-sided
two-proportion test at the 99% level,

    threshold = p_ref - Z * sqrt(p_ref (1 - p_ref) (1/n_window + 1/n_ref))

so only escalation counts and a fixed level set it -- no test-period tuning.
Unlike the label-free statistics, sampling noise matters here: a window has
only ~1-2k escalated groups on CSCAS.

Blind spot: missed attacks (recall) are never reviewed, so they don't show.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z_ONE_SIDED_99 = 2.326

FEEDBACK_COLUMNS = (
    "fb_precision",
    "fb_precision_ref",
    "fb_n_escalated",
    "fb_n_escalated_ref",
    "fb_precision_thr",
    "fb_precision_elevated",
)


@dataclass(frozen=True, slots=True)
class FeedbackCounts:
    n_escalated: int  # escalated AND labelled
    n_true_attack: int

    @property
    def precision(self) -> float:
        return self.n_true_attack / self.n_escalated if self.n_escalated else math.nan


def feedback_counts(
    flagged: np.ndarray | None, labeled: np.ndarray | None, is_attack: np.ndarray | None
) -> FeedbackCounts | None:
    if flagged is None or labeled is None or is_attack is None:
        return None
    esc = (flagged > 0) & (labeled > 0)
    return FeedbackCounts(int(esc.sum()), int((esc & (is_attack > 0)).sum()))


def feedback_signal(
    window: FeedbackCounts | None,
    ref: FeedbackCounts | None,
    z: float = Z_ONE_SIDED_99,
) -> dict:
    out: dict = dict.fromkeys(FEEDBACK_COLUMNS)
    if window is None or ref is None:
        return out
    out["fb_n_escalated"] = window.n_escalated
    out["fb_n_escalated_ref"] = ref.n_escalated
    out["fb_precision"] = window.precision if window.n_escalated else None
    out["fb_precision_ref"] = ref.precision if ref.n_escalated else None
    if not window.n_escalated or not ref.n_escalated:
        return out
    p = ref.precision
    se = math.sqrt(
        max(p * (1 - p), 1e-12) * (1 / window.n_escalated + 1 / ref.n_escalated)
    )
    thr = p - z * se
    out["fb_precision_thr"] = thr
    out["fb_precision_elevated"] = bool(window.precision < thr)
    return out
