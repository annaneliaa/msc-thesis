"""Human-workload / alert-reduction metrics for a fixed recall target.

Given a continuous attack score and the true labels, finds — for each target
recall — the highest-precision threshold that still guarantees at least that
recall, then reports the resulting confusion matrix and the fraction of
alert-groups an analyst would no longer need to manually review at that
operating point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

DEFAULT_RECALL_TARGETS: tuple[float, ...] = (0.90, 0.95, 0.99)


def compute_workload_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    targets: tuple[float, ...] = DEFAULT_RECALL_TARGETS,
) -> dict[str, dict | None]:
    """For each target recall, report precision/FPs/workload reduction at the
    highest threshold that still achieves that recall.

    workload_reduction = (tn + fn) / total: the fraction of alert-groups not
    flagged for review, i.e. the volume an analyst is spared from inspecting
    while the model still guarantees the target recall on true attacks.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    result: dict[str, dict | None] = {}

    if len(np.unique(y_true)) < 2 or (y_true == 1).sum() == 0:
        return {f"{r:.2f}": None for r in targets}

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, scores)

    for r in targets:
        key = f"{r:.2f}"
        valid = np.where(recall_curve[:-1] >= r - 1e-9)[0]
        if valid.size == 0:
            result[key] = None
            continue
        threshold = float(thresholds[valid.max()])

        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = tn + fp + fn + tp

        result[key] = {
            "threshold": threshold,
            "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "workload_reduction": float((tn + fn) / total) if total else float("nan"),
            "fp_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        }

    return result
