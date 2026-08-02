"""Training-pool construction for the alert_groups-based experiment pipeline
(experiments/baseline.py, experiments/symbolic.py via training/service.py).

Generalizes baselines/_sampling.py's three pool-construction strategies from
"operate on a raw CSV DataFrame with Label/SCAS columns" to "operate on
already-train-split (y_train, scas_train) arrays, return row indices /
ready-to-use imbalance kwargs". Semantics are ported as-is -- see
_sampling.py's module docstring for the original Baseline 1/2/reasoning.

guided_by_scas_pool is CSCAS-only (same reasoning as _sampling.py: AIT-ADS
has no SCAS-equivalent signal). Callers must not invoke it with scas=None.
"""

from __future__ import annotations

import numpy as np


def random_undersample_pool(y: np.ndarray, seed: int) -> np.ndarray:
    """Baseline 1: keep every positive row, randomly undersample the
    negative pool down to len(positive). Returns row indices into y."""
    rng = np.random.RandomState(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    neg_sample = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    return np.concatenate([pos_idx, neg_sample])


def guided_by_scas_pool(y: np.ndarray, scas: np.ndarray, seed: int) -> np.ndarray:
    """Baseline 2: guided by CSCAS's own SCAS outlier/inlier split
    (Imbalanced-0.5). CSCAS-only -- `scas` must not contain None/NaN.

    n_inliers = len(positive) // 2; n_outliers = len(positive) - n_inliers,
    matching _sampling.guided_by_cscas_pool exactly. If the training window
    is small enough that the outlier or inlier pool can't supply the
    requested count (comfortable at full-CSCAS scale -- 4,153 outliers vs.
    133,614 inliers -- not guaranteed in a narrow mined window), clamp to
    what's available and warn rather than raising.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.flatnonzero(y == 1)
    irr_inlier_idx = np.flatnonzero((y == 0) & (scas == 0))
    irr_outlier_idx = np.flatnonzero((y == 0) & (scas == 1))

    n_inliers = len(pos_idx) // 2
    n_outliers = len(pos_idx) - n_inliers

    if n_inliers > len(irr_inlier_idx):
        print(
            f"  [warn] guided pool: requested {n_inliers} SCAS-inlier rows, "
            f"only {len(irr_inlier_idx)} available -- clamping."
        )
        n_inliers = len(irr_inlier_idx)
    if n_outliers > len(irr_outlier_idx):
        print(
            f"  [warn] guided pool: requested {n_outliers} SCAS-outlier rows, "
            f"only {len(irr_outlier_idx)} available -- clamping."
        )
        n_outliers = len(irr_outlier_idx)

    inlier_sample = rng.choice(irr_inlier_idx, size=n_inliers, replace=False)
    outlier_sample = rng.choice(irr_outlier_idx, size=n_outliers, replace=False)
    return np.concatenate([pos_idx, inlier_sample, outlier_sample])


def class_weighted_extra_kwargs(y: np.ndarray) -> dict:
    """Third condition: natural-ratio pool (no resampling -- caller keeps
    every row), plus ready-to-use imbalance-handling kwargs for whichever
    model factory can use them (class_weight for sklearn, scale_pos_weight
    for XGBoost/torch_nn). Dataset-agnostic."""
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    return {
        "class_weight": "balanced",
        "scale_pos_weight": n_negative / n_positive,
    }
