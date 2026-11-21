"""
Paired bootstrap over the test groups, for comparisons with too few steps for
a test across steps (AIT-ADS: 5 scenarios, so a Wilcoxon over scenarios can't
go below p = 0.0625 even when one method wins all 5).

Input: the per-group test scores a run saves (test_scores.csv.gz: step_i,
scenario, arm, seed, test_idx, y, proba, pred). Per step (scenario) and pair
of arms (a, b):

  1. resample the step's test groups with replacement, attacks and benign
     separately (stratified: every resample keeps the step's attack count,
     so PR-AUC is always defined);
  2. on that resample, each arm's metric is the mean over its seeds (each
     seed's own model and threshold), as in the summary;
  3. Δ = metric(a) - metric(b), the same resample for both arms (paired).

Output per (scenario, metric, a, b): the point Δ on the full test set, the
2.5 / 97.5 percentiles of the resampled Δ (95% CI), and a two-sided
bootstrap p (2 x the smaller share of resamples on either side of 0). The
"all" rows average Δ over the scenarios within each resample (scenarios
fixed, test groups random): one CI for the mean gain across scenarios.

The p values are per comparison, not corrected for the number of pairs.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("pr_auc", "f1")
ALL = "all"


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """sklearn's average_precision_score (step-wise, ties as one threshold),
    vectorised for the many bootstrap calls. NaN without positives."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], y[order]
    tp = np.cumsum(t)
    # one point per distinct score: the last index of each tie block
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = tp[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int((pred & (y == 1)).sum())
    denom = int(pred.sum()) + int(y.sum())
    return 2 * tp / denom if denom else 0.0


def _metric(metric: str, y: np.ndarray, proba: np.ndarray, pred: np.ndarray):
    if metric == "pr_auc":
        return average_precision(y, proba)
    if metric == "f1":
        return f1(y, pred)
    raise ValueError(f"unknown metric {metric!r}")


def _step_arrays(scores: pd.DataFrame):
    """One step's scores -> y (n,), and per arm: proba / pred (seeds, n)."""
    y = None
    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, g in scores.groupby("arm", sort=False):
        piv_p = g.pivot(index="test_idx", columns="seed", values="proba").sort_index()
        piv_d = g.pivot(index="test_idx", columns="seed", values="pred").sort_index()
        arms[arm] = (
            piv_p.to_numpy(dtype=float).T,
            piv_d.to_numpy(dtype=bool).T,
        )
        if y is None:
            y = (
                g.drop_duplicates("test_idx")
                .set_index("test_idx")
                .sort_index()["y"]
                .to_numpy(dtype=int)
            )
    return y, arms


def _arm_metric(metric, y, proba, pred, idx=None) -> float:
    if idx is not None:
        y, proba, pred = y[idx], proba[:, idx], pred[:, idx]
    return float(np.nanmean([_metric(metric, y, p, d) for p, d in zip(proba, pred)]))


def paired_bootstrap(
    scores: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    metrics=METRICS,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Per (scenario, metric, arm a, arm b): Δ = a - b, 95% CI, bootstrap p
    (see the module docstring). `pairs` defaults to every pair of arms."""
    rng = np.random.default_rng(seed)
    steps = sorted(scores["step_i"].unique())
    arms_all = list(dict.fromkeys(scores["arm"]))
    pairs = pairs or list(combinations(arms_all, 2))
    point: dict = {}  # (step, metric, arm) -> metric on the full test set
    boot: dict = {}  # (step, metric, arm) -> (n_boot,) resampled metric
    scenario_of = {}
    for step in steps:
        sc = scores[scores["step_i"] == step]
        scenario_of[step] = sc["scenario"].iloc[0]
        y, arms = _step_arrays(sc)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0:
            continue  # no test attacks: nothing to compare
        draws = [
            np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            for _ in range(n_boot)
        ]
        for metric in metrics:
            for arm, (proba, pred) in arms.items():
                point[(step, metric, arm)] = _arm_metric(metric, y, proba, pred)
                boot[(step, metric, arm)] = np.array(
                    [_arm_metric(metric, y, proba, pred, idx) for idx in draws]
                )

    rows = []
    for metric in metrics:
        for a, b in pairs:
            per_step_d, per_step_p = [], []
            for step in steps:
                ka, kb = (step, metric, a), (step, metric, b)
                if ka not in boot or kb not in boot:
                    continue
                d = boot[ka] - boot[kb]
                per_step_d.append(d)
                per_step_p.append(point[ka] - point[kb])
                rows.append(
                    _row(
                        scenario_of[step], step, metric, a, b, point[ka] - point[kb], d
                    )
                )
            if per_step_d:
                rows.append(
                    _row(
                        ALL,
                        np.nan,
                        metric,
                        a,
                        b,
                        float(np.mean(per_step_p)),
                        np.mean(per_step_d, axis=0),
                        n_steps=len(per_step_d),
                    )
                )
    return pd.DataFrame(rows)


def _row(scenario, step, metric, a, b, delta, d: np.ndarray, n_steps: int = 1):
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
    return {
        "scenario": scenario,
        "step_i": step,
        "metric": metric,
        "a": a,
        "b": b,
        "n_steps": n_steps,
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "p_boot": p,
        "significant": bool(len(d)) and (lo > 0 or hi < 0),
        "n_boot": len(d),
    }


def write_bootstrap(
    out_dir: Path, scores: pd.DataFrame, n_boot: int = 1000
) -> pd.DataFrame:
    out = paired_bootstrap(scores, n_boot=n_boot)
    out.to_csv(Path(out_dir) / "bootstrap.csv", index=False)
    return out
