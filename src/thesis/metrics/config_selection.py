"""Top-k config selection from screening-sweep results.

Turns per-window screening results (one row per window x config evaluation,
e.g. `artifacts/experiments/screening_sweep/<scenario>/<run>/per_window_results.csv`)
into a ranked shortlist of configs.

Selection rule (see e.g. src/thesis/notebooks/config_selection.ipynb for the
calling convention -- thresholds like `lam`, `k`, and the floor threshold are
set by the caller, not hardcoded here):

1. `summarize_configs` -- collapse per-window rows into one row per config
   (a `group_cols` combination, e.g. feature_set x mining_setting x
   granularity), with the mean/std/min/p10 of `score_col` across windows.
2. `rank_configs` -- rank configs by `score_mean - lam * score_std`: mean LR
   score as the primary criterion, penalized by cross-window instability
   rather than treating stability as a separate filter.
3. `apply_floor_check` -- a sanity check applied to the ranked/shortlisted
   configs, *not* part of the ranking formula: flags configs whose worst
   window falls below a threshold, even if their mean score is strong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GROUP_COLS = ["feature_set", "mining_setting", "granularity", "model"]


def summarize_configs(
    df: pd.DataFrame,
    score_col: str,
    group_cols: list[str] = DEFAULT_GROUP_COLS,
    n_features_col: str | None = "n_features",
) -> pd.DataFrame:
    """Aggregate per-window scores into one row per config (`group_cols`).

    Only `group_cols` present in `df` are used, so the same call works for
    scenarios with a single model or a single granularity. Reports
    score_mean/std/min/p10/max/count for `score_col`, plus the mean of
    `n_features_col` (assumed constant within a config -- reporting the mean
    surfaces it if that assumption breaks).
    """
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        raise ValueError(f"none of {DEFAULT_GROUP_COLS} found in df columns")

    grouped = df.groupby(group_cols, dropna=False)[score_col]
    summary = grouped.agg(
        score_mean="mean",
        score_std="std",
        score_min="min",
        score_max="max",
        n_windows="count",
    ).reset_index()
    summary["score_std"] = summary["score_std"].fillna(0.0)
    summary["score_cv"] = np.where(
        summary["score_mean"] != 0,
        summary["score_std"] / summary["score_mean"],
        np.nan,
    )
    summary["score_p10"] = grouped.quantile(0.1).to_numpy()

    if n_features_col and n_features_col in df.columns:
        n_feat = (
            df.groupby(group_cols, dropna=False)[n_features_col]
            .mean()
            .rename("n_features_mean")
            .reset_index()
        )
        summary = summary.merge(n_feat, on=group_cols)

    return summary


def rank_configs(
    summary: pd.DataFrame,
    lam: float = 0.5,
    score_mean_col: str = "score_mean",
    score_std_col: str = "score_std",
    tiebreak_col: str | None = "n_features_mean",
    tiebreak_tol: float = 0.0,
) -> pd.DataFrame:
    """Rank configs by `score_mean - lam * score_std`, descending.

    `tiebreak_col` / `tiebreak_tol`: configs are grouped into "tie bands" by
    rounding `selection_score` to the nearest `tiebreak_tol`, then each band
    is sorted by ascending `tiebreak_col` (fewer features wins) instead of by
    the sub-tolerance score difference. `tiebreak_tol=0.0` (default) disables
    this and ranks purely by `selection_score`.
    """
    ranked = summary.copy()
    ranked["selection_score"] = ranked[score_mean_col] - lam * ranked[score_std_col]

    if tiebreak_col and tiebreak_tol > 0 and tiebreak_col in ranked.columns:
        tie_band = np.round(ranked["selection_score"] / tiebreak_tol)
        ranked = (
            ranked.assign(_tie_band=tie_band)
            .sort_values(["_tie_band", tiebreak_col], ascending=[False, True])
            .drop(columns="_tie_band")
        )
    else:
        ranked = ranked.sort_values("selection_score", ascending=False)

    return ranked.reset_index(drop=True)


def apply_floor_check(
    ranked: pd.DataFrame,
    floor_threshold: float,
    floor_col: str = "score_min",
) -> pd.DataFrame:
    """Flag configs whose floor performance falls below `floor_threshold`.

    `floor_col` is typically `score_min` (strictest) or `score_p10` (from
    `summarize_configs`). Adds a `passes_floor` bool column; does not drop
    rows -- a config failing this check should be inspected, not silently
    removed.
    """
    out = ranked.copy()
    out["passes_floor"] = out[floor_col] >= floor_threshold
    return out


def select_top_k(ranked: pd.DataFrame, k: int) -> pd.DataFrame:
    """Take the top-k configs from an already-ranked frame (see `rank_configs`).

    Run `apply_floor_check` on the result to sanity-check survivors -- floor
    performance is a check on the shortlist, not a filter baked into ranking.
    """
    return ranked.head(k).reset_index(drop=True)
