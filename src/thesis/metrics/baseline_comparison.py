"""Pairs symbolic screening-sweep rows with their baseline counterpart.

`per_window_results.csv` (see `thesis.experiments.screening_sweep`) contains,
for each (scenario, granularity, window_id, model), one `feature_set="baseline"`
row (no symbolic features) and one `feature_set="symbolic"` row per mining
setting -- all evaluated on the exact same window's train/test split. Because
the split is identical, a symbolic row can be compared directly against its
baseline row without any renormalization.

`pair_with_baseline` does that join and computes per-window deltas;
`summarize_comparison` collapses the pairs into one row per symbolic config
(mirroring `config_selection.summarize_configs`'s grouping), for feeding into
a table or a plot next to the ranked-config comparison.
"""

from __future__ import annotations

import pandas as pd

JOIN_COLS = ["scenario", "granularity", "window_id", "model"]
METRIC_COLS = ["auc", "precision", "recall", "tp", "fp", "tn", "fn"]
CONFIG_COLS = ["mining_setting", "granularity", "model"]


def pair_with_baseline(
    per_window: pd.DataFrame,
    join_cols: list[str] = JOIN_COLS,
    metric_cols: list[str] = METRIC_COLS,
) -> pd.DataFrame:
    """Inner-join each symbolic row to the baseline row from the same window.

    Only `join_cols` / `metric_cols` present in `per_window` are used (mirrors
    `summarize_configs`'s handling of missing columns, so the same call works
    across scenarios with e.g. a single model). Baseline metric columns are
    suffixed `_base`, symbolic ones `_sym`; all other symbolic-side columns
    (`mining_setting`, `feature_set`, window context, ...) are kept unsuffixed.

    Adds delta columns: `auc_delta`, `precision_delta`, `recall_delta`
    (symbolic - baseline), `fp_reduction` / `fn_increase` (count deltas,
    positive = symbolic removed FPs / symbolic missed more attacks
    respectively), and `fp_reduction_pct` (`fp_reduction` relative to
    baseline FP count; NaN where baseline had zero FPs).
    """
    join_cols = [c for c in join_cols if c in per_window.columns]
    metric_cols = [c for c in metric_cols if c in per_window.columns]

    baseline = per_window[per_window["feature_set"] == "baseline"]
    symbolic = per_window[per_window["feature_set"] == "symbolic"]

    baseline_metrics = baseline[join_cols + metric_cols].rename(
        columns={c: f"{c}_base" for c in metric_cols}
    )
    symbolic_metrics = symbolic.rename(columns={c: f"{c}_sym" for c in metric_cols})

    paired = symbolic_metrics.merge(baseline_metrics, on=join_cols, how="inner")

    if "auc_sym" in paired.columns and "auc_base" in paired.columns:
        paired["auc_delta"] = paired["auc_sym"] - paired["auc_base"]
    if "precision_sym" in paired.columns and "precision_base" in paired.columns:
        paired["precision_delta"] = paired["precision_sym"] - paired["precision_base"]
    if "recall_sym" in paired.columns and "recall_base" in paired.columns:
        paired["recall_delta"] = paired["recall_sym"] - paired["recall_base"]
    if "fp_sym" in paired.columns and "fp_base" in paired.columns:
        paired["fp_reduction"] = paired["fp_base"] - paired["fp_sym"]
        paired["fp_reduction_pct"] = paired["fp_reduction"] / paired["fp_base"].replace(
            0, pd.NA
        )
    if "fn_sym" in paired.columns and "fn_base" in paired.columns:
        paired["fn_increase"] = paired["fn_sym"] - paired["fn_base"]

    return paired


def summarize_comparison(
    paired: pd.DataFrame,
    group_cols: list[str] = CONFIG_COLS,
) -> pd.DataFrame:
    """Collapse per-window paired deltas into one row per symbolic config."""
    group_cols = [c for c in group_cols if c in paired.columns]
    if not group_cols:
        raise ValueError(f"none of {CONFIG_COLS} found in paired columns")

    delta_cols = [
        c
        for c in [
            "auc_delta",
            "precision_delta",
            "recall_delta",
            "fp_reduction",
            "fp_reduction_pct",
            "fn_increase",
        ]
        if c in paired.columns
    ]

    grouped = paired.groupby(group_cols, dropna=False)
    summary = (
        grouped[delta_cols]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in delta_cols})
        .reset_index()
    )
    summary.insert(len(group_cols), "n_windows", grouped.size().reset_index(drop=True))
    return summary
