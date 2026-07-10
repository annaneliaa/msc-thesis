"""Parameter-importance (sensitivity) analysis for screening-sweep configs.

`config_selection.py` picks a top-k shortlist; this module answers a
different question about the *same* `summarize_configs`/`rank_configs`
output: of the axes the sweep varied (feature_set, mining_setting's
sub-thresholds, granularity, model, ...), which ones actually move the
score, and which ones are noise? That's the evidence for a thesis claim like
"we fix max_depth and only vary granularity in Experiments 2-4" -- it needs
to show the fixed axis had small effect and the varied one didn't.

`expand_mining_setting` first unpacks the opaque `mining_setting` name
(e.g. `"gr3.0_md5"`) into its numeric sub-parameters (`min_growth_rate`,
`max_depth`) so they can be screened individually rather than as one
combined categorical axis.

`parameter_importance` then runs a one-way ANOVA of `score_col` against each
candidate parameter in turn (marginalizing over all other parameters, one
row per config as the unit of observation -- the same rows `config_selection`
ranks). `eta_squared` (SS_between / SS_total) is the effect-size to lead
with: it's on a 0-1 scale and comparable across parameters regardless of how
many levels each has, unlike the F-statistic. `main_effects` produces the
long-form table (parameter, level, mean, std, n) behind a main-effects plot.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

MINING_SETTING_PATTERN = re.compile(
    r"gr(?P<min_growth_rate>[\d.]+)_md(?P<max_depth>\d+)"
)


def parse_mining_setting(name: str | None) -> dict[str, float | None]:
    """Parse a `"gr{min_growth_rate}_md{max_depth}"` name into its sub-thresholds.

    Returns `{"min_growth_rate": None, "max_depth": None}` for anything that
    doesn't match (e.g. the baseline feature_set's `mining_setting` is NaN)
    rather than raising, so callers can `.apply` this over a mixed column.
    """
    if not isinstance(name, str):
        return {"min_growth_rate": None, "max_depth": None}
    m = MINING_SETTING_PATTERN.match(name)
    if not m:
        return {"min_growth_rate": None, "max_depth": None}
    return {
        "min_growth_rate": float(m.group("min_growth_rate")),
        "max_depth": int(m.group("max_depth")),
    }


def expand_mining_setting(
    df: pd.DataFrame, col: str = "mining_setting"
) -> pd.DataFrame:
    """Add `min_growth_rate` / `max_depth` columns parsed from `col`.

    Lets those two thresholds be screened as individual parameters instead
    of only as the bundled `mining_setting` categorical -- a parameter
    importance table over the bundle can only ever say "mining_setting
    matters", not which of its two axes is doing the work.
    """
    parsed = pd.DataFrame(list(df[col].apply(parse_mining_setting)), index=df.index)
    out = df.copy()
    out["min_growth_rate"] = parsed["min_growth_rate"]
    out["max_depth"] = parsed["max_depth"]
    return out


def parameter_importance(
    df: pd.DataFrame,
    score_col: str,
    param_cols: list[str],
) -> pd.DataFrame:
    """Rank `param_cols` by how much of the spread in `score_col` each explains.

    For each parameter: drop rows where it's null, group the rest by its
    distinct levels, and run a one-way ANOVA of `score_col` across those
    groups (`scipy.stats.f_oneway`). Reports `eta_squared` (SS_between /
    SS_total -- lead with this: it's 0-1 and comparable across parameters
    with different numbers of levels, unlike the raw F-statistic), the
    F-stat/p-value, `n_levels`, `n_configs`, and `effect_range` (max level
    mean minus min level mean, in `score_col`'s own units -- the intuitive
    "how much does this knob move the needle" number to quote alongside
    eta_squared).

    A parameter with a single level in `df` (e.g. `model` if only one was
    swept) can't be tested -- it gets `NaN` stats and `n_levels=1` rather
    than being dropped, so it still shows up as "not screened here".

    Rows are one-per-config (e.g. `config_selection.summarize_configs`
    output), so this asks "does varying this parameter, marginalized over
    every other parameter, move the mean score enough to matter" -- not a
    population-level claim about `score_col`'s per-window noise.
    """
    rows = []
    for param in param_cols:
        if param not in df.columns:
            continue
        sub = df.dropna(subset=[param, score_col])
        levels = sub[param].unique()
        groups = [
            sub.loc[sub[param] == level, score_col].to_numpy() for level in levels
        ]
        group_means = np.array([g.mean() for g in groups])

        if len(levels) < 2:
            rows.append(
                {
                    "parameter": param,
                    "n_levels": len(levels),
                    "n_configs": len(sub),
                    "eta_squared": np.nan,
                    "f_stat": np.nan,
                    "p_value": np.nan,
                    "effect_range": np.nan,
                }
            )
            continue

        grand_mean = sub[score_col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((sub[score_col] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        f_stat, p_value = stats.f_oneway(*groups)

        rows.append(
            {
                "parameter": param,
                "n_levels": len(levels),
                "n_configs": len(sub),
                "eta_squared": eta_sq,
                "f_stat": f_stat,
                "p_value": p_value,
                "effect_range": group_means.max() - group_means.min(),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        "eta_squared", ascending=False, na_position="last"
    ).reset_index(drop=True)


def main_effects(
    df: pd.DataFrame,
    score_col: str,
    param_cols: list[str],
) -> pd.DataFrame:
    """Long-form (parameter, level, mean, std, n) table for main-effects plots.

    One row per (parameter, level) pair: the mean/std of `score_col` across
    all configs at that level, marginalized over every other parameter --
    the same grouping `parameter_importance` tests, laid out for a bar or
    line plot instead of a single effect-size number.
    """
    rows = []
    for param in param_cols:
        if param not in df.columns:
            continue
        sub = df.dropna(subset=[param, score_col])
        grouped = sub.groupby(param, dropna=False)[score_col]
        for level, values in grouped:
            rows.append(
                {
                    "parameter": param,
                    "level": level,
                    "mean": values.mean(),
                    "std": values.std(ddof=0),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)
