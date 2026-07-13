"""Parameter-importance (sensitivity) analysis for any per-config results table.

Applies to any dataframe with one row per (config[, window/horizon/step])
and a numeric score column -- `screening_sweep.py`'s `per_window_results.csv`
and `config_selection.summarize_configs` output, but equally
`temporal_decay.py`'s `per_horizon_results.csv`, `rolling_walk_forward.py`'s
`per_step_results.csv`, `monitor_drift.py`'s `per_horizon_results.csv`, or an
unrelated sweep with its own parameter columns. Of the axes a sweep varied,
which ones actually move the score, and which ones are noise? That's the
evidence for a thesis claim like "we fix max_depth and only vary granularity
in Experiments 2-4" -- it needs to show the fixed axis had small effect and
the varied one didn't.

`parameter_importance` runs a one-way ANOVA of `score_col` against each
candidate parameter in turn (marginalizing over all other parameters, one
row per config as the unit of observation -- the same rows `config_selection`
ranks). `eta_squared` (SS_between / SS_total) is the effect-size to lead
with: it's on a 0-1 scale and comparable across parameters regardless of how
many levels each has, unlike the F-statistic. `main_effects` produces the
long-form table (parameter, level, mean, std, n) behind a main-effects plot.
Both take a plain `param_cols: list[str]` of columns already in `df` -- no
knowledge of where those columns came from.

Several sweeps in this codebase (screening_sweep, temporal_decay,
rolling_walk_forward, monitor_drift) bundle their tunable thresholds into one
opaque `mining_setting` name (e.g. `"gr3.0_md5"`) shared via a
`configs/*.yaml` file of `MiningSettingSpec` entries (see
`thesis.config.load_mining_settings`) instead of putting each threshold in
its own column. A parameter-importance table over that bundle could only
ever say "mining_setting matters", not which of its thresholds is doing the
work -- `expand_named_setting` unpacks it (or any other named-preset axis,
mining-related or not) into one column per parameter first, given a
`{name: {parameter: value}}` map. `mining_setting_param_map` builds that map
from a loaded `MiningSettingSpec` list by reading the spec's actual field
values, so it stays correct if a YAML ever adds a third axis (e.g.
`min_samples_leaf`) or changes its naming convention -- nothing here is tied
to the `"gr{X}_md{Y}"` string shape.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from thesis.schemas.mining import MiningSettingSpec


def expand_named_setting(
    df: pd.DataFrame,
    param_map: dict[str, dict[str, Any]],
    col: str,
) -> pd.DataFrame:
    """Join each row's `col` value to the parameters that name denotes.

    `param_map` is `{name: {parameter: value, ...}, ...}` -- e.g. built by
    `mining_setting_param_map` from a loaded `MiningSettingSpec` list, or by
    hand for any other named-preset axis (a `mining_filters_*.yaml` filter
    preset, a `run_attribute_mining_window_sweep.py` run tag, ...). Adds one
    column per parameter key seen across `param_map`'s values. Rows whose
    `col` value has no entry in `param_map` (e.g. the baseline feature_set,
    which has no `mining_setting`) get `NaN` in every added column rather
    than raising, so callers can dropna per-parameter downstream (as
    `parameter_importance`/`main_effects` do).
    """
    parsed = pd.DataFrame([param_map.get(name, {}) for name in df[col]], index=df.index)
    return pd.concat([df, parsed], axis=1)


def mining_setting_param_map(
    mining_settings: list["MiningSettingSpec"],
) -> dict[str, dict[str, Any]]:
    """Build an `expand_named_setting` `param_map` from loaded `MiningSettingSpec`s.

    Flattens each spec's `contrast` and `tree` fields into one dict per
    `name` (e.g. `min_growth_rate`, `max_depth`, plus every other
    `AttributeMiningConfig` field -- `min_attack_coverage`,
    `min_samples_leaf`, etc.) by reading the spec object directly, so a
    parameter that's currently fixed across every named setting in the YAML
    still shows up as a same-valued column (surfaced by `parameter_importance`
    as "not screened here", `n_levels == 1`) instead of being silently
    absent. `contrast` and `tree` share no field names, so merging them
    unprefixed is unambiguous.
    """
    return {
        spec.name: {**spec.contrast.model_dump(), **spec.tree.model_dump()}
        for spec in mining_settings
    }


# Cohen's (1988) conventional benchmarks for eta-squared -- a reference scale
# for "how big is this effect", independent of and not to be confused with
# whatever ad hoc cutoff a caller uses to decide "vary this in later
# experiments" (e.g. config_selection.ipynb's IMPORTANT_ETA_THRESHOLD).
EFFECT_SIZE_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}


def effect_size_label(eta_squared: float) -> str:
    """Cohen's-convention label for an `eta_squared` value.

    One of `"negligible"`, `"small"`, `"medium"`, `"large"` -- or `"n/a"` for
    `NaN` (a parameter `parameter_importance` couldn't test, e.g. it had a
    single level in this sweep). Purely a labeling convenience over
    `EFFECT_SIZE_THRESHOLDS`; doesn't change how `parameter_importance`
    ranks or filters anything.
    """
    if pd.isna(eta_squared):
        return "n/a"
    if eta_squared >= EFFECT_SIZE_THRESHOLDS["large"]:
        return "large"
    if eta_squared >= EFFECT_SIZE_THRESHOLDS["medium"]:
        return "medium"
    if eta_squared >= EFFECT_SIZE_THRESHOLDS["small"]:
        return "small"
    return "negligible"


def restrict_to_fixed_levels(
    df: pd.DataFrame,
    fixed_levels: dict[str, Any],
) -> pd.DataFrame:
    """Keep only rows matching every `param -> level` pin in `fixed_levels`.

    The generic other half of a "fix the parameters `parameter_importance`
    found negligible, keep every row for the ones that still matter" shortlist:
    pass it `{param: winning_level}` for whichever parameters came back below
    your importance cutoff (e.g. `min_growth_rate` at 3.0), and it drops every
    row using a different level of those parameters -- without touching rows'
    values for any parameter not in `fixed_levels` (e.g. `granularity`,
    `max_depth`), so full coverage is kept on the axes that do matter.

    A row is kept if, for every `param` in `fixed_levels` that's a column in
    `df`, that row's value either equals `fixed_levels[param]` or is null --
    a baseline config has no mining-derived parameter value to match against,
    so there's nothing to filter it on and it's always kept.
    """
    mask = pd.Series(True, index=df.index)
    for param, level in fixed_levels.items():
        if param not in df.columns:
            continue
        col = df[param]
        mask &= col.isna() | (col == level)
    return df[mask]


_COMPARISONS: dict[str, Any] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def feasible_region(
    df: pd.DataFrame,
    constraints: dict[str, tuple[str, float]],
) -> pd.DataFrame:
    """Flag which rows satisfy several quality/budget constraints at once.

    The multi-constraint counterpart to `restrict_to_fixed_levels`: that one
    keeps rows matching an exact `param -> level` pin, this one keeps track of
    which rows clear a `metric -> (op, threshold)` bar, on possibly several
    metrics simultaneously -- e.g. "does this config's schema stay small
    enough *and* precise enough *and* stable enough" rather than one metric
    read in isolation. `constraints` is `{column: (op, threshold)}`, `op` one
    of `">="`, `"<="`, `">"`, `"<"`, `"=="`, e.g.
    `{"mean_tree_precision_attack": (">=", 0.7), "n_features": ("<=", 150)}`.

    Adds one `passes_<column>` bool per constraint (same naming spirit as
    `config_selection.apply_floor_check`'s `passes_floor`, generalized to
    several columns at once), plus `n_constraints_satisfied` (int) and
    `feasible` (bool, every constraint met). Rows are never dropped -- a
    near-miss config's *specific* blocking constraint is often as useful to
    see as the final pass/fail verdict, so filtering to `df[df["feasible"]]`
    is left to the caller.

    NaN in a constrained column reads as failing that constraint (every
    comparison against NaN is `False`), which silently looks the same as
    "genuinely below the threshold" -- a caller working with a metric that
    can legitimately be NaN (e.g. a mining stat that's undefined when a
    config produced zero leaves of some class) should check for that
    separately rather than trust `feasible == False` to mean "measured and
    failed".
    """
    out = df.copy()
    passes_cols = []
    for column, (op, threshold) in constraints.items():
        passes_col = f"passes_{column}"
        out[passes_col] = _COMPARISONS[op](out[column], threshold)
        passes_cols.append(passes_col)
    out["n_constraints_satisfied"] = out[passes_cols].sum(axis=1)
    out["feasible"] = out[passes_cols].all(axis=1)
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
