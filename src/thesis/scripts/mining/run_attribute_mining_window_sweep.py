"""
Attribute mining window sweep.

Mines features in consecutive temporal windows of a scenario's alert_groups
(at multiple granularity levels) using the per-alert-group attribute mining
pipeline (mining/attribute_contrast_mining.py + mining/decision_tree_rule_mining.py)
and analyses stability over time, Jaccard convergence, why categorical
predicates get dropped by Step 1's filter, feature persistence, cross-
granularity consistency, and drift in the decision tree's top rule.

This is the attribute-mining counterpart to run_mining_window_sweep.py, which
only covers the old cross-signature/cross-alert co-occurrence path (Eclat +
PrefixSpan). There is no "modes" axis here (benign/mixed/smart): the two-stage
attribute pipeline mines both classes jointly in a single pass per window
(Step 1's contrast stats and Step 2's class_weight="balanced" tree both use
the window's own label distribution directly), so there's nothing to hold out
or rescue after the fact.

Each window's mining is done via direct function calls (build_categorical_
predicate_matrix -> compute_predicate_contrast_stats -> filter_contrast_
survivors -> build_training_matrix -> fit_rule_tree -> extract_leaf_rules),
not the MLflow-wrapped run_alert_group_attribute_mining_job -- a sweep needs
many fast window/granularity combinations, not individually tracked
experiment runs.

Usage:
    # CSCAS (pre-grouped Suricata scenario) -- run once first:
    #   python src/thesis/scripts/run_ingest_cscas.py
    python src/thesis/scripts/run_attribute_mining_window_sweep.py cscas \\
        --granularities 0.1 0.2 0.33

    # Tune the Step 1 / Step 2 thresholds
    python src/thesis/scripts/run_attribute_mining_window_sweep.py cscas \
        --granularities 0.25 \
        --min-growth-rate 4.0 --max-depth 5

    # Skip Step 1's contrast-set filter: hand every candidate predicate to
    # Step 2's tree instead of only the ones that cleared the coverage/
    # growth-rate thresholds
    python src/thesis/scripts/run_attribute_mining_window_sweep.py cscas \\
        --granularities 0.25 --no-contrast-filter

Output (under artifacts/experiments/run_attribute_mining_window_sweep/<dataset>/<run_tag>/;
        --output-dir overrides this):
    window_features_<scenario>_<gran>_<win>_step1_raw.csv         — every Step 1
        candidate (single + pairwise categorical predicate), before filtering
    window_features_<scenario>_<gran>_<win>_step1_survivors.csv   — Step 1
        candidates that cleared the contrast-set filter
    window_features_<scenario>_<gran>_<win>_step2_survivors.csv   — Step 2
        decision-tree leaf rules
    mined_features_overview_step1_raw.csv          — all step1_raw files, concatenated
    mined_features_overview_step1_survivors.csv    — all step1_survivors files, concatenated
    mined_features_overview_step2_survivors.csv    — all step2_survivors files, concatenated
    table1_stability.csv            — within-scenario stability across windows
    table2_convergence.csv          — Jaccard convergence per transition
    table3_contrast_drop_diagnosis.csv — why Step 1 dropped a categorical candidate
    table4_persistence.csv          — feature persistence histogram
    table5_cross_granularity.csv    — cross-granularity consistency
    table6_rule_drift.csv           — top decision-tree rule per window, over time
    summary.txt                     — human-readable digest
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from thesis.configs import dataset_for_scenario
from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    extract_leaf_rules,
    fit_rule_tree,
)
from thesis.pipeline.pipeline import alert_group_from_dict
from thesis.schemas.mining import AttributeMiningConfig
from thesis.visualization.mining import attribute_window_sweep as plots

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_CACHE_DIR = _REPO / "artifacts" / "cache"
_EXPERIMENTS_DIR = (
    _REPO / "artifacts" / "experiments" / "run_attribute_mining_window_sweep"
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_raw_alert_groups(
    scenario: str, grouping_method: str = "cscas_pregrouped"
) -> list[dict]:
    path = (
        _CACHE_DIR
        / scenario
        / "groups"
        / grouping_method
        / "alert_groups"
        / "alert_groups_raw.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"No alert_groups found for '{scenario}' (grouping_method="
            f"'{grouping_method}') at {path}\n"
            "Run `python src/thesis/scripts/run_ingest_cscas.py` first."
        )
    with path.open() as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("group_label") in ("benign", "attack")]
    rows.sort(key=lambda r: r.get("start_ts") or 0)
    return rows


def _compute_f_attack(rows: list[dict]) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    n_attack = sum(1 for r in rows if r.get("group_label") == "attack")
    return n_attack / n


def _window_cache_key(
    scenario: str,
    gran: float,
    win_idx: int,
    config: AttributeMiningConfig,
    no_contrast_filter: bool = False,
    eval_frac: float = 0.3,
) -> str:
    gran_tag = f"{gran:.6f}".rstrip("0").rstrip(".")
    c, t = config.contrast, config.tree
    # Every parameter that changes what gets mined must be part of the cache
    # key -- otherwise a later run with different thresholds silently reuses
    # another run's cached parquets for the same (scenario, gran, win_idx).
    filter_tag = "__nofilter" if no_contrast_filter else ""
    return (
        f"{scenario}__{gran_tag}__{win_idx}"
        f"__cov{c.min_attack_coverage:g}-{c.min_benign_coverage:g}"
        f"__gr{c.min_growth_rate:g}__depth{t.max_depth}__leaf{t.min_samples_leaf}"
        f"__mid{t.min_impurity_decrease:g}__ef{eval_frac:g}"
        f"{filter_tag}"
    )


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def _read_itemset_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "itemset" in df.columns:
        # Parquet round-trips tuples as lists/arrays; restore for set/isin ops.
        df["itemset"] = df["itemset"].apply(
            lambda s: tuple(s) if not isinstance(s, tuple) else s
        )
    return df


def _run_mining_for_window(
    win_rows: list[dict],
    scenario: str,
    gran: float,
    win_idx: int,
    config: AttributeMiningConfig,
    cache_dir: Path,
    no_contrast_filter: bool = False,
    eval_frac: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mine one window. Returns (contrast_stats_all, contrast_survivors,
    decision_tree_rules). Cached as parquet in cache_dir to avoid redundant
    recomputation on reruns.

    no_contrast_filter=True skips Step 1's contrast-set filter entirely, so
    survivors_df == stats_df and every candidate (single + pairwise
    categorical predicate) is handed to Step 2's tree instead of just the
    ones that cleared min_attack_coverage/min_benign_coverage/min_growth_rate.

    eval_frac chronologically holds out the last eval_frac of this window
    from Step 2's tree fit: the tree is fit on the first (1 - eval_frac)
    portion only, and its leaves' support/confidence numbers are recomputed
    against the held-out last eval_frac portion instead of the data the tree
    was fit on. This is deliberately NOT a random split -- win_rows is
    already temporally sorted (see _load_raw_alert_groups), and randomly
    holding out samples from within the window would let the tree train on
    later data and "test" on earlier data, leaking the future into the fit.
    A chronological split answers the question that matters for a system
    that only ever sees future traffic: does a rule mined from the earlier
    part of the window still hold up on data that came after it. Step 1
    (contrast_stats_all/contrast_survivors, used by tables 1-5's stability
    analysis) is unaffected and still computed on the full window.
    """
    key = _window_cache_key(
        scenario, gran, win_idx, config, no_contrast_filter, eval_frac
    )
    win_cache = cache_dir / key
    stats_path = win_cache / "contrast_stats_all.parquet"
    survivors_path = win_cache / "contrast_survivors.parquet"
    leaves_path = win_cache / "decision_tree_rules.parquet"

    if stats_path.exists() and survivors_path.exists() and leaves_path.exists():
        return (
            _read_itemset_parquet(stats_path),
            _read_itemset_parquet(survivors_path),
            _read_itemset_parquet(leaves_path),
        )

    win_cache.mkdir(parents=True, exist_ok=True)
    alert_groups = [alert_group_from_dict(r) for r in win_rows]

    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(
        alert_groups
    )
    stats_df = compute_predicate_contrast_stats(X_cat, y, column_predicate_map)
    if no_contrast_filter:
        survivors_df = stats_df
    else:
        survivors_df = filter_contrast_survivors(
            stats_df,
            min_attack_coverage=config.contrast.min_attack_coverage,
            min_benign_coverage=config.contrast.min_benign_coverage,
            min_growth_rate=config.contrast.min_growth_rate,
            max_p_value=config.contrast.max_p_value,
        )
    surviving_cols = surviving_single_columns(survivors_df)

    X_train, kept_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_cols
    )

    n = len(X_train)
    n_fit = int(round((1.0 - eval_frac) * n)) if 0.0 < eval_frac < 1.0 else n
    n_fit = max(1, min(n_fit, n - 1)) if n > 1 else n
    X_fit, y_fit = X_train.iloc[:n_fit], y.iloc[:n_fit]
    X_eval, y_eval = X_train.iloc[n_fit:], y.iloc[n_fit:]
    if X_eval.empty:
        # Window too small to hold anything out; fall back to resubstitution
        # rather than reporting rules with an empty (all-NaN) evaluation.
        X_eval, y_eval = X_fit, y_fit

    n_attack_eval = int((y_eval.to_numpy() == 1).sum())
    if n_attack_eval < config.tree.min_samples_leaf:
        print(
            f"    [warn] gran={gran:.2f} window {win_idx}: only {n_attack_eval} "
            f"attack alert_groups in the {eval_frac:.0%} chronological eval "
            "holdout -- this window's rule confidence estimates may be unreliable.",
            end=" ",
        )

    tree = fit_rule_tree(
        X_fit,
        y_fit,
        max_depth=config.tree.max_depth,
        min_samples_leaf=config.tree.min_samples_leaf,
        class_weight=config.tree.class_weight,
        random_state=config.tree.random_state,
        min_impurity_decrease=config.tree.min_impurity_decrease,
    )
    leaves_df, _predicates = extract_leaf_rules(
        tree, X_eval, y_eval, kept_predicate_map
    )
    if not leaves_df.empty:
        leaves_df["n_attack_eval"] = n_attack_eval

    stats_df.to_parquet(stats_path, index=False)
    survivors_df.to_parquet(survivors_path, index=False)
    leaves_df.to_parquet(leaves_path, index=False)

    return stats_df, survivors_df, leaves_df


def _feature_id_from_row(row: pd.Series, mining_type: str) -> str:
    itemset = row.get("itemset")
    if itemset is None:
        return "?"
    prefix = "contrast" if mining_type == "contrast_categorical" else "tree"
    return f"{prefix}:" + "|".join(sorted(str(x) for x in itemset))


def _feature_set_from_dfs(
    survivors_df: pd.DataFrame, leaves_df: pd.DataFrame
) -> set[str]:
    ids: set[str] = set()
    for _, row in survivors_df.iterrows():
        ids.add(_feature_id_from_row(row, "contrast_categorical"))
    for _, row in leaves_df.iterrows():
        ids.add(_feature_id_from_row(row, "decision_tree_rule"))
    return ids


_CONTRAST_DETAIL_COLUMNS = [
    "support",
    "support_count",
    "confidence_attack",
    "confidence_benign",
    "growth_rate",
    "p_value",
]
_TREE_DETAIL_COLUMNS = [
    "support",
    "support_count",
    "confidence_attack",
    "confidence_benign",
    "n_attack",
    "n_benign",
    "depth",
    "leaf_id",
    "n_attack_eval",
]


def _feature_detail_rows(df: pd.DataFrame, mining_type: str) -> list[dict]:
    """
    Row-per-pattern view of a mined DataFrame, for the human-inspectable
    per-pipeline-step overview CSVs -- unlike _feature_set_from_dfs (an opaque
    id used only for Jaccard/stability bookkeeping), this keeps the actual
    support/confidence numbers so rules can be eyeballed directly. Which
    pipeline step a row came from is conveyed by which file it's written to
    (step1_raw / step1_survivors / step2_survivors), not by a column.
    """
    if df.empty:
        return []
    cols = (
        _CONTRAST_DETAIL_COLUMNS
        if mining_type == "contrast_categorical"
        else _TREE_DETAIL_COLUMNS
    )
    rows: list[dict] = []
    for _, row in df.iterrows():
        rec = {
            "mining_type": mining_type,
            "pattern": " AND ".join(str(x) for x in row["itemset"]),
        }
        for col in cols:
            if col in row.index:
                rec[col] = row.get(col)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Window result record
# ---------------------------------------------------------------------------


class WindowResult(NamedTuple):
    scenario: str
    gran: float
    win_idx: int
    win_start_frac: float
    win_end_frac: float
    n_alert_groups: int
    n_attack_alert_groups: int
    features_raw: set[str]
    features_filtered: set[str]
    contrast_stats_all: pd.DataFrame
    contrast_survivors: pd.DataFrame
    decision_tree_rules: pd.DataFrame


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------


def table1_stability(results: list[WindowResult]) -> pd.DataFrame:
    """How does the (survivor + leaf-rule) feature set change between consecutive windows?"""
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.gran)].append(r)

    for (scenario, gran), wins in groups.items():
        wins = sorted(wins, key=lambda w: w.win_idx)
        prev_set: set[str] | None = None
        for w in wins:
            cur = w.features_filtered
            base = {
                "scenario": scenario,
                "gran": gran,
                "window": w.win_idx,
                "win_range": f"{w.win_start_frac:.0%}–{w.win_end_frac:.0%}",
                "n_tx": w.n_alert_groups,
                "n_benign_tx": w.n_alert_groups - w.n_attack_alert_groups,
                "n_attack_tx": w.n_attack_alert_groups,
                "n_features": len(cur),
            }
            if prev_set is None:
                row = {
                    **base,
                    "n_added": None,
                    "n_removed": None,
                    "n_unchanged": None,
                    "jaccard": None,
                }
            else:
                added, removed, unchanged = (
                    cur - prev_set,
                    prev_set - cur,
                    cur & prev_set,
                )
                union = cur | prev_set
                jaccard = len(unchanged) / len(union) if union else float("nan")
                row = {
                    **base,
                    "n_added": len(added),
                    "n_removed": len(removed),
                    "n_unchanged": len(unchanged),
                    "jaccard": round(jaccard, 4),
                }
            rows.append(row)
            prev_set = cur
    return pd.DataFrame(rows)


def table2_convergence(results: list[WindowResult]) -> pd.DataFrame:
    """Jaccard similarity between consecutive windows per granularity, to compare convergence trajectories."""
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.gran)].append(r)

    for (scenario, gran), wins in groups.items():
        wins = sorted(wins, key=lambda w: w.win_idx)
        for i in range(1, len(wins)):
            prev, cur = wins[i - 1].features_filtered, wins[i].features_filtered
            union = prev | cur
            jaccard = len(prev & cur) / len(union) if union else float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "gran": gran,
                    "transition": f"w{i - 1}→w{i}",
                    "win_range": f"{wins[i - 1].win_start_frac:.0%}–{wins[i].win_end_frac:.0%}",
                    "jaccard": round(jaccard, 4),
                    "n_prev": len(prev),
                    "n_cur": len(cur),
                }
            )
    return pd.DataFrame(rows)


def table3_contrast_drop_diagnosis(
    results: list[WindowResult],
    min_attack_coverage: float,
    min_benign_coverage: float,
    min_growth_rate: float,
) -> pd.DataFrame:
    """
    For categorical predicates/pairs that Step 1 mined but then dropped,
    report which criterion failed: growth_rate too close to 1 (fires about
    equally in both classes), or insufficient coverage on the class it leans
    toward (fires on too few groups of that class to be a reliable signal).
    """
    rows = []
    inv_threshold = 1.0 / min_growth_rate if min_growth_rate > 0 else float("inf")

    for r in results:
        if r.contrast_stats_all.empty:
            continue
        survivor_itemsets = (
            set(r.contrast_survivors["itemset"])
            if not r.contrast_survivors.empty
            else set()
        )
        dropped = r.contrast_stats_all[
            ~r.contrast_stats_all["itemset"].isin(survivor_itemsets)
        ]
        for _, row in dropped.iterrows():
            growth_rate = row["growth_rate"]
            attack_leaning = growth_rate >= min_growth_rate
            benign_leaning = (
                row["confidence_benign"] > 0 and growth_rate <= inv_threshold
            )

            if not (attack_leaning or benign_leaning):
                reason = "growth_rate_too_close_to_1"
            elif attack_leaning and row["confidence_attack"] < min_attack_coverage:
                reason = "insufficient_attack_coverage"
            elif benign_leaning and row["confidence_benign"] < min_benign_coverage:
                reason = "insufficient_benign_coverage"
            else:
                reason = "other"  # e.g. failed an enabled p-value gate

            rows.append(
                {
                    "scenario": r.scenario,
                    "gran": r.gran,
                    "window": r.win_idx,
                    "itemset": row["itemset"],
                    "growth_rate": round(growth_rate, 3),
                    "confidence_attack": round(row["confidence_attack"], 4),
                    "confidence_benign": round(row["confidence_benign"], 4),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def table4_persistence(results: list[WindowResult]) -> pd.DataFrame:
    """For each feature (within a scenario/gran), how many windows does it appear in?"""
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.gran)].append(r)

    for (scenario, gran), wins in groups.items():
        n_windows = len(wins)
        feat_counts: dict[str, int] = defaultdict(int)
        for w in wins:
            for feat in w.features_filtered:
                feat_counts[feat] += 1

        hist: dict[int, int] = defaultdict(int)
        for count in feat_counts.values():
            hist[count] += 1

        for n_wins_present, n_features in sorted(hist.items()):
            rows.append(
                {
                    "scenario": scenario,
                    "gran": gran,
                    "n_windows_present": n_wins_present,
                    "pct_of_total_windows": round(n_wins_present / n_windows, 3),
                    "n_features_with_this_persistence": n_features,
                }
            )
    return pd.DataFrame(rows)


def table5_cross_granularity(results: list[WindowResult]) -> pd.DataFrame:
    """Does a feature mined at coarse granularity also appear in the finer sub-windows covering the same range?"""
    rows = []
    by_scenario: dict[str, list[WindowResult]] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario].append(r)

    for scenario, wins in by_scenario.items():
        grans = sorted({w.gran for w in wins})
        if len(grans) < 2:
            continue

        fine_gran = grans[0]
        fine_wins = [w for w in wins if w.gran == fine_gran]
        for coarse_gran in grans[1:]:
            coarse_wins = [w for w in wins if w.gran == coarse_gran]
            for cw in coarse_wins:
                overlapping = [
                    fw
                    for fw in fine_wins
                    if fw.win_start_frac >= cw.win_start_frac - 1e-6
                    and fw.win_end_frac <= cw.win_end_frac + 1e-6
                ]
                if not overlapping:
                    continue

                coarse_feats = cw.features_filtered
                fine_union = set().union(*(fw.features_filtered for fw in overlapping))
                fine_intersection = set(overlapping[0].features_filtered)
                for fw in overlapping[1:]:
                    fine_intersection &= fw.features_filtered

                n_in_any = len(coarse_feats & fine_union)
                n_in_all = len(coarse_feats & fine_intersection)
                n_coarse = len(coarse_feats)

                rows.append(
                    {
                        "scenario": scenario,
                        "coarse_gran": coarse_gran,
                        "fine_gran": fine_gran,
                        "coarse_win": cw.win_idx,
                        "coarse_range": f"{cw.win_start_frac:.0%}–{cw.win_end_frac:.0%}",
                        "n_overlapping_fine_windows": len(overlapping),
                        "n_coarse_features": n_coarse,
                        "n_also_in_any_fine_window": n_in_any,
                        "n_also_in_all_fine_windows": n_in_all,
                        "frac_in_any": round(n_in_any / n_coarse, 3)
                        if n_coarse
                        else None,
                        "frac_in_all": round(n_in_all / n_coarse, 3)
                        if n_coarse
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def table6_rule_drift(results: list[WindowResult]) -> pd.DataFrame:
    """
    Tracks the strongest attack-indicating decision-tree leaf rule (highest
    confidence_attack, tie-broken by support) in each window, so drift in the
    dominant discriminative rule -- as opposed to churn in the full feature
    set -- is visible at a glance across time.

    confidence_attack/confidence_benign are chronological-holdout estimates
    (the tree is fit on the window's first (1-eval_frac) portion and
    evaluated on the last eval_frac), not resubstitution -- see
    _run_mining_for_window. n_attack_eval is how many attack alert_groups
    were actually in that holdout; treat a window's numbers with less trust
    when it's small (below the tree's own min_samples_leaf is a reasonable
    floor), since the confidence estimate itself is then noisy regardless of
    how clean it looks.
    """
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.gran)].append(r)

    for (scenario, gran), wins in groups.items():
        for w in sorted(wins, key=lambda w: w.win_idx):
            base = {
                "scenario": scenario,
                "gran": gran,
                "window": w.win_idx,
                "win_range": f"{w.win_start_frac:.0%}–{w.win_end_frac:.0%}",
            }
            empty_row = {
                **base,
                "top_rule": None,
                "confidence_attack": None,
                "confidence_benign": None,
                "support": None,
                "support_count": None,
                "n_attack_eval": None,
            }
            if w.decision_tree_rules.empty:
                rows.append(empty_row)
                continue
            top = w.decision_tree_rules.sort_values(
                ["confidence_attack", "support"], ascending=False
            ).iloc[0]
            if len(top["itemset"]) == 0:
                # Root == the only leaf (no split happened, usually because the
                # window is too small for --min-samples-leaf) -- confidence_attack/
                # confidence_benign would both trivially read 1.0 here (the one
                # leaf contains every group), which is not a real rule.
                rows.append(
                    {
                        **empty_row,
                        "top_rule": "(no split -- window too small for min-samples-leaf)",
                    }
                )
                continue
            rows.append(
                {
                    **base,
                    "top_rule": " AND ".join(str(x) for x in top["itemset"]),
                    "confidence_attack": round(top["confidence_attack"], 4),
                    "confidence_benign": round(top["confidence_benign"], 4),
                    "support": round(top["support"], 4),
                    "support_count": int(top["support_count"]),
                    "n_attack_eval": int(top["n_attack_eval"])
                    if "n_attack_eval" in top.index and pd.notna(top["n_attack_eval"])
                    else None,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_table(df: pd.DataFrame, title: str, max_rows: int = 40) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)
    if df.empty:
        print("  (no data)")
        return
    print(df.to_string(index=False, max_rows=max_rows))


def _summary_lines(t1, t2, t3, t4, t6) -> str:
    lines = []
    if not t1.empty:
        avg_j = t1["jaccard"].dropna().mean()
        min_j = t1["jaccard"].dropna().min()
        lines.append(
            f"Stability (T1): mean Jaccard={avg_j:.3f}, min={min_j:.3f} across all consecutive windows"
        )
    if not t3.empty:
        by_reason = t3["reason"].value_counts()
        lines.append(f"Drop diagnosis (T3): {by_reason.to_dict()}")
    if not t4.empty:
        max_per_gran = t4.groupby(["scenario", "gran"])["n_windows_present"].max()
        lines.append(
            f"Persistence (T4): max windows-present per (scenario,gran): {max_per_gran.max()}"
        )
    if not t6.empty:
        n_unique_rules = t6["top_rule"].nunique()
        lines.append(
            f"Rule drift (T6): {n_unique_rules} distinct top rules seen across {len(t6)} windows"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine attribute features in temporal windows and analyse stability/drift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="Scenario names (must have alert_group cache under artifacts/cache/<scenario>/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all scenarios found under artifacts/cache/ with a cscas_pregrouped-style cache.",
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.33],
        metavar="FRAC",
        help="Window size as fraction of total alert_groups. Default: 0.1 0.2 0.33",
    )
    parser.add_argument(
        "--grouping-method",
        type=str,
        default="cscas_pregrouped",
        metavar="METHOD",
        help="Grouping method subdir to read alert_groups from. Default: cscas_pregrouped",
    )
    parser.add_argument("--min-attack-coverage", type=float, default=0.05)
    parser.add_argument("--min-benign-coverage", type=float, default=0.05)
    parser.add_argument("--min-growth-rate", type=float, default=3.0)
    parser.add_argument(
        "--max-p-value",
        type=float,
        default=None,
        help="Optional chi-square significance gate for Step 1. Default: off.",
    )
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument(
        "--no-contrast-filter",
        action="store_true",
        dest="no_contrast_filter",
        help=(
            "Skip Step 1's contrast-set filter entirely: every candidate "
            "single/pairwise categorical predicate is handed to Step 2's "
            "decision tree, instead of only the ones that clear "
            "--min-attack-coverage/--min-benign-coverage/--min-growth-rate."
        ),
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.3,
        dest="eval_frac",
        help=(
            "Fraction of each window (chronologically last) held out from "
            "Step 2's tree fit and used instead to recompute leaf support/"
            "confidence, so rule_drift/step2_survivors numbers reflect "
            "out-of-window generalization rather than a resubstitution "
            "estimate. Default: 0.3. Set to 0 to fit and evaluate on the "
            "full window (the old resubstitution behavior)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override. Default: artifacts/experiments/run_attribute_mining_window_sweep/<dataset>/<run_tag>/",
    )
    args = parser.parse_args()

    if args.all:
        scenarios = sorted(
            d.name
            for d in _CACHE_DIR.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and (
                d
                / "groups"
                / args.grouping_method
                / "alert_groups"
                / "alert_groups_raw.json"
            ).exists()
        )
        if args.scenarios:
            print(f"[warn] --all overrides positional scenarios {args.scenarios}")
    else:
        scenarios = args.scenarios
    if not scenarios:
        parser.error("Provide at least one scenario name or use --all.")
    args.scenarios = scenarios

    config = AttributeMiningConfig()
    config.contrast.min_attack_coverage = args.min_attack_coverage
    config.contrast.min_benign_coverage = args.min_benign_coverage
    config.contrast.min_growth_rate = args.min_growth_rate
    config.contrast.max_p_value = args.max_p_value
    config.tree.max_depth = args.max_depth
    config.tree.min_samples_leaf = args.min_samples_leaf
    config.tree.class_weight = args.class_weight

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gran_tag = "-".join(f"{g:g}" for g in args.granularities)
    filter_run_tag = "_nofilter" if args.no_contrast_filter else ""
    run_tag = f"{ts}_gr{args.min_growth_rate:g}_depth{args.max_depth}_gran-{gran_tag}{filter_run_tag}"
    if args.output_dir is not None:
        out_dir: Path = args.output_dir
    else:
        datasets = {dataset_for_scenario(s) or "unknown" for s in scenarios}
        if len(datasets) > 1:
            print(
                f"[warn] scenarios span multiple datasets {sorted(datasets)} — grouping under 'mixed'"
            )
            dataset_tag = "mixed"
        else:
            dataset_tag = datasets.pop()
        out_dir = _EXPERIMENTS_DIR / dataset_tag / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    win_mining_cache = _REPO / "artifacts" / "cache" / "_attribute_window_mining"
    win_mining_cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: mine all windows
    # ------------------------------------------------------------------
    all_results: list[WindowResult] = []
    all_step1_raw_rows: list[dict] = []
    all_step1_survivor_rows: list[dict] = []
    all_step2_survivor_rows: list[dict] = []

    for scenario in args.scenarios:
        print(f"\n[scenario={scenario}] Loading alert_groups...")
        raw_rows = _load_raw_alert_groups(
            scenario, grouping_method=args.grouping_method
        )
        f_attack_scenario = _compute_f_attack(raw_rows)
        print(f"  {len(raw_rows)} alert_groups total, f_attack={f_attack_scenario:.4f}")

        for gran in args.granularities:
            n_total = len(raw_rows)
            win_size = max(1, int(gran * n_total))
            n_windows = max(1, n_total // win_size)

            print(
                f"  [gran={gran:.0%}] {n_windows} windows of ~{win_size} alert_groups each"
            )

            for win_idx in range(n_windows):
                start = win_idx * win_size
                end = (win_idx + 1) * win_size if win_idx < n_windows - 1 else n_total
                win_rows = raw_rows[start:end]
                win_start_frac, win_end_frac = start / n_total, end / n_total
                n_attack = sum(1 for r in win_rows if r.get("group_label") == "attack")

                print(
                    f"    window {win_idx}: [{win_start_frac:.0%}, {win_end_frac:.0%})"
                    f"  {len(win_rows)} tx  {n_attack} attack",
                    end=" ... ",
                    flush=True,
                )

                stats_df, survivors_df, leaves_df = _run_mining_for_window(
                    win_rows=win_rows,
                    scenario=scenario,
                    gran=gran,
                    win_idx=win_idx,
                    config=config,
                    cache_dir=win_mining_cache,
                    no_contrast_filter=args.no_contrast_filter,
                    eval_frac=args.eval_frac,
                )

                features_raw = _feature_set_from_dfs(stats_df, leaves_df)
                features_filtered = _feature_set_from_dfs(survivors_df, leaves_df)
                print(f"raw={len(features_raw)} filtered={len(features_filtered)}")

                all_results.append(
                    WindowResult(
                        scenario=scenario,
                        gran=gran,
                        win_idx=win_idx,
                        win_start_frac=win_start_frac,
                        win_end_frac=win_end_frac,
                        n_alert_groups=len(win_rows),
                        n_attack_alert_groups=n_attack,
                        features_raw=features_raw,
                        features_filtered=features_filtered,
                        contrast_stats_all=stats_df,
                        contrast_survivors=survivors_df,
                        decision_tree_rules=leaves_df,
                    )
                )

                def _tag_rows(rows: list[dict]) -> list[dict]:
                    return [
                        {"scenario": scenario, "gran": gran, "win_idx": win_idx, **rec}
                        for rec in rows
                    ]

                def _finalize(rows: list[dict]) -> pd.DataFrame:
                    df_ = pd.DataFrame(rows)
                    if not df_.empty:
                        df_ = df_.drop_duplicates(subset=["pattern"])
                    return df_

                step1_raw_df = _finalize(
                    _tag_rows(_feature_detail_rows(stats_df, "contrast_categorical"))
                )
                step1_survivors_df = _finalize(
                    _tag_rows(
                        _feature_detail_rows(survivors_df, "contrast_categorical")
                    )
                )
                step2_survivors_df = _finalize(
                    _tag_rows(_feature_detail_rows(leaves_df, "decision_tree_rule"))
                )

                wf_dir = out_dir / "window_features"
                wf_dir.mkdir(parents=True, exist_ok=True)
                win_stem = f"window_features_{scenario}_{gran:.2f}_{win_idx}"
                step1_raw_df.to_csv(wf_dir / f"{win_stem}_step1_raw.csv", index=False)
                step1_survivors_df.to_csv(
                    wf_dir / f"{win_stem}_step1_survivors.csv", index=False
                )
                step2_survivors_df.to_csv(
                    wf_dir / f"{win_stem}_step2_survivors.csv", index=False
                )

                all_step1_raw_rows.extend(step1_raw_df.to_dict("records"))
                all_step1_survivor_rows.extend(step1_survivors_df.to_dict("records"))
                all_step2_survivor_rows.extend(step2_survivors_df.to_dict("records"))

    # ------------------------------------------------------------------
    # Phase 2: analysis tables
    # ------------------------------------------------------------------
    print("\n\nComputing analysis tables...")

    t1 = table1_stability(all_results)
    t2 = table2_convergence(all_results)
    t3 = table3_contrast_drop_diagnosis(
        all_results,
        args.min_attack_coverage,
        args.min_benign_coverage,
        args.min_growth_rate,
    )
    t4 = table4_persistence(all_results)
    t5 = table5_cross_granularity(all_results)
    t6 = table6_rule_drift(all_results)

    _print_table(t1, "T1 — Within-scenario stability (consecutive Jaccard)")
    _print_table(t2, "T2 — Convergence curves (Jaccard per transition)")
    _print_table(t3, "T3 — Contrast-set drop diagnosis")
    _print_table(t4, "T4 — Feature persistence histogram")
    _print_table(t5, "T5 — Cross-granularity consistency")
    _print_table(t6, "T6 — Decision-tree top-rule drift")

    for df, name in [
        (t1, "table1_stability"),
        (t2, "table2_convergence"),
        (t3, "table3_contrast_drop_diagnosis"),
        (t4, "table4_persistence"),
        (t5, "table5_cross_granularity"),
        (t6, "table6_rule_drift"),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)

    def _write_overview(rows: list[dict], name: str) -> None:
        df_ = pd.DataFrame(rows)
        if not df_.empty:
            df_ = df_.sort_values(
                ["scenario", "gran", "win_idx", "support"],
                ascending=[True, True, True, False],
            )
        df_.to_csv(out_dir / f"{name}.csv", index=False)

    _write_overview(all_step1_raw_rows, "mined_features_overview_step1_raw")
    _write_overview(all_step1_survivor_rows, "mined_features_overview_step1_survivors")
    _write_overview(all_step2_survivor_rows, "mined_features_overview_step2_survivors")

    summary = _summary_lines(t1, t2, t3, t4, t6)
    (out_dir / "summary.txt").write_text(
        textwrap.dedent(f"""\
        Attribute Mining Window Sweep — {ts}
        Scenarios : {", ".join(args.scenarios)}
        Granularities: {", ".join(f"{g:.0%}" for g in args.granularities)}
        min_attack_coverage={args.min_attack_coverage}  min_benign_coverage={args.min_benign_coverage}
        min_growth_rate={args.min_growth_rate}  max_depth={args.max_depth}  min_samples_leaf={args.min_samples_leaf}
        no_contrast_filter={args.no_contrast_filter}  eval_frac={args.eval_frac}

        {summary}
        """)
    )

    # ------------------------------------------------------------------
    # Phase 3: plots
    # ------------------------------------------------------------------
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"\nGenerating plots → {plots_dir}")

    if not t1.empty:
        plots.plot_feature_count(t1, plots_dir)
        plots.plot_churn(t1, plots_dir)
    if not t2.empty:
        plots.plot_convergence(t2, plots_dir)
    if not t3.empty:
        plots.plot_drop_diagnosis(t3, plots_dir)
    if not t4.empty:
        plots.plot_persistence(t4, plots_dir)
    if not t5.empty:
        plots.plot_cross_granularity(t5, plots_dir)
    if not t6.empty:
        plots.plot_rule_drift(t6, plots_dir)
    plots.plot_feature_lifecycle(out_dir, plots_dir, gran_filter=args.granularities)

    print(f"\n\nOutputs saved to: {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
