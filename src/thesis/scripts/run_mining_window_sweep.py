"""
Mining window stability sweep.

Mines features in consecutive temporal windows of a scenario's alert_groups
(at multiple granularity levels) and analyses:

  1. Within-scenario stability  — does the mined feature set change over time?
     Tracks additions, removals, Jaccard similarity between consecutive windows.
  2. Cross-scenario sharing     — which features are shared across scenarios and
     at what frequency?
  3. Cross-scenario convergence — does Jaccard(window_n, window_n-1) plateau?
  4. Dropped-feature diagnosis  — for features that failed the filter, was
     support_attack ≈ support_benign × f_attack (uniform noise) or
     support_attack >> expectation (genuinely attack-correlated)?
  5. Benign-vs-mixed delta      — what features appear in mixed mode but not
     benign-only, i.e. what does removing attack traffic cost us?
  6. Feature persistence        — histogram of how many windows each feature
     survives (robust vs. ephemeral signals).
  7. Cross-granularity consistency — does a feature mined at coarse granularity
     also appear in the fine sub-windows that cover the same range?

Usage:
    python src/thesis/scripts/run_mining_window_sweep.py fox harrison \\
        --filter-config src/thesis/configs/mining_filters_simple.yaml \\
        --granularities 0.1 0.2 0.33 \\
        --modes benign mixed smart

    # Specific scenarios, one granularity
    python src/thesis/scripts/run_mining_window_sweep.py fox \\
        --granularities 0.25 \\
        --modes benign

    # CSCAS (pre-grouped Suricata scenario) — run once first:
    #   python src/thesis/scripts/run_ingest_cscas.py
    # then use --grouping-method since CSCAS alert_groups live under
    # groups/suricata_grouped/ rather than groups/fixed_window/. Sequence
    # mining is skipped automatically for these groups (sorted_items is
    # always empty — see _run_mining_for_window).
    python src/thesis/scripts/run_mining_window_sweep.py cscas \
        --grouping-method suricata_grouped \
        --granularities 0.1 0.2 0.33 \
        --modes benign mixed

Output (under artifacts/experiments/mining_window_sweep/<dataset>/<run_tag>/,
        where <dataset> is looked up per scenario via scenarios.json, e.g.
        'ait-ads' or 'cscas'; falls back to 'unknown'/'mixed' if scenarios
        aren't listed there or span multiple datasets; <run_tag> is
        <timestamp>_<filter_config_stem>_<modes>_gran-<granularities>, e.g.
        20260701_120000_mining_filters_simple_benign-mixed_gran-0.1-0.2-0.33;
        --output-dir overrides this entirely):
    window_features_<scenario>_<mode>_<gran>.csv  — mined features per window
    table1_stability_<scenario>_<mode>.csv         — within-scenario stability
    table2_sharing.csv                             — cross-scenario feature sharing
    table3_convergence.csv                         — Jaccard convergence curves
    table4_dropped_diagnosis.csv                   — filter-drop analysis
    table5_benign_vs_mixed.csv                     — benign/mixed delta
    table6_persistence.csv                         — feature persistence histograms
    table7_cross_granularity.csv                   — cross-granularity consistency
    summary.txt                                    — human-readable digest
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from thesis.configs import dataset_for_scenario
from thesis.mining.util import (
    filter_mined_sequences,
    remove_prefix_subsumed,
)
from thesis.mining.util import filter_mined_itemsets, remove_subset_subsumed

import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_CACHE_DIR = _REPO / "artifacts" / "cache"
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_mining_window_sweep"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_raw_alert_groups(
    scenario: str, grouping_method: str = "fixed_window"
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
        if grouping_method == "suricata_grouped":
            hint = "Run `python src/thesis/scripts/run_ingest_cscas.py` first."
        else:
            hint = (
                "Run `python -m thesis.experiments.runner baseline "
                f"{scenario}` first (or any experiment that ingests this "
                "scenario) to populate alert_groups."
            )
        raise FileNotFoundError(
            f"No alert_groups found for '{scenario}' (grouping_method="
            f"'{grouping_method}') at {path}\n{hint}"
        )
    with path.open() as f:
        rows = json.load(f)
    rows.sort(key=lambda r: r.get("start_ts") or 0)
    return rows


def _strip_attacks(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("group_label") == "benign"]


def _rows_to_mining_format(rows: list[dict]) -> list[dict]:
    """Re-serialise AlertGroup rows into the MiningAlertGroup format the mining jobs expect."""
    out = []
    for r in rows:
        out.append(
            {
                "alert_group_id": r["alert_group_id"],
                "window_start": r.get("start_ts"),
                "window_end": r.get("end_ts"),
                "n_alerts": r.get("n_alerts"),
                "abs_items": r.get("abs_items", []),
                "sorted_items": r.get("sorted_items", []),
                "group_label": r.get("group_label"),
                "alert_labels": r.get("alert_labels"),
                "weight": r.get("weight", 1.0),
            }
        )
    return out


def _save_temp_alert_groups(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f)


def _window_cache_key(scenario: str, mode: str, gran: float, win_idx: int) -> str:
    gran_tag = f"{gran:.6f}".rstrip("0").rstrip(".")
    return f"{scenario}__{mode}__{gran_tag}__{win_idx}"


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def _smart_filter_itemsets(
    raw_df: pd.DataFrame,
    fc_itemsets,
    f_attack: float,
    ratio_threshold: float,
) -> pd.DataFrame:
    """
    Filter itemsets like filter_mined_itemsets, but rescue features that fail
    only the support_diff criterion if their attack support is explained by
    uniform background traffic: support_attack ≤ support_benign × f_attack × ratio_threshold.
    All other criteria (min_k, min_support_count, confidence, overlap,
    non-redundancy) apply unchanged.
    """

    if raw_df.empty:
        return raw_df.copy()

    # Parquet round-trips convert tuples to lists; restore for itemset operations
    raw_df = raw_df.copy()
    if "itemset" in raw_df.columns:
        raw_df["itemset"] = raw_df["itemset"].apply(
            lambda s: tuple(s) if not isinstance(s, tuple) else s
        )

    f = fc_itemsets

    # Apply all structural + discriminative filters *except* support_diff
    structurally_valid = filter_mined_itemsets(
        raw_df,
        min_k=f.min_k,
        max_k=f.max_k,
        min_support_count=f.min_support_count,
        min_abs_support_diff=0.0,  # disabled here; handled below
        min_confidence_attack=f.min_confidence_attack,
        max_confidence_attack=f.max_confidence_attack,
        min_confidence_benign=f.min_confidence_benign,
        max_overlap=f.max_overlap,
        remove_subsumed=False,  # defer until after rescue
    )

    if structurally_valid.empty:
        return structurally_valid

    sb = structurally_valid.get(
        "support_benign", pd.Series(0.0, index=structurally_valid.index)
    )
    sa = structurally_valid.get(
        "support_attack", pd.Series(0.0, index=structurally_valid.index)
    )
    sd = structurally_valid.get("support_diff", (sb - sa))

    expected_sa = sb * f_attack
    ratio = sa / (expected_sa.replace(0, float("nan")))

    # Rescue only features where sb already clears the threshold independently —
    # i.e., the feature is genuinely strong in benign but attack contamination is
    # pulling support_diff below the cutoff.  Features with sb < min_abs_support_diff
    # are weak regardless of attack support and should not be rescued.
    rescue = (sb >= f.min_abs_support_diff) & (ratio <= ratio_threshold)
    passes = (sd.abs() >= f.min_abs_support_diff) | rescue
    result = structurally_valid[passes].reset_index(drop=True)

    if f.remove_subsumed:
        result = remove_subset_subsumed(result)

    return result


def _smart_filter_sequences(
    raw_df: pd.DataFrame,
    fc_sequences,
    f_attack: float,
    ratio_threshold: float,
) -> pd.DataFrame:
    """
    Same rescue logic as _smart_filter_itemsets but for sequence DataFrames.
    """

    if raw_df.empty:
        return raw_df.copy()

    # Parquet round-trips convert tuples to lists, breaking add_lift_scores
    raw_df = raw_df.copy()
    if "sequence" in raw_df.columns:
        raw_df["sequence"] = raw_df["sequence"].apply(
            lambda s: tuple(s) if not isinstance(s, tuple) else s
        )

    f = fc_sequences

    structurally_valid = filter_mined_sequences(
        raw_df,
        min_k=f.min_k,
        min_support_count=f.min_support_count,
        min_abs_support_diff=0.0,
        min_confidence_attack=f.min_confidence_attack,
        max_confidence_attack=f.max_confidence_attack,
        min_confidence_benign=f.min_confidence_benign,
        min_lift=f.min_lift,
        max_overlap=f.max_overlap,
        remove_subsumed=False,
    )

    if structurally_valid.empty:
        return structurally_valid

    sb = structurally_valid.get(
        "support_benign", pd.Series(0.0, index=structurally_valid.index)
    )
    sa = structurally_valid.get(
        "support_attack", pd.Series(0.0, index=structurally_valid.index)
    )
    sd = structurally_valid.get("support_diff", (sb - sa))

    expected_sa = sb * f_attack
    ratio = sa / (expected_sa.replace(0, float("nan")))

    rescue = (sb >= f.min_abs_support_diff) & (ratio <= ratio_threshold)
    passes = (sd.abs() >= f.min_abs_support_diff) | rescue
    result = structurally_valid[passes].reset_index(drop=True)

    if f.remove_subsumed:
        result = remove_prefix_subsumed(result)

    return result


def _run_mining_for_window(
    rows: list[dict],
    scenario: str,
    mode: str,
    gran: float,
    win_idx: int,
    min_support: float,
    max_itemset_size: int,
    max_seq_len: int,
    filter_config: Path | None,
    cache_dir: Path,
    smart_ratio_threshold: float = 1.5,
    f_attack_scenario: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mine one window. Returns (eclat_raw, eclat_filtered, seq_raw, seq_filtered).
    Results are cached in cache_dir to avoid redundant runs.

    mode="smart" uses the same full (mixed) alert_groups as mode="mixed" but
    applies a ratio-based rescue to features that fail the support_diff filter
    only because attack traffic appears at background-noise rate.

    f_attack_scenario must be the scenario-level attack fraction (n_attack /
    n_total across ALL alert_groups, not just this window). Using a per-window
    f_attack is circular: in a heavily attack-laden window the expected
    background support inflates, rescuing almost everything incorrectly.
    """
    key = _window_cache_key(scenario, mode, gran, win_idx)
    win_cache = cache_dir / key
    eclat_raw_path = win_cache / "eclat_raw.parquet"
    eclat_filt_path = win_cache / "eclat_filtered.parquet"
    seq_raw_path = win_cache / "seq_raw.parquet"
    seq_filt_path = win_cache / "seq_filtered.parquet"
    filter_hash_path = win_cache / "filter_config_hash.txt"

    # Compute a fingerprint of the current filter config so we can detect stale
    # filtered caches without re-running the expensive mining step.
    def _filter_fingerprint() -> str:
        import hashlib

        if filter_config is None:
            return "none"
        raw = Path(filter_config).read_bytes() if Path(filter_config).exists() else b""
        return hashlib.md5(raw).hexdigest()

    raw_cached = eclat_raw_path.exists() and seq_raw_path.exists()
    filt_cached = eclat_filt_path.exists() and seq_filt_path.exists()
    filter_hash_matches = (
        filter_hash_path.exists()
        and filter_hash_path.read_text().strip() == _filter_fingerprint()
    )

    if raw_cached and filt_cached and filter_hash_matches:
        return (
            pd.read_parquet(eclat_raw_path),
            pd.read_parquet(eclat_filt_path),
            pd.read_parquet(seq_raw_path),
            pd.read_parquet(seq_filt_path),
        )

    win_cache.mkdir(parents=True, exist_ok=True)

    # For smart mode, mining runs on full mixed alert_groups (same as "mixed")
    # but filtering uses the ratio-based rescue instead of hard support_diff.
    # We reuse the mixed mode's cached raw parquets if available to avoid
    # re-running the expensive mining step.
    if mode == "smart":
        mixed_key = _window_cache_key(scenario, "mixed", gran, win_idx)
        mixed_cache = cache_dir / mixed_key
        mixed_eclat_raw = mixed_cache / "eclat_raw.parquet"
        mixed_seq_raw = mixed_cache / "seq_raw.parquet"
        if mixed_eclat_raw.exists() and mixed_seq_raw.exists():
            eclat_raw = pd.read_parquet(mixed_eclat_raw)
            seq_raw = pd.read_parquet(mixed_seq_raw)
            # Skip to filtering
            eclat_raw.to_parquet(eclat_raw_path, index=False)
            seq_raw.to_parquet(seq_raw_path, index=False)
        else:
            # Mixed not yet cached — mine from scratch (will also populate below)
            eclat_raw, seq_raw = None, None
    else:
        eclat_raw, seq_raw = None, None

    if eclat_raw is None:
        # Persist window alert_groups so consecutive runs skip re-slicing
        tx_path = win_cache / "alert_groups.json"
        if not tx_path.exists():
            with tx_path.open("w") as f:
                json.dump(_rows_to_mining_format(rows), f)

        from thesis.mining.itemset_mining_job import run_alert_group_eclat_job
        from thesis.mining.sequence_mining_job import run_alert_group_prefixspan_job

        run_name = f"window_sweep_{key}"
        job_dir = win_cache / "mining"
        job_dir.mkdir(exist_ok=True)

        eclat_result = run_alert_group_eclat_job(
            alert_groups_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_itemset_size,
            target_label="benign",
            run_dir=job_dir,
        )
        eclat_raw = eclat_result.mined_df.copy()

        # Pre-grouped scenarios (e.g. CSCAS/suricata_grouped) have no
        # intra-group alert order — sorted_items is always empty — so
        # PrefixSpan would mine zero sequences and the downstream confidence
        # scoring/sort step chokes on the resulting columnless DataFrame.
        # Skip sequence mining entirely in that case.
        has_sequence_data = any(r.get("sorted_items") for r in rows)
        if has_sequence_data:
            seq_result = run_alert_group_prefixspan_job(
                alert_groups_path=tx_path,
                scenario_name=scenario,
                run_name=run_name,
                min_support=min_support,
                max_len=max_seq_len,
                target_label="benign",
                run_dir=job_dir,
            )
            seq_raw = seq_result.mined_df.copy()
        else:
            print(
                "  [skip] All alert_groups have empty sorted_items — "
                "skipping sequence mining."
            )
            seq_raw = pd.DataFrame()

        eclat_raw.to_parquet(eclat_raw_path, index=False)
        seq_raw.to_parquet(seq_raw_path, index=False)

    eclat_filtered = eclat_raw.copy()
    seq_filtered = seq_raw.copy()

    if filter_config is not None:
        from thesis.config import load_mining_filter_config

        fc = load_mining_filter_config(filter_config)

        if mode == "smart":
            eclat_filtered = _smart_filter_itemsets(
                eclat_raw, fc.itemsets, f_attack_scenario, smart_ratio_threshold
            )
            seq_filtered = _smart_filter_sequences(
                seq_raw, fc.item_sequences, f_attack_scenario, smart_ratio_threshold
            )
        else:
            from thesis.mining.util import filter_mined_itemsets, filter_mined_sequences

            f = fc.itemsets
            eclat_filtered = filter_mined_itemsets(
                eclat_raw,
                min_k=f.min_k,
                max_k=f.max_k,
                min_support_count=f.min_support_count,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                max_overlap=f.max_overlap,
                remove_subsumed=f.remove_subsumed,
            )

            f = fc.item_sequences
            seq_filtered = filter_mined_sequences(
                seq_raw,
                min_k=f.min_k,
                min_support_count=f.min_support_count,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                min_lift=f.min_lift,
                max_overlap=f.max_overlap,
                remove_subsumed=f.remove_subsumed,
            )

    eclat_filtered.to_parquet(eclat_filt_path, index=False)
    seq_filtered.to_parquet(seq_filt_path, index=False)
    filter_hash_path.write_text(_filter_fingerprint())

    return eclat_raw, eclat_filtered, seq_raw, seq_filtered


def _feature_id_from_itemset_row(row: pd.Series) -> str:
    if "itemset" in row.index:
        return "itemset:" + "|".join(sorted(str(x) for x in row["itemset"]))
    if "itemset_str" in row.index:
        return "itemset:" + str(row["itemset_str"])
    return "?"


def _feature_id_from_seq_row(row: pd.Series) -> str:
    if "sequence_str" in row.index:
        return "seq:" + str(row["sequence_str"])
    if "sequence" in row.index:
        return "seq:" + str(row["sequence"])
    return "?"


def _feature_set_from_dfs(eclat_df: pd.DataFrame, seq_df: pd.DataFrame) -> set[str]:
    ids: set[str] = set()
    for _, row in eclat_df.iterrows():
        ids.add(_feature_id_from_itemset_row(row))
    for _, row in seq_df.iterrows():
        ids.add(_feature_id_from_seq_row(row))
    return ids


# ---------------------------------------------------------------------------
# Window result record
# ---------------------------------------------------------------------------


class WindowResult(NamedTuple):
    scenario: str
    mode: str
    gran: float
    win_idx: int
    win_start_frac: float
    win_end_frac: float
    n_alert_groups: int
    n_attack_alert_groups: int
    features_raw: set[str]
    features_filtered: set[str]
    eclat_raw: pd.DataFrame
    eclat_filtered: pd.DataFrame
    seq_raw: pd.DataFrame
    seq_filtered: pd.DataFrame


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------


def _compute_f_attack(rows: list[dict]) -> float:
    """Fraction of alert_groups labelled attack."""
    n = len(rows)
    if n == 0:
        return 0.0
    n_attack = sum(1 for r in rows if r.get("group_label") == "attack")
    return n_attack / n


def table1_stability(results: list[WindowResult]) -> pd.DataFrame:
    """
    Within-scenario, within-mode, within-granularity: how does the feature set
    change between consecutive windows?
    """
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.mode, r.gran)].append(r)

    for (scenario, mode, gran), wins in groups.items():
        wins = sorted(wins, key=lambda w: w.win_idx)
        prev_set: set[str] | None = None
        for w in wins:
            cur = w.features_filtered
            if prev_set is None:
                row = {
                    "scenario": scenario,
                    "mode": mode,
                    "gran": gran,
                    "window": w.win_idx,
                    "win_range": f"{w.win_start_frac:.0%}–{w.win_end_frac:.0%}",
                    "n_tx": w.n_alert_groups,
                    "n_benign_tx": w.n_alert_groups - w.n_attack_alert_groups,
                    "n_attack_tx": w.n_attack_alert_groups,
                    "n_features": len(cur),
                    "n_added": None,
                    "n_removed": None,
                    "n_unchanged": None,
                    "jaccard": None,
                }
            else:
                added = cur - prev_set
                removed = prev_set - cur
                unchanged = cur & prev_set
                union = cur | prev_set
                jaccard = len(unchanged) / len(union) if union else float("nan")
                row = {
                    "scenario": scenario,
                    "mode": mode,
                    "gran": gran,
                    "window": w.win_idx,
                    "win_range": f"{w.win_start_frac:.0%}–{w.win_end_frac:.0%}",
                    "n_tx": w.n_alert_groups,
                    "n_benign_tx": w.n_alert_groups - w.n_attack_alert_groups,
                    "n_attack_tx": w.n_attack_alert_groups,
                    "n_features": len(cur),
                    "n_added": len(added),
                    "n_removed": len(removed),
                    "n_unchanged": len(unchanged),
                    "jaccard": round(jaccard, 4),
                }
            rows.append(row)
            prev_set = cur
    return pd.DataFrame(rows)


def table2_sharing(results: list[WindowResult]) -> pd.DataFrame:
    """
    Cross-scenario: for each feature, which scenarios mine it and in how many
    windows?  Returns one row per (feature, scenario, mode, gran).
    """
    rows = []
    for r in results:
        for feat in r.features_filtered:
            rows.append(
                {
                    "feature": feat,
                    "scenario": r.scenario,
                    "mode": r.mode,
                    "gran": r.gran,
                    "window": r.win_idx,
                }
            )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Aggregate: count windows each (feature, scenario, mode, gran) appears in
    agg = (
        df.groupby(["feature", "scenario", "mode", "gran"])
        .agg(n_windows=("window", "count"))
        .reset_index()
    )
    # Count how many scenarios mine this feature (at the given mode/gran)
    n_scenarios = (
        agg.groupby(["feature", "mode", "gran"])
        .agg(n_scenarios=("scenario", "nunique"))
        .reset_index()
    )
    agg = agg.merge(n_scenarios, on=["feature", "mode", "gran"], how="left")
    agg = agg.sort_values(
        ["mode", "gran", "n_scenarios", "n_windows"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return agg


def table3_convergence(results: list[WindowResult]) -> pd.DataFrame:
    """
    Cross-scenario: Jaccard similarity between consecutive windows per
    (scenario, mode, gran), to compare convergence trajectories.
    """
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.mode, r.gran)].append(r)

    for (scenario, mode, gran), wins in groups.items():
        wins = sorted(wins, key=lambda w: w.win_idx)
        for i in range(1, len(wins)):
            prev = wins[i - 1].features_filtered
            cur = wins[i].features_filtered
            union = prev | cur
            jaccard = len(prev & cur) / len(union) if union else float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "mode": mode,
                    "gran": gran,
                    "transition": f"w{i - 1}→w{i}",
                    "win_range": f"{wins[i - 1].win_start_frac:.0%}–{wins[i].win_end_frac:.0%}",
                    "jaccard": round(jaccard, 4),
                    "n_prev": len(prev),
                    "n_cur": len(cur),
                }
            )
    return pd.DataFrame(rows)


def table4_dropped_diagnosis(
    results: list[WindowResult],
    all_rows_by_scenario: dict[str, list[dict]],
) -> pd.DataFrame:
    """
    For features that were mined but then dropped by the filter (present in raw
    but not in filtered), check whether support_attack ≈ support_benign × f_attack
    (uniform noise → mislabelled benign) or support_attack >> expectation
    (attack-concentrated → correctly dropped).
    """
    rows = []
    for r in results:
        # Skip benign-only windows — support_attack is trivially 0 there so the
        # ratio formula produces misleading verdicts.
        if r.n_attack_alert_groups == 0:
            continue

        dropped_ids = r.features_raw - r.features_filtered

        # Only work on itemsets (eclat) because sequences don't always carry
        # support_attack in a consistent single column
        raw_df = r.eclat_raw.copy()
        if raw_df.empty or "support_attack" not in raw_df.columns:
            continue
        if "support_benign" not in raw_df.columns:
            continue

        # Build id → row mapping for raw itemsets
        raw_df["_feat_id"] = raw_df.apply(_feature_id_from_itemset_row, axis=1)
        dropped_raw = raw_df[raw_df["_feat_id"].isin(dropped_ids)]

        scenario_rows = all_rows_by_scenario[r.scenario]
        f_attack = _compute_f_attack(scenario_rows)

        for _, row in dropped_raw.iterrows():
            sb = row.get("support_benign", 0.0)
            sa = row.get("support_attack", 0.0)
            expected_sa = sb * f_attack
            ratio = (sa / expected_sa) if expected_sa > 1e-9 else float("nan")
            verdict = (
                "uniform_noise"
                if (not pd.isna(ratio) and ratio <= 1.5)
                else "attack_concentrated"
            )
            rows.append(
                {
                    "scenario": r.scenario,
                    "mode": r.mode,
                    "gran": r.gran,
                    "window": r.win_idx,
                    "feature": row["_feat_id"],
                    "support_benign": round(sb, 4),
                    "support_attack": round(sa, 4),
                    "f_attack": round(f_attack, 4),
                    "expected_support_attack": round(expected_sa, 4),
                    "ratio_actual_over_expected": round(ratio, 3)
                    if not pd.isna(ratio)
                    else None,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def table5_benign_vs_mixed_delta(
    results: list[WindowResult],
) -> pd.DataFrame:
    """
    For each (scenario, gran, window), compare features mined in benign mode
    vs. mixed mode.  Reports features that appear only in mixed (attack-dependent
    features) and features only in benign.
    """
    rows = []
    # Index by (scenario, gran, win_idx, mode)
    idx: dict[tuple, set[str]] = {}
    for r in results:
        idx[(r.scenario, r.gran, r.win_idx, r.mode)] = r.features_filtered

    seen: set[tuple] = set()
    for r in results:
        key = (r.scenario, r.gran, r.win_idx)
        if key in seen:
            continue
        seen.add(key)
        benign_feats = idx.get((*key, "benign"), set())
        mixed_feats = idx.get((*key, "mixed"), set())
        if not benign_feats and not mixed_feats:
            continue
        only_mixed = mixed_feats - benign_feats
        only_benign = benign_feats - mixed_feats
        shared = benign_feats & mixed_feats
        rows.append(
            {
                "scenario": r.scenario,
                "gran": r.gran,
                "window": r.win_idx,
                "win_range": f"{r.win_start_frac:.0%}–{r.win_end_frac:.0%}",
                "n_benign_only": len(only_benign),
                "n_mixed_only": len(only_mixed),
                "n_shared": len(shared),
                "n_benign_total": len(benign_feats),
                "n_mixed_total": len(mixed_feats),
            }
        )
    return pd.DataFrame(rows)


def table6_persistence(results: list[WindowResult]) -> pd.DataFrame:
    """
    For each feature (within a scenario/mode/gran), count in how many windows
    it appears.  Summarised as a histogram table.
    """
    rows = []
    groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        groups[(r.scenario, r.mode, r.gran)].append(r)

    for (scenario, mode, gran), wins in groups.items():
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
                    "mode": mode,
                    "gran": gran,
                    "n_windows_present": n_wins_present,
                    "pct_of_total_windows": round(n_wins_present / n_windows, 3),
                    "n_features_with_this_persistence": n_features,
                }
            )
    return pd.DataFrame(rows)


def table7_cross_granularity(results: list[WindowResult]) -> pd.DataFrame:
    """
    Does a feature mined at coarse granularity (e.g. gran=0.33, window 0→33%)
    also appear in the fine sub-windows that cover the same range?

    For each coarse window, we find all fine windows that fall within its
    fractional span and check what fraction of the coarse features also appear
    in ALL / ANY of those fine windows.
    """
    rows = []
    # Group by (scenario, mode)
    sm_groups: dict[tuple, list[WindowResult]] = defaultdict(list)
    for r in results:
        sm_groups[(r.scenario, r.mode)].append(r)

    for (scenario, mode), wins in sm_groups.items():
        grans = sorted({w.gran for w in wins})
        if len(grans) < 2:
            continue

        for coarse_gran in grans[1:]:  # every gran bigger than the smallest
            fine_gran = grans[0]  # compare against finest granularity
            coarse_wins = [w for w in wins if w.gran == coarse_gran]
            fine_wins = [w for w in wins if w.gran == fine_gran]

            for cw in coarse_wins:
                # fine windows overlapping this coarse window
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
                fine_intersection = (
                    set(overlapping[0].features_filtered) if overlapping else set()
                )
                for fw in overlapping[1:]:
                    fine_intersection &= fw.features_filtered

                n_in_any = len(coarse_feats & fine_union)
                n_in_all = len(coarse_feats & fine_intersection)
                n_coarse = len(coarse_feats)

                rows.append(
                    {
                        "scenario": scenario,
                        "mode": mode,
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


def _summary_lines(
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t5: pd.DataFrame,
    t6: pd.DataFrame,
    t7: pd.DataFrame,
) -> str:
    lines = []

    if not t1.empty:
        avg_j = t1["jaccard"].dropna().mean()
        min_j = t1["jaccard"].dropna().min()
        lines.append(
            f"Stability (T1): mean Jaccard={avg_j:.3f}, min={min_j:.3f} across all consecutive windows"
        )

    if not t2.empty:
        shared = t2[t2["n_scenarios"] > 1]
        lines.append(
            f"Sharing (T2): {len(t2['feature'].unique())} unique features; "
            f"{len(shared['feature'].unique())} appear in >1 scenario"
        )

    if not t4.empty:
        uniform = (t4["verdict"] == "uniform_noise").sum()
        attack = (t4["verdict"] == "attack_concentrated").sum()
        lines.append(
            f"Dropped diagnosis (T4): {uniform} uniform-noise drops, {attack} attack-concentrated drops"
        )

    if not t5.empty:
        avg_mixed_only = t5["n_mixed_only"].mean()
        lines.append(
            f"Benign/mixed delta (T5): on average {avg_mixed_only:.1f} features appear only in mixed mode per window"
        )

    if not t6.empty:
        # Features present in every window
        max_per_gran = t6.groupby(["scenario", "mode", "gran"])[
            "n_windows_present"
        ].max()
        lines.append(
            f"Persistence (T6): max windows-present per (scenario,mode,gran): {max_per_gran.max()}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine features in temporal windows and analyse stability/sharing.",
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
        help="Run for all scenarios found under artifacts/cache/.",
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
        "--modes",
        nargs="+",
        choices=["benign", "mixed", "smart"],
        default=["benign", "mixed"],
        help=(
            "AlertGroup selection and filter strategy. "
            "'benign': strip attacks before mining. "
            "'mixed': keep all alert_groups, standard support_diff filter. "
            "'smart': keep all alert_groups, rescue features whose attack support "
            "is explained by background noise (ratio ≤ --smart-ratio). "
            "Default: benign mixed"
        ),
    )
    parser.add_argument(
        "--smart-ratio",
        type=float,
        default=1.5,
        metavar="R",
        help=(
            "Ratio threshold for smart mode: features with "
            "support_attack / (support_benign × f_attack) ≤ R are kept even if "
            "they fail the support_diff filter. Default: 1.5"
        ),
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to mining filter config YAML. If omitted, all mined patterns pass.",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Minimum support for mining (default: 0.05).",
    )
    parser.add_argument(
        "--max-itemset-size",
        type=int,
        default=3,
        help="Max itemset size for Eclat (default: 3).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=5,
        help="Max sequence length for PrefixSpan (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override. Default: artifacts/experiments/mining_window_sweep/<ts>/",
    )
    parser.add_argument(
        "--grouping-method",
        type=str,
        default="fixed_window",
        metavar="METHOD",
        help=(
            "Grouping method subdir to read alert_groups from, i.e. "
            "artifacts/cache/<scenario>/groups/<METHOD>/alert_groups/alert_groups_raw.json. "
            "Default: fixed_window"
        ),
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

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filter_tag = (
        args.filter_config.stem if args.filter_config is not None else "nofilter"
    )
    modes_tag = "-".join(args.modes)
    gran_tag = "-".join(f"{g:g}" for g in args.granularities)
    run_tag = f"{ts}_{filter_tag}_{modes_tag}_gran-{gran_tag}"
    if args.output_dir is not None:
        out_dir: Path = args.output_dir
    else:
        datasets = {dataset_for_scenario(s) or "unknown" for s in scenarios}
        if len(datasets) > 1:
            print(
                f"[warn] scenarios span multiple datasets {sorted(datasets)} — "
                "grouping this run's output under 'mixed'"
            )
            dataset_tag = "mixed"
        else:
            dataset_tag = datasets.pop()
        out_dir = _EXPERIMENTS_DIR / dataset_tag / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    win_mining_cache = _REPO / "artifacts" / "cache" / "_window_mining"
    win_mining_cache.mkdir(parents=True, exist_ok=True)

    filter_config: Path | None = args.filter_config
    if filter_config is not None and not filter_config.is_absolute():
        filter_config = _REPO / filter_config

    # ------------------------------------------------------------------
    # Phase 1: mine all windows
    # ------------------------------------------------------------------
    all_results: list[WindowResult] = []
    all_rows_by_scenario: dict[str, list[dict]] = {}

    for scenario in args.scenarios:
        print(f"\n[scenario={scenario}] Loading alert_groups...")
        raw_rows = _load_raw_alert_groups(
            scenario, grouping_method=args.grouping_method
        )
        all_rows_by_scenario[scenario] = raw_rows
        f_attack_scenario = _compute_f_attack(raw_rows)
        print(f"  {len(raw_rows)} alert_groups total, f_attack={f_attack_scenario:.4f}")

        for mode in args.modes:
            # smart uses the full mixed alert_group set but a smarter filter
            rows = _strip_attacks(raw_rows) if mode == "benign" else raw_rows
            n_label = len(rows)
            print(f"\n  [mode={mode}] {n_label} alert_groups")

            for gran in args.granularities:
                n_total = len(rows)
                win_size = max(1, int(gran * n_total))
                n_windows = max(1, n_total // win_size)

                print(
                    f"  [gran={gran:.0%}] {n_windows} windows of ~{win_size} alert_groups each"
                )

                for win_idx in range(n_windows):
                    start = win_idx * win_size
                    # last window absorbs remainder
                    end = (
                        (win_idx + 1) * win_size if win_idx < n_windows - 1 else n_total
                    )
                    win_rows = rows[start:end]
                    win_start_frac = start / n_total
                    win_end_frac = end / n_total

                    n_attack = sum(
                        1 for r in win_rows if r.get("group_label") == "attack"
                    )

                    print(
                        f"    window {win_idx}: [{win_start_frac:.0%}, {win_end_frac:.0%})"
                        f"  {len(win_rows)} tx  {n_attack} attack",
                        end=" ... ",
                        flush=True,
                    )

                    eclat_raw, eclat_filt, seq_raw, seq_filt = _run_mining_for_window(
                        rows=win_rows,
                        scenario=scenario,
                        mode=mode,
                        gran=gran,
                        win_idx=win_idx,
                        min_support=args.min_support,
                        smart_ratio_threshold=args.smart_ratio,
                        f_attack_scenario=f_attack_scenario,
                        max_itemset_size=args.max_itemset_size,
                        max_seq_len=args.max_seq_len,
                        filter_config=filter_config,
                        cache_dir=win_mining_cache,
                    )

                    features_raw = _feature_set_from_dfs(eclat_raw, seq_raw)
                    features_filtered = _feature_set_from_dfs(eclat_filt, seq_filt)

                    print(f"raw={len(features_raw)} filtered={len(features_filtered)}")

                    all_results.append(
                        WindowResult(
                            scenario=scenario,
                            mode=mode,
                            gran=gran,
                            win_idx=win_idx,
                            win_start_frac=win_start_frac,
                            win_end_frac=win_end_frac,
                            n_alert_groups=len(win_rows),
                            n_attack_alert_groups=n_attack,
                            features_raw=features_raw,
                            features_filtered=features_filtered,
                            eclat_raw=eclat_raw,
                            eclat_filtered=eclat_filt,
                            seq_raw=seq_raw,
                            seq_filtered=seq_filt,
                        )
                    )

                    # Save window features CSV for inspection
                    wf_rows = [
                        {"feature": f, "source": "raw"} for f in features_raw
                    ] + [
                        {"feature": f, "source": "filtered"} for f in features_filtered
                    ]
                    wf_df = pd.DataFrame(wf_rows).drop_duplicates()
                    wf_dir = out_dir / "window_features"
                    wf_dir.mkdir(parents=True, exist_ok=True)
                    wf_name = (
                        f"window_features_{scenario}_{mode}_{gran:.2f}_{win_idx}.csv"
                    )
                    wf_df.to_csv(wf_dir / wf_name, index=False)

    # ------------------------------------------------------------------
    # Phase 2: analysis tables
    # ------------------------------------------------------------------
    print("\n\nComputing analysis tables...")

    t1 = table1_stability(all_results)
    t2 = table2_sharing(all_results)
    t3 = table3_convergence(all_results)
    t4 = table4_dropped_diagnosis(all_results, all_rows_by_scenario)
    t5 = table5_benign_vs_mixed_delta(all_results)
    t6 = table6_persistence(all_results)
    t7 = table7_cross_granularity(all_results)

    # Print tables
    _print_table(t1, "T1 — Within-scenario stability (consecutive Jaccard)")
    _print_table(
        t2[t2["n_scenarios"] > 1] if not t2.empty else t2,
        "T2 — Cross-scenario feature sharing (features in >1 scenario)",
    )
    _print_table(t3, "T3 — Convergence curves (Jaccard per transition)")
    _print_table(t4, "T4 — Dropped feature diagnosis")
    _print_table(t5, "T5 — Benign vs mixed delta per window")
    _print_table(t6, "T6 — Feature persistence histogram")
    _print_table(t7, "T7 — Cross-granularity consistency")

    # Save CSVs
    for df, name in [
        (t1, "table1_stability"),
        (t2, "table2_sharing"),
        (t3, "table3_convergence"),
        (t4, "table4_dropped_diagnosis"),
        (t5, "table5_benign_vs_mixed"),
        (t6, "table6_persistence"),
        (t7, "table7_cross_granularity"),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)

    # Summary text
    summary = _summary_lines(t1, t2, t3, t4, t5, t6, t7)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        textwrap.dedent(f"""\
        Mining Window Sweep — {ts}
        Scenarios : {", ".join(args.scenarios)}
        Modes     : {", ".join(args.modes)}
        Granularities: {", ".join(f"{g:.0%}" for g in args.granularities)}
        Filter config: {filter_config or "(none)"}
        Min support  : {args.min_support}

        {summary}
        """)
    )

    print(f"\n\nOutputs saved to: {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
