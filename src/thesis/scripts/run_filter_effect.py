"""
Filter effect experiment: does the model learn FP-reducing features on its own,
or does it need explicit mining filters to do so?

For a fixed scenario and grouping method, runs one symbolic experiment per filter
condition and compares:
  - Feature pipeline funnel (mined → filter → final → nonzero-coeff)
  - Performance: AUC, precision, recall, FP count
  - Feature overlap across conditions (Jaccard of nonzero-coeff features)
  - Feature type and source-label composition of what the model actually uses

Filter conditions (in order of increasing strictness):
  none          -- all mined patterns, no filter (model must self-select)
  default       -- minimal noise removal (min_support=10, no discrimination)
  discriminative-- moderate filtering (support_diff≥0.10, remove_subsumed)
  strict        -- aggressive filtering (support_diff≥0.20, lift≥2.0)
  benign_focused-- keeps only patterns that are predominantly benign-traffic

Output (under artifacts/experiments/run_filter_effect/filter_effect_<ts>/<scenario>/):
    filter_effect_<ts>.json      -- combined results for all conditions
    baseline_<ts>.json           -- baseline result (no symbolic features)
    symbolic_<cond>_<ts>.json    -- per-condition symbolic result
    analysis_<ts>.txt            -- console output log
    plots/
        feature_funnel.png
        performance.png
        fp_analysis.png
        feature_overlap.png
        feature_type_breakdown.png

Usage:
    python src/thesis/scripts/run_filter_effect.py <scenario> \\
        [--grouping fixed_window]   # grouping method (default: fixed_window)
        [--conditions none,default,discriminative,strict,benign_focused]
        [--force]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.config import GroupingConfig
from thesis.experiments.baseline import run_baseline_experiment
from thesis.experiments.symbolic import run_symbolic_experiment
from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR
from thesis.schemas.experiments import (
    BaselineExperimentConfig,
    ExperimentResult,
    SymbolicExperimentConfig,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_OUTPUT_BASE = _REPO / "artifacts" / "experiments" / "run_filter_effect"
_CONFIGS_DIR = _REPO / "src" / "thesis" / "configs"

_FILTER_CONFIG_FILES: dict[str, Path | None] = {
    "none": None,
    "default": _CONFIGS_DIR / "mining_filters_default.yaml",
    "discriminative": _CONFIGS_DIR / "mining_filters_discriminative.yaml",
    "strict": _CONFIGS_DIR / "mining_filters_strict.yaml",
    "benign_focused": _CONFIGS_DIR / "mining_filters_benign_focused.yaml",
}

_ALL_CONDITIONS = list(_FILTER_CONFIG_FILES.keys())

_COLORS = {
    "none": "#BBBBBB",
    "default": "#9EC8E8",
    "discriminative": "#4C72B0",
    "strict": "#1A3A6E",
    "benign_focused": "#55A868",
}

_LABELS = {
    "none": "None\n(all mined)",
    "default": "Default\n(noise only)",
    "discriminative": "Discriminative\n(moderate)",
    "strict": "Strict\n(aggressive)",
    "benign_focused": "Benign-\nfocused",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._file = log_path.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _grouping_config(mode: str) -> GroupingConfig:
    return GroupingConfig(mode=mode)


def _run_condition(
    scenario: str,
    condition: str,
    grouping: GroupingConfig,
    cache_dir: Path,
    grouping_cache_dir: Path,
    results_dir: Path,
    transactions_dir: Path,
) -> ExperimentResult:
    return run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir,
            grouping=grouping,
            filter_config=_FILTER_CONFIG_FILES[condition],
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            grouping_cache_dir=grouping_cache_dir,
            results_dir=results_dir,
            transactions_dir=transactions_dir,
        )
    )


# ---------------------------------------------------------------------------
# Feature extraction helpers (shared with run_feature_overlap_analysis)
# ---------------------------------------------------------------------------


def _i(v: Any) -> int:
    return int(v) if v is not None else 0


def _feature_funnel(sym_path: Path) -> dict:
    with sym_path.open() as f:
        sym = json.load(f)
    m = sym.get("mining", {})
    met = sym.get("metrics", {})
    top_coeff = met.get("top_feature_importances", {}).get("by_coefficient", {})
    top_perm = met.get("top_feature_importances", {}).get("by_permutation", {})

    n_mined = (
        _i(m.get("n_itemsets_mined"))
        + _i(m.get("n_sequences_mined"))
        + _i(m.get("n_or_mined"))
    )

    # After abstraction: fall back to n_mined when abstraction was skipped (fields are None).
    n_abs_parts = (
        _i(m.get("n_itemsets_after_abstraction"))
        + _i(m.get("n_sequences_after_abstraction"))
        + _i(m.get("n_or_after_abstraction"))
    )
    n_after_abstraction = n_abs_parts if n_abs_parts > 0 else n_mined

    # n_candidate_features = itemsets_after_filter + sequences_after_filter + OR_after_abstraction.
    # OR features bypass the filter step entirely, so this is the honest post-filter total.
    n_after_filter = _i(m.get("n_candidate_features"))

    return {
        "n_mined": n_mined,
        "n_after_abstraction": n_after_abstraction,
        "n_after_filter": n_after_filter,
        "n_final": _i(m.get("n_features_final")),
        "n_nonzero_coeff": sum(1 for v in top_coeff.values() if v["importance"] > 0),
        "n_nonzero_perm": sum(1 for v in top_perm.values() if v["importance"] > 0),
    }


def _top_feature_names(sym_path: Path, k: int, by: str = "by_coefficient") -> set[str]:
    with sym_path.open() as f:
        sym = json.load(f)
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    ranked = sorted(top.items(), key=lambda kv: kv[1]["importance"], reverse=True)
    return {name for name, info in ranked[:k] if info["importance"] > 0}


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _mining_type_breakdown(sym_path: Path) -> dict[str, int]:
    with sym_path.open() as f:
        sym = json.load(f)
    top = (
        sym.get("metrics", {})
        .get("top_feature_importances", {})
        .get("by_coefficient", {})
    )
    counts: dict[str, int] = {}
    for info_dict in top.values():
        if info_dict["importance"] <= 0:
            continue
        mtype = info_dict.get("feature_info", {}).get("mining_type", "base")
        counts[mtype] = counts.get(mtype, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _result_to_dict(r: ExperimentResult) -> dict:
    return {
        "schema_name": r.schema_name,
        "schema_version": r.schema_version,
        "grouping_mode": r.grouping_mode,
        "auc": r.auc,
        "n_transactions": r.n_transactions,
        "n_mixed_dropped": r.n_mixed_dropped,
        "n_features": r.n_features,
        "metrics": r.metrics,
        "results_file": str(r.results_file),
    }


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------


def _print_funnel_table(conditions: list[str], funnels: dict[str, dict]) -> None:
    stages = [
        ("n_mined", "Mined total"),
        ("n_after_abstraction", "After abstraction"),
        ("n_after_filter", "After filter (+OR)"),
        ("n_final", "Final (dedup)"),
        ("n_nonzero_coeff", "Nonzero coeff"),
        ("n_nonzero_perm", "Nonzero perm"),
    ]
    cw = 16
    w = 26 + cw * len(conditions)
    print("\n" + "═" * w)
    print("  FEATURE PIPELINE FUNNEL")
    print("─" * w)
    print(f"  {'Stage':<24}" + "".join(f"{c:>{cw}}" for c in conditions))
    print("─" * w)
    for key, label in stages:
        row = f"  {label:<24}" + "".join(
            f"{funnels.get(c, {}).get(key, 0):>{cw},}" for c in conditions
        )
        print(row)
    print("═" * w + "\n")


def _print_performance_table(
    conditions: list[str],
    results: dict[str, ExperimentResult],
    baseline: ExperimentResult,
) -> None:
    cw = 14
    w = 26 + cw * (len(conditions) + 1)
    print("═" * w)
    print("  PERFORMANCE  (baseline for comparison)")
    print("─" * w)
    print(
        f"  {'Metric':<24}"
        + f"{'baseline':>{cw}}"
        + "".join(f"{c:>{cw}}" for c in conditions)
    )
    print("─" * w)
    for label, key in [
        ("AUC", "auc"),
        ("Train AUC", "train_auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Balanced acc.", "balanced_accuracy"),
    ]:
        base_v = baseline.metrics.get(
            key, baseline.auc if key == "auc" else float("nan")
        )
        row = f"  {label:<24}{base_v:>{cw}.4f}"
        for c in conditions:
            met = results[c].metrics
            v = met.get(key, results[c].auc if key == "auc" else float("nan"))
            row += f"{v:>{cw}.4f}"
        print(row)
    print("─" * w)
    for label, key in [("TP", "tp"), ("FP", "fp"), ("TN", "tn"), ("FN", "fn")]:
        base_v = baseline.metrics.get(key, 0)
        row = f"  {label:<24}{int(base_v):>{cw},}"
        for c in conditions:
            v = results[c].metrics.get(key, 0)
            row += f"{int(v):>{cw},}"
        print(row)
    print("─" * w)
    # FP delta vs baseline
    row = f"  {'FP delta vs baseline':<24}{'':>{cw}}"
    for c in conditions:
        delta = int(results[c].metrics.get("fp", 0)) - int(
            baseline.metrics.get("fp", 0)
        )
        row += f"{delta:>+{cw},}"
    print(row)
    print("═" * w + "\n")


def _print_overlap_table(
    conditions: list[str], sym_paths: dict[str, Path], k: int
) -> None:
    feature_sets = {c: _top_feature_names(sym_paths[c], k) for c in conditions}
    print(f"  TOP-{k} FEATURE JACCARD OVERLAP (nonzero coeff, by_coefficient)")
    w = 10 + 9 * len(conditions)
    print("─" * w)
    print(f"  {'':8}" + "".join(f"{c[:8]:>9}" for c in conditions))
    for ca in conditions:
        row = f"  {ca[:8]:<8}"
        for cb in conditions:
            row += f"{_jaccard(feature_sets[ca], feature_sets[cb]):>9.3f}"
        print(row)
    print()
    if len(conditions) > 1:
        shared = set.intersection(*feature_sets.values())
        print(f"  Shared by ALL conditions ({len(shared)} features):")
        for name in sorted(shared):
            print(f"    {name}")
    print()


def _print_type_breakdown(conditions: list[str], sym_paths: dict[str, Path]) -> None:
    all_types: set[str] = set()
    bds = {}
    for c in conditions:
        bd = _mining_type_breakdown(sym_paths[c])
        bds[c] = bd
        all_types |= set(bd.keys())
    cw = 14
    w = 26 + cw * len(conditions)
    print("═" * w)
    print("  NONZERO-COEFF FEATURE TYPE BREAKDOWN")
    print("─" * w)
    print(f"  {'Type':<24}" + "".join(f"{c:>{cw}}" for c in conditions))
    print("─" * w)
    for mtype in sorted(all_types):
        row = f"  {mtype:<24}" + "".join(
            f"{bds[c].get(mtype, 0):>{cw},}" for c in conditions
        )
        print(row)
    print("═" * w + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_funnel(
    conditions: list[str], funnels: dict[str, dict], out_dir: Path
) -> None:
    stages = [
        ("n_mined", "Mined"),
        ("n_after_abstraction", "After\nabstraction"),
        ("n_after_filter", "After filter\n(+OR pass-through)"),
        ("n_final", "Final\n(dedup)"),
        ("n_nonzero_coeff", "Learned\n(coeff>0)"),
    ]
    x = np.arange(len(stages))
    w = 0.15
    offsets = [w * (i - (len(conditions) - 1) / 2) for i in range(len(conditions))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(conditions):
        vals = [funnels.get(cond, {}).get(key, 0) for key, _ in stages]
        bars = ax.bar(x + offsets[i], vals, w, label=cond, color=_COLORS[cond])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:,}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stages])
    ax.set_ylabel("Feature count (log scale)")
    ax.set_title("Feature pipeline funnel by filter condition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "feature_funnel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_performance(
    conditions: list[str],
    results: dict[str, ExperimentResult],
    baseline: ExperimentResult,
    out_dir: Path,
) -> None:
    metrics = [
        ("auc", "AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    all_labels = ["baseline"] + conditions
    x = np.arange(len(metrics))
    w = 0.12
    offsets = [w * (i - (len(all_labels) - 1) / 2) for i in range(len(all_labels))]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(all_labels):
        result = baseline if label == "baseline" else results[label]
        vals = []
        for key, _ in metrics:
            v = result.metrics.get(key, result.auc if key == "auc" else float("nan"))
            vals.append(v)
        color = "#5B9BD5" if label == "baseline" else _COLORS[label]
        bars = ax.bar(x + offsets[i], vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val == val and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score")
    ax.set_title("Detection performance by filter condition")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "performance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_fp_analysis(
    conditions: list[str],
    results: dict[str, ExperimentResult],
    baseline: ExperimentResult,
    out_dir: Path,
) -> None:
    all_labels = ["baseline"] + conditions
    fp_vals = []
    prec_vals = []
    for label in all_labels:
        r = baseline if label == "baseline" else results[label]
        fp_vals.append(r.metrics.get("fp", 0))
        prec_vals.append(r.metrics.get("precision", float("nan")))

    x = np.arange(len(all_labels))
    colors = ["#5B9BD5"] + [_COLORS[c] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    bars = ax1.bar(x, fp_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, fp_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(fp_vals) * 0.01,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=15, ha="right")
    ax1.set_ylabel("False positives")
    ax1.set_title("FP count by filter condition")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, prec_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars2, prec_vals):
        if val == val:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=15, ha="right")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision by filter condition")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fp_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_overlap_heatmap(
    conditions: list[str], sym_paths: dict[str, Path], k: int, out_dir: Path
) -> None:
    feature_sets = {c: _top_feature_names(sym_paths[c], k) for c in conditions}
    n = len(conditions)
    matrix = np.zeros((n, n))
    for i, ca in enumerate(conditions):
        for j, cb in enumerate(conditions):
            matrix[i, j] = _jaccard(feature_sets[ca], feature_sets[cb])

    fig, ax = plt.subplots(figsize=(5 + n * 0.5, 4 + n * 0.3))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticklabels(conditions)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black" if matrix[i, j] < 0.7 else "white",
            )
    ax.set_title(f"Top-{k} feature Jaccard overlap (nonzero coeff)")
    fig.tight_layout()
    out = out_dir / "feature_overlap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_type_breakdown(
    conditions: list[str], sym_paths: dict[str, Path], out_dir: Path
) -> None:
    all_types_set: set[str] = set()
    bds = {}
    for c in conditions:
        bd = _mining_type_breakdown(sym_paths[c])
        bds[c] = bd
        all_types_set |= set(bd.keys())
    all_types = sorted(all_types_set)
    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    for mtype in all_types:
        vals = np.array([bds[c].get(mtype, 0) for c in conditions], dtype=float)
        bars = ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=mtype,
            color=type_colors.get(mtype, "#BBBBBB"),
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Nonzero-coeff features")
    ax.set_title("Learned feature type breakdown by filter condition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "feature_type_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the effect of mining filters on model size and FP performance."
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--grouping",
        default="fixed_window",
        choices=["fixed_window", "fixed_window_host", "time_delta", "time_delta_host"],
        help="Grouping method to use (default: fixed_window).",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(_ALL_CONDITIONS),
        help=f"Comma-separated filter conditions to run (default: all). Options: {_ALL_CONDITIONS}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top features for Jaccard overlap (default: 25).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results already exist.",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    unknown = [c for c in conditions if c not in _FILTER_CONFIG_FILES]
    if unknown:
        parser.error(f"Unknown conditions: {unknown}. Options: {_ALL_CONDITIONS}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_BASE / f"filter_effect_{ts}" / args.scenario
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"analysis_{ts}.txt"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        _main_body(args, conditions, ts, run_dir, plots_dir)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _main_body(
    args: argparse.Namespace,
    conditions: list[str],
    ts: str,
    run_dir: Path,
    plots_dir: Path,
) -> None:
    scenario = args.scenario
    grouping = _grouping_config(args.grouping)

    cache_dir = CACHE_DIR / scenario
    groups_cache_dir = CACHE_DIR / "groups" / scenario / args.grouping
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Filter effect experiment: {scenario}  grouping={args.grouping}")
    print(f" Conditions: {conditions}")
    print(f"{'='*60}\n")

    # Baseline (no symbolic features)
    print("--- [baseline] ---")
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir,
            grouping=grouping,
            grouping_cache_dir=groups_cache_dir,
            results_dir=run_dir,
            transactions_dir=run_dir / "transactions",
        )
    )
    gc.collect()

    # One symbolic run per filter condition
    sym_results: dict[str, ExperimentResult] = {}
    sym_paths: dict[str, Path] = {}

    for i, cond in enumerate(conditions):
        print(f"\n--- [{i+1}/{len(conditions)}] condition: {cond} ---")
        result = _run_condition(
            scenario=scenario,
            condition=cond,
            grouping=grouping,
            cache_dir=cache_dir,
            grouping_cache_dir=groups_cache_dir,
            results_dir=run_dir,
            transactions_dir=run_dir / "transactions",
        )
        sym_results[cond] = result
        sym_paths[cond] = Path(result.results_file)
        gc.collect()

    # Combined JSON
    combined: dict[str, Any] = {
        "experiment": "filter_effect",
        "scenario": scenario,
        "timestamp": ts,
        "grouping": args.grouping,
        "conditions": conditions,
        "baseline": _result_to_dict(baseline),
    }
    for cond in conditions:
        combined[cond] = {
            "filter_config": str(_FILTER_CONFIG_FILES[cond])
            if _FILTER_CONFIG_FILES[cond]
            else None,
            **_result_to_dict(sym_results[cond]),
        }

    out_json = run_dir / f"filter_effect_{ts}.json"
    with out_json.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results → {out_json}")

    # Feature funnels (requires reading the saved symbolic JSONs)
    funnels = {c: _feature_funnel(sym_paths[c]) for c in conditions}

    # Summary table
    cw = 16
    w = 26 + cw * (len(conditions) + 1)
    print(f"\n{'─'*w}")
    print(
        f"  {'':24}" + f"{'baseline':>{cw}}" + "".join(f"{c:>{cw}}" for c in conditions)
    )
    print(f"{'─'*w}")
    for label, getter in [
        ("AUC", lambda r: r.auc),
        ("Precision", lambda r: r.metrics.get("precision", float("nan"))),
        ("Recall", lambda r: r.metrics.get("recall", float("nan"))),
        ("F1", lambda r: r.metrics.get("f1", float("nan"))),
        ("FP", lambda r: r.metrics.get("fp", 0)),
    ]:
        fmt = ".4f" if label != "FP" else ".0f"
        row = f"  {label:<24}{getter(baseline):>{cw}{fmt}}"
        row += "".join(f"{getter(sym_results[c]):>{cw}{fmt}}" for c in conditions)
        print(row)
    print(f"{'─'*w}")

    # Detailed console tables
    _print_funnel_table(conditions, funnels)
    _print_performance_table(conditions, sym_results, baseline)
    _print_type_breakdown(conditions, sym_paths)
    _print_overlap_table(conditions, sym_paths, args.top_k)

    # Plots
    print(f"[plots] Writing to {plots_dir}")
    _plot_funnel(conditions, funnels, plots_dir)
    _plot_performance(conditions, sym_results, baseline, plots_dir)
    _plot_fp_analysis(conditions, sym_results, baseline, plots_dir)
    _plot_overlap_heatmap(conditions, sym_paths, args.top_k, plots_dir)
    _plot_type_breakdown(conditions, sym_paths, plots_dir)

    print(f"\nDone. Output in {run_dir}")


if __name__ == "__main__":
    main()
