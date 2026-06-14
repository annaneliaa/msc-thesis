"""
Compare fixed-window grouping vs AlertBERT grouping for one scenario.

Runs baseline + symbolic experiments under each grouping method using separate
cache directories so the two pipelines do not interfere with each other.

Usage:
    python src/thesis/scripts/run_grouping_compare.py <scenario> \\
        --alertbert-model-id mlm_1l_1h_16d_original_1_60k \\
        [--alertbert-models-path external/AlertBERT/saved_models] \\
        [--filter-config src/thesis/configs/mining_filters_strict.yaml] \\
        [--force] [--force-grouping] \\
        [--replot]

Output (all under artifacts/experiments/run_grouping_compare/grouping_compare_<run_ts>/):
    scenario/<scenario>/grouping_compare_<ts>.json
    scenario/<scenario>/confusion_matrices_<ts>.txt
    scenario/<scenario>/baseline_<ts>.json
    scenario/<scenario>/symbolic_<ts>.json
    plots/metrics.png        -- AUC / F1 / Precision / Recall per method
    plots/transactions.png   -- transaction count + size distribution
    plots/purity.png         -- transaction label purity (if label data available)
    plots/grouping_compare_<run_ts>.log
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR

from thesis.config import AlertBERTConfig, GroupingConfig
from thesis.experiments.baseline import (
    _load_transactions,
    run_baseline_experiment,
)
from thesis.experiments.symbolic import (
    run_symbolic_experiment,
)
from thesis.schemas.experiments import (
    ExperimentResult,
    BaselineExperimentConfig,
    SymbolicExperimentConfig,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))
EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_grouping_compare"

# Fixed AlertBERT hyperparameters, change here to tune for the scenario
_AB_DELTA = 2.0
_AB_THETA = 6.0
_AB_DIM_REDUCTION = 2
_AB_DEVICE = "cpu"

_LABELS = {
    "fixed_window": "Fixed window",
    "fixed_window_host": "Fixed window (host)",
    "time_delta": "Time-delta",
    "time_delta_host": "Time-delta (host)",
    "alertbert": "AlertBERT",
}
_SHORT_LABELS = {
    "fixed_window": "FW",
    "fixed_window_host": "FWH",
    "time_delta": "TD",
    "time_delta_host": "TDH",
    "alertbert": "AB",
}
_COLORS = {
    "fixed_window": "#4C72B0",
    "fixed_window_host": "#9EC8E8",
    "time_delta": "#55A868",
    "time_delta_host": "#A8D5B5",
    "alertbert": "#DD8452",
}


class _Tee:
    """Mirror stdout to both terminal and a log file."""

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
# Grouping config helpers
# ---------------------------------------------------------------------------


def _fixed_grouping() -> GroupingConfig:
    return GroupingConfig(mode="fixed_window")


def _time_delta_grouping() -> GroupingConfig:
    return GroupingConfig(mode="time_delta")


def _fixed_window_host_grouping() -> GroupingConfig:
    return GroupingConfig(mode="fixed_window_host")


def _time_delta_host_grouping() -> GroupingConfig:
    return GroupingConfig(mode="time_delta_host")


def _alertbert_grouping(model_id: str, models_path: str) -> GroupingConfig:
    return GroupingConfig(
        mode="alertbert",
        alertbert=AlertBERTConfig(
            model_id=model_id,
            models_path=models_path,
            delta=_AB_DELTA,
            theta=_AB_THETA,
            dim_reduction=_AB_DIM_REDUCTION,
            device=_AB_DEVICE,
        ),
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_pair(
    scenario: str,
    grouping: GroupingConfig,
    cache_dir: Path,
    filter_config: Path | None,
    results_dir: Path,
    grouping_cache_dir: Path | None = None,
    transactions_dir: Path | None = None,
    alerts_json_path: Path | None = None,
) -> tuple[ExperimentResult, ExperimentResult, list]:
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir,
            grouping=grouping,
            grouping_cache_dir=grouping_cache_dir,
            results_dir=results_dir,
            transactions_dir=transactions_dir,
            alerts_json_path=alerts_json_path,
        )
    )
    symbolic = run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir,
            grouping=grouping,
            filter_config=filter_config,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            grouping_cache_dir=grouping_cache_dir,
            results_dir=results_dir,
            transactions_dir=transactions_dir,
            alerts_json_path=alerts_json_path,
        )
    )
    txs = _load_transactions(
        scenario,
        cache_dir,
        groups_cache_dir=grouping_cache_dir,
        transactions_dir=transactions_dir,
    )
    return baseline, symbolic, txs


# ---------------------------------------------------------------------------
# Transaction statistics
# ---------------------------------------------------------------------------


def _compute_tx_stats(txs: list) -> dict[str, Any]:
    if not txs:
        return {
            "n_transactions": 0,
            "sizes": [],
            "item_sizes": [],
            "unique_types": [],
            "has_label_data": False,
        }

    sizes = [t.n_alerts for t in txs]
    item_sizes = [len(t.abs_items) for t in txs]
    # number of distinct alert token-sets per transaction (alert type diversity)
    unique_types = [len({frozenset(s) for s in t.sorted_items}) for t in txs]
    has_labels = any(t.alert_labels is not None for t in txs)

    stats: dict[str, Any] = {
        "n_transactions": len(txs),
        "sizes": sizes,
        "item_sizes": item_sizes,
        "unique_types": unique_types,
        "mean_alerts_per_tx": float(np.mean(sizes)),
        "median_alerts_per_tx": float(np.median(sizes)),
        "max_alerts_per_tx": int(max(sizes)),
        "mean_items_per_tx": float(np.mean(item_sizes)),
        "mean_unique_types_per_tx": float(np.mean(unique_types)),
        "n_benign": sum(1 for t in txs if t.tx_label == "benign"),
        "n_attack": sum(1 for t in txs if t.tx_label == "attack"),
        "has_label_data": has_labels,
    }

    if has_labels:
        n_pure_benign = sum(1 for t in txs if t.tx_label == "benign")
        n_pure_attack = sum(1 for t in txs if t.tx_label == "attack")
        n_mixed = sum(1 for t in txs if t.tx_label == "mixed")
        stats["n_mixed"] = n_mixed
        stats["n_pure_attack"] = n_pure_attack
        stats["n_pure_benign"] = n_pure_benign
        stats["purity_frac"] = 1.0 - n_mixed / len(txs)

    return stats


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _result_to_dict(r: ExperimentResult | ExperimentResult) -> dict:
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


_STATS_LIST_KEYS = {"sizes", "item_sizes", "unique_types"}


def _stats_for_json(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k not in _STATS_LIST_KEYS}


def _result_from_dict(d: dict, scenario: str) -> ExperimentResult:
    return ExperimentResult(
        scenario=scenario,
        model_name=d.get("model_name", "logreg"),
        model_version=d.get("model_version", "0.1.0"),
        schema_name=d.get("schema_name", ""),
        schema_version=d.get("schema_version", ""),
        auc=d.get("auc", float("nan")),
        n_transactions=d.get("n_transactions", 0),
        n_mixed_dropped=d.get("n_mixed_dropped", 0),
        n_features=d.get("n_features", 0),
        metrics=d.get("metrics", {}),
        results_file=Path(d.get("results_file", ".")),
        grouping_mode=d.get("grouping_mode", ""),
    )


def _latest_json(directory: Path, prefix: str) -> Path | None:
    candidates = sorted(directory.glob(f"{prefix}_*.json"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def _cm_block(result: ExperimentResult, label: str) -> str:
    m = result.metrics
    tp = m.get("tp", "?")
    tn = m.get("tn", "?")
    fp = m.get("fp", "?")
    fn = m.get("fn", "?")
    lines = [
        f"  {label}",
        f"    {'':22s} Predicted benign  Predicted attack",
        f"    {'Actual benign':<22s} {str(tn):>16}  {str(fp):>15}",
        f"    {'Actual attack':<22s} {str(fn):>16}  {str(tp):>15}",
    ]
    return "\n".join(lines)


_ALL_METHODS = [
    "fixed_window",
    "fixed_window_host",
    "time_delta",
    "time_delta_host",
    "alertbert",
]


def save_confusion_matrices(
    results: dict[str, dict[str, ExperimentResult]],
    scenario: str,
    timestamp: str,
    out_dir: Path,
) -> Path:
    """Write confusion matrices for all method×experiment combinations to a text file."""
    sep = "─" * 56
    blocks = [
        f"Confusion Matrices — scenario: {scenario} — {timestamp}",
        sep,
    ]
    for method in _ALL_METHODS:
        for exp_key, exp_label in [("baseline", "Baseline"), ("symbolic", "Symbolic")]:
            result = results[method][exp_key]
            header = f"{_LABELS[method]} / {exp_label}"
            blocks.append(_cm_block(result, header))
            blocks.append("")
    blocks.append(sep)

    out = out_dir / f"confusion_matrices_{timestamp}.txt"
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"  Confusion matrices → {out}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _metric(result: ExperimentResult | ExperimentResult, name: str) -> float:
    if name == "auc":
        return result.auc
    return result.metrics.get(name, float("nan"))


def plot_metrics(data: dict, out_dir: Path, filtered: bool = False) -> None:
    """2×2 grid: AUC / F1 / Precision / Recall, grouped by experiment type and method."""
    metric_pairs = [
        ("auc", "AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    exp_types = [("baseline", "Baseline"), ("symbolic", "Symbolic")]
    methods = [m for m in _ALL_METHODS if m in data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x = np.arange(len(exp_types))
    w = 0.15
    n = len(methods)
    offsets = [w * (i - (n - 1) / 2) for i in range(n)]

    for ax, (metric, title) in zip(axes, metric_pairs):
        for j, method in enumerate(methods):
            vals = [_metric(data[method][exp_key], metric) for exp_key, _ in exp_types]
            bars = ax.bar(
                x + offsets[j], vals, w, label=_LABELS[method], color=_COLORS[method]
            )
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in exp_types])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Detection metrics by grouping method — {data['scenario']}", fontsize=12
    )
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _boxplot(ax, box_data, methods, ylabel, title):
    bp = ax.boxplot(
        box_data,
        labels=[_LABELS[m] for m in methods],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(_COLORS[method])
        patch.set_alpha(0.75)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_transactions(data: dict, out_dir: Path, filtered: bool = False) -> None:
    """2×2 grid: transaction count, alerts/tx, itemset size, unique alert types."""
    methods = [m for m in _ALL_METHODS if m in data]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (0,0) — transaction count
    ax = axes[0, 0]
    counts = [data[m]["tx_stats"]["n_transactions"] for m in methods]
    bars = ax.bar(
        [_LABELS[m] for m in methods],
        counts,
        color=[_COLORS[m] for m in methods],
        width=0.5,
        edgecolor="white",
    )
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Number of transactions")
    ax.set_title("Transaction count")
    ax.grid(axis="y", alpha=0.3)

    # (0,1) — alerts per transaction
    _boxplot(
        axes[0, 1],
        [data[m]["tx_stats"]["sizes"] for m in methods],
        methods,
        "Alerts per transaction",
        "Transaction size (# alerts)",
    )

    # (1,0) — itemset size (unique items/tokens per transaction)
    _boxplot(
        axes[1, 0],
        [data[m]["tx_stats"].get("item_sizes", []) for m in methods],
        methods,
        "Unique items per transaction",
        "Itemset size (# unique tokens)",
    )

    # (1,1) — unique alert types per transaction
    # (distinct per-alert token-sets, i.e. distinct (short, host, sig) combinations)
    _boxplot(
        axes[1, 1],
        [data[m]["tx_stats"].get("unique_types", []) for m in methods],
        methods,
        "Unique alert types per transaction",
        "Alert type diversity\n(distinct token-sets)",
    )

    fig.suptitle(f"Transaction structure — {data['scenario']}", fontsize=12)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "transactions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_purity(data: dict, out_dir: Path, filtered: bool = False) -> None:
    """Stacked horizontal bar: pure-benign / pure-attack / mixed per grouping method."""
    methods = [m for m in _ALL_METHODS if m in data]

    if not any(data[m]["tx_stats"]["has_label_data"] for m in methods):
        print("  [skip purity.png] No alert_labels data available.")
        return

    fig, ax = plt.subplots(figsize=(9, 3))

    for i, method in enumerate(methods):
        stats = data[method]["tx_stats"]
        if not stats["has_label_data"]:
            continue
        total = stats["n_transactions"]
        n_b = stats.get("n_pure_benign", 0)
        n_a = stats.get("n_pure_attack", 0)
        n_m = stats.get("n_mixed", 0)
        fracs = [n_b / total, n_a / total, n_m / total]
        lefts = [0, n_b / total, (n_b + n_a) / total]
        seg_colors = ["#5B9BD5", "#ED7D31", "#A9A9A9"]
        seg_labels = ["Pure benign", "Pure attack", "Mixed"]
        seg_counts = [n_b, n_a, n_m]

        for frac, left, color, label, count in zip(
            fracs, lefts, seg_colors, seg_labels, seg_counts
        ):
            ax.barh(i, frac, left=left, color=color, label=label if i == 0 else "")
            if frac > 0.04:
                ax.text(
                    left + frac / 2,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_LABELS[m] for m in methods])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of transactions")
    ax.set_title(f"Transaction label purity — {data['scenario']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "purity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Phase 4: Feature overlap analysis (absorbed from run_feature_overlap_analysis.py)
# ---------------------------------------------------------------------------


def _fa_i(v: Any) -> int:
    return int(v) if v is not None else 0


def _fa_feature_funnel(sym: dict) -> dict:
    """Extract mining pipeline stage counts from a full symbolic result dict."""
    m = sym.get("mining", {})
    met = sym.get("metrics", {})
    top_coeff = met.get("top_feature_importances", {}).get("by_coefficient", {})
    top_perm = met.get("top_feature_importances", {}).get("by_permutation", {})
    n_mined = (
        _fa_i(m.get("n_itemsets_mined"))
        + _fa_i(m.get("n_sequences_mined"))
        + _fa_i(m.get("n_or_mined"))
    )
    n_abs_parts = (
        _fa_i(m.get("n_itemsets_after_abstraction"))
        + _fa_i(m.get("n_sequences_after_abstraction"))
        + _fa_i(m.get("n_or_after_abstraction"))
    )
    n_after_abstraction = n_abs_parts if n_abs_parts > 0 else n_mined
    return {
        "n_mined": n_mined,
        "n_after_abstraction": n_after_abstraction,
        "n_after_filter": _fa_i(m.get("n_candidate_features")),
        "n_final": _fa_i(m.get("n_features_final")),
        "n_nonzero_coeff": sum(1 for v in top_coeff.values() if v["importance"] != 0),
        "n_nonzero_perm": sum(1 for v in top_perm.values() if v["importance"] > 0),
        "n_symbolic_used": _fa_i(met.get("n_symbolic_features_used")),
    }


def _fa_top_feature_names(sym: dict, k: int, by: str = "by_coefficient") -> set[str]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    ranked = sorted(top.items(), key=lambda kv: abs(kv[1]["importance"]), reverse=True)
    return {name for name, info in ranked[:k] if info["importance"] != 0}


def _fa_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _fa_mining_type_breakdown(
    sym: dict, by: str = "by_coefficient", sign: str = "positive"
) -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        imp = info_dict["importance"]
        if imp == 0:
            continue
        if sign == "positive" and imp < 0:
            continue
        if sign == "negative" and imp > 0:
            continue
        mtype = info_dict.get("feature_info", {}).get("mining_type", "base")
        counts[mtype] = counts.get(mtype, 0) + 1
    return counts


def _fa_source_label_breakdown(
    sym: dict, by: str = "by_coefficient", sign: str = "positive"
) -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        imp = info_dict["importance"]
        if imp == 0:
            continue
        if sign == "positive" and imp < 0:
            continue
        if sign == "negative" and imp > 0:
            continue
        src = info_dict.get("feature_info", {}).get("source_label", "base")
        counts[src] = counts.get(src, 0) + 1
    return counts


def _fa_print_funnel_table(methods: list[str], funnels: dict[str, dict]) -> None:
    stages = [
        ("n_mined", "Mined total"),
        ("n_after_abstraction", "After abstraction"),
        ("n_after_filter", "After filter (+OR)"),
        ("n_final", "Final (dedup)"),
        ("n_nonzero_coeff", "Nonzero coeff"),
        ("n_nonzero_perm", "Nonzero perm"),
    ]
    col_w = 14
    w = 26 + col_w * len(methods)
    print("\n" + "═" * w)
    print("  FEATURE PIPELINE FUNNEL")
    print("─" * w)
    print(f"  {'Stage':<24}" + "".join(f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods))
    print("─" * w)
    for key, label in stages:
        print(
            f"  {label:<24}"
            + "".join(f"{funnels[m].get(key, 0):>{col_w},}" for m in methods)
        )
    print("═" * w)
    print(f"  Keys: {', '.join(f'{_SHORT_LABELS[m]}={m}' for m in methods)}\n")


def _fa_print_fp_table(
    methods: list[str], compare: dict, sym_data: dict[str, dict]
) -> None:
    col_w = 13
    w = 26 + col_w * len(methods)
    print("═" * w)
    print("  FALSE POSITIVE ANALYSIS")
    print("─" * w)
    print(
        f"  {'Metric':<24}" + "".join(f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods)
    )
    print("─" * w)
    for label, source, key in [
        ("Baseline FP", "baseline", "fp"),
        ("Symbolic FP", "symbolic", "fp"),
        ("Baseline precision", "baseline", "precision"),
        ("Symbolic precision", "symbolic", "precision"),
        ("Baseline recall", "baseline", "recall"),
        ("Symbolic recall", "symbolic", "recall"),
        ("Baseline F1", "baseline", "f1"),
        ("Symbolic F1", "symbolic", "f1"),
    ]:
        vals = []
        for m in methods:
            met = compare.get(m, {}).get(source, {}).get("metrics", {})
            v = met.get(key, float("nan"))
            if key in ("fp", "tp", "tn", "fn"):
                vals.append(f"{int(v):>{col_w},}" if v == v else f"{'?':>{col_w}}")
            else:
                vals.append(f"{v:>{col_w}.3f}" if v == v else f"{'?':>{col_w}}")
        print(f"  {label:<24}" + "".join(vals))
    print("─" * w)
    row_delta = f"  {'FP delta (sym-base)':<24}"
    row_pct = f"  {'FP reduction %':<24}"
    for m in methods:
        base_fp = (
            compare.get(m, {})
            .get("baseline", {})
            .get("metrics", {})
            .get("fp", float("nan"))
        )
        sym_fp = (
            compare.get(m, {})
            .get("symbolic", {})
            .get("metrics", {})
            .get("fp", float("nan"))
        )
        if base_fp == base_fp and sym_fp == sym_fp:
            delta = int(sym_fp) - int(base_fp)
            pct = (base_fp - sym_fp) / base_fp * 100 if base_fp > 0 else 0.0
            row_delta += f"{delta:>+{col_w},}"
            row_pct += f"{pct:>{col_w}.1f}%"
        else:
            row_delta += f"{'?':>{col_w}}"
            row_pct += f"{'?':>{col_w}}"
    print(row_delta)
    print(row_pct)
    print("═" * w + "\n")


def _fa_print_generalization_table(
    methods: list[str], sym_data: dict[str, dict]
) -> None:
    col_w = 13
    w = 26 + col_w * len(methods)
    print("═" * w)
    print("  GENERALIZATION GAP (train AUC - test AUC)")
    print("─" * w)
    print(
        f"  {'Metric':<24}" + "".join(f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods)
    )
    print("─" * w)
    for label, key in [
        ("Test AUC", "auc"),
        ("Train AUC", "train_auc"),
        ("Gap (train-test)", "performance_gap_train_vs_test"),
    ]:
        row = f"  {label:<24}"
        for m in methods:
            v = sym_data.get(m, {}).get("metrics", {}).get(key, float("nan"))
            row += f"{v:>{col_w}.4f}" if v == v else f"{'?':>{col_w}}"
        print(row)
    print("─" * w)
    row = f"  {'Feature sparsity':<24}"
    for m in methods:
        v = sym_data.get(m, {}).get("metrics", {}).get("feature_sparsity", float("nan"))
        row += f"{v:>{col_w}.4f}" if v == v else f"{'?':>{col_w}}"
    print(row)
    print("═" * w + "\n")


def _fa_print_type_breakdown(methods: list[str], sym_data: dict[str, dict]) -> None:
    col_w = 10
    w = 26 + col_w * len(methods)
    all_types: set[str] = set()
    breakdowns: dict[str, dict] = {}
    for m in methods:
        bd = _fa_mining_type_breakdown(sym_data.get(m, {}))
        breakdowns[m] = bd
        all_types |= set(bd.keys())
    print("═" * w)
    print("  NONZERO-COEFF FEATURE TYPE BREAKDOWN")
    print("─" * w)
    print(f"  {'Type':<24}" + "".join(f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods))
    print("─" * w)
    for mtype in sorted(all_types):
        print(
            f"  {mtype:<24}"
            + "".join(f"{breakdowns[m].get(mtype, 0):>{col_w},}" for m in methods)
        )
    all_src: set[str] = set()
    src_breakdowns: dict[str, dict] = {}
    for m in methods:
        bd = _fa_source_label_breakdown(sym_data.get(m, {}))
        src_breakdowns[m] = bd
        all_src |= set(bd.keys())
    print("─" * w)
    print("  SOURCE LABEL BREAKDOWN")
    print("─" * w)
    for src in sorted(all_src):
        print(
            f"  {src:<24}"
            + "".join(f"{src_breakdowns[m].get(src, 0):>{col_w},}" for m in methods)
        )
    print("═" * w + "\n")


def _fa_print_overlap_table(
    methods: list[str], sym_data: dict[str, dict], k: int
) -> None:
    feature_sets = {m: _fa_top_feature_names(sym_data.get(m, {}), k) for m in methods}
    print(f"  TOP-{k} FEATURE JACCARD OVERLAP (by_coefficient, nonzero only)")
    print("─" * (10 + 9 * len(methods)))
    print(f"  {'':6}" + "".join(f"{_SHORT_LABELS[m]:>9}" for m in methods))
    for ma in methods:
        row = f"  {_SHORT_LABELS[ma]:<6}"
        for mb in methods:
            j = _fa_jaccard(feature_sets[ma], feature_sets[mb])
            row += f"{j:>9.3f}" if not (j != j) else f"{'—':>9}"
        print(row)
    print()
    nonempty = [s for s in feature_sets.values() if s]
    if len(methods) > 1 and nonempty:
        shared = set.intersection(*nonempty)
        print(f"  Shared by ALL methods ({len(shared)} features):")
        for f in sorted(shared):
            print(f"    {f}")
    print()


def _fa_plot_funnel(
    methods: list[str], funnels: dict[str, dict], out_dir: Path, filtered: bool = False
) -> None:
    stages = [
        ("n_mined", "Mined"),
        ("n_after_abstraction", "After\nabstraction"),
        ("n_after_filter", "After filter\n(+OR pass-through)"),
        ("n_final", "Final\n(dedup)"),
        ("n_nonzero_coeff", "Learned\n(nonzero coeff)"),
    ]
    x = np.arange(len(stages))
    w = 0.15
    offsets = [w * (i - (len(methods) - 1) / 2) for i in range(len(methods))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [funnels[method].get(key, 0) for key, _ in stages]
        bars = ax.bar(
            x + offsets[i], vals, w, label=_LABELS[method], color=_COLORS[method]
        )
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
    ax.set_title("Feature pipeline funnel by grouping method")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_funnel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_overlap_heatmap(
    methods: list[str],
    sym_data: dict[str, dict],
    k: int,
    out_dir: Path,
    filtered: bool = False,
) -> None:
    feature_sets = {m: _fa_top_feature_names(sym_data.get(m, {}), k) for m in methods}
    n = len(methods)
    matrix = np.zeros((n, n))
    matrix_vis = np.zeros((n, n))
    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            v = _fa_jaccard(feature_sets[ma], feature_sets[mb])
            matrix[i, j] = v
            matrix_vis[i, j] = 0.0 if (v != v) else v

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_vis, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    labels = [_SHORT_LABELS[m] for m in methods]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            txt = "—" if (v != v) else f"{v:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color="black" if matrix_vis[i, j] < 0.7 else "white",
            )
    ax.set_title(f"Top-{k} feature Jaccard overlap (nonzero coeff)")
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_overlap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_fp_analysis(
    methods: list[str], compare: dict, out_dir: Path, filtered: bool = False
) -> None:
    metrics_to_plot = [
        ("fp", "False Positives", True),
        ("precision", "Precision", False),
        ("recall", "Recall", False),
        ("f1", "F1", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    x = np.arange(len(methods))
    w = 0.35
    for ax, (key, title, is_count) in zip(axes, metrics_to_plot):
        base_vals = [
            compare.get(m, {})
            .get("baseline", {})
            .get("metrics", {})
            .get(key, float("nan"))
            for m in methods
        ]
        sym_vals = [
            compare.get(m, {})
            .get("symbolic", {})
            .get("metrics", {})
            .get(key, float("nan"))
            for m in methods
        ]
        b1 = ax.bar(
            x - w / 2, base_vals, w, label="Baseline", color="#5B9BD5", alpha=0.85
        )
        b2 = ax.bar(
            x + w / 2, sym_vals, w, label="Symbolic", color="#ED7D31", alpha=0.85
        )
        for bar, val in zip(list(b1) + list(b2), base_vals + sym_vals):
            if val == val and val > 0:
                fmt = f"{int(val)}" if is_count else f"{val:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([_SHORT_LABELS[m] for m in methods])
        if not is_count:
            ax.set_ylim(0, 1.15)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Baseline vs Symbolic detection metrics — {compare.get('scenario', '')}"
    )
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "fp_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_type_breakdown(
    methods: list[str], sym_data: dict[str, dict], out_dir: Path, filtered: bool = False
) -> None:
    all_types_set: set[str] = set()
    breakdowns: dict[str, dict] = {}
    for m in methods:
        bd = _fa_mining_type_breakdown(sym_data.get(m, {}))
        breakdowns[m] = bd
        all_types_set |= set(bd.keys())
    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    for mtype in sorted(all_types_set):
        vals = np.array([breakdowns[m].get(mtype, 0) for m in methods], dtype=float)
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
    ax.set_xticklabels([_SHORT_LABELS[m] for m in methods])
    ax.set_ylabel("Count of nonzero-coeff features")
    ax.set_title("Feature type breakdown (nonzero model coefficients)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_type_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_coeff_vs_perm(
    methods: list[str],
    sym_data: dict[str, dict],
    k: int,
    out_dir: Path,
    filtered: bool = False,
) -> None:
    if len(methods) < 2:
        return
    coeff_sets = {
        m: _fa_top_feature_names(sym_data.get(m, {}), k, "by_coefficient")
        for m in methods
    }
    perm_sets = {
        m: _fa_top_feature_names(sym_data.get(m, {}), k, "by_permutation")
        for m in methods
    }
    agreement = [_fa_jaccard(coeff_sets[m], perm_sets[m]) for m in methods]
    agreement_vis = [0.0 if (v != v) else v for v in agreement]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(
        [_SHORT_LABELS[m] for m in methods],
        agreement_vis,
        color=[_COLORS[m] for m in methods],
        alpha=0.85,
    )
    for bar, val in zip(bars, agreement):
        txt = "—" if (val != val) else f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Coeff vs permutation importance agreement (top-{k})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "coeff_vs_perm_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_signed_coefficients(
    methods: list[str],
    sym_data: dict[str, dict],
    k: int,
    out_dir: Path,
    filtered: bool = False,
) -> None:
    from matplotlib.patches import Patch

    present = [m for m in methods if m in sym_data]
    if not present:
        return
    n = len(present)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig_h = max(k * 0.28 + 1.5, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, fig_h * n_rows))
    axes_flat: list = np.array(axes).flatten().tolist() if n > 1 else [axes]

    for ax_idx, method in enumerate(present):
        ax = axes_flat[ax_idx]
        top = (
            sym_data[method]
            .get("metrics", {})
            .get("top_feature_importances", {})
            .get("by_coefficient", {})
        )
        ranked = sorted(
            top.items(), key=lambda kv: abs(kv[1]["importance"]), reverse=True
        )
        ranked = [
            (name, info["importance"])
            for name, info in ranked
            if info["importance"] != 0
        ][:k]
        ranked = ranked[::-1]
        names = [r[0][:50] for r in ranked]
        values = [r[1] for r in ranked]
        colors = ["#4C72B0" if v > 0 else "#C94040" for v in values]
        y = np.arange(len(names))
        ax.barh(y, values, color=colors, alpha=0.85, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(_LABELS[method], fontsize=9)
        ax.set_xlabel("Coefficient value", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes_flat[len(present) :]:
        ax.set_visible(False)

    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.85, label="Positive → attack"),
        Patch(facecolor="#C94040", alpha=0.85, label="Negative → benign"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Top-{k} features by |coefficient| — logistic regression", fontsize=11
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "signed_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _fa_plot_sign_split_breakdown(
    methods: list[str], sym_data: dict[str, dict], out_dir: Path, filtered: bool = False
) -> None:
    from matplotlib.patches import Patch

    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }
    src_colors = {"attack": "#C94040", "benign": "#4C72B0", "unknown": "#AAAAAA"}

    fig, (ax_type, ax_src) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(methods))

    for ax, breakdown_fn, colors, title in [
        (
            ax_type,
            _fa_mining_type_breakdown,
            type_colors,
            "Mining type by coefficient sign",
        ),
        (
            ax_src,
            _fa_source_label_breakdown,
            src_colors,
            "Source label by coefficient sign",
        ),
    ]:
        all_cats: set[str] = set()
        pos_bds: dict[str, dict] = {}
        neg_bds: dict[str, dict] = {}
        for m in methods:
            sym = sym_data.get(m, {})
            pos_bds[m] = breakdown_fn(sym, sign="positive")
            neg_bds[m] = breakdown_fn(sym, sign="negative")
            all_cats |= set(pos_bds[m]) | set(neg_bds[m])

        pos_bottoms = np.zeros(len(methods))
        neg_bottoms = np.zeros(len(methods))
        legend_handles: list = []

        for cat in sorted(all_cats):
            color = colors.get(cat, "#BBBBBB")
            pos_vals = np.array([pos_bds[m].get(cat, 0) for m in methods], dtype=float)
            neg_vals = np.array([-neg_bds[m].get(cat, 0) for m in methods], dtype=float)

            bars_pos = ax.bar(x, pos_vals, bottom=pos_bottoms, color=color, alpha=0.85)
            bars_neg = ax.bar(
                x, neg_vals, bottom=neg_bottoms, color=color, alpha=0.45, hatch="//"
            )

            for bar, val in zip(bars_pos, pos_vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(val)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )
            for bar, val in zip(bars_neg, neg_vals):
                if val < 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(-val)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                        fontweight="bold",
                    )

            pos_bottoms += pos_vals
            neg_bottoms += neg_vals
            legend_handles.append(Patch(facecolor=color, alpha=0.85, label=cat))

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([_SHORT_LABELS[m] for m in methods])
        ax.set_ylabel("Feature count")
        ax.set_title(title)
        ax.legend(handles=legend_handles, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02,
            0.98,
            "↑ positive coeff (attack)",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="#333333",
        )
        ax.text(
            0.02,
            0.02,
            "↓ negative coeff (benign)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            color="#333333",
        )

    fig.suptitle("Feature breakdown by coefficient sign", fontsize=11)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "sign_split_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _run_feature_analysis(
    methods: list[str],
    sym_paths: dict[str, Path],
    compare: dict,
    top_k: int,
    out_dir: Path,
    filtered: bool,
) -> None:
    """Phase 4: load symbolic JSONs and run feature overlap analysis."""
    sym_data: dict[str, dict] = {}
    for m in methods:
        p = sym_paths.get(m)
        if p and p.exists():
            with p.open() as f:
                sym_data[m] = json.load(f)
        else:
            print(f"  [warn] No symbolic JSON for {m}, skipping in feature analysis.")

    if not sym_data:
        print("  [skip] No symbolic data available for feature analysis.")
        return

    present = [m for m in methods if m in sym_data]
    funnels = {m: _fa_feature_funnel(sym_data[m]) for m in present}

    _fa_print_funnel_table(present, funnels)
    _fa_print_fp_table(present, compare, sym_data)
    _fa_print_generalization_table(present, sym_data)
    _fa_print_type_breakdown(present, sym_data)
    _fa_print_overlap_table(present, sym_data, top_k)

    fa_dir = out_dir / "feature_analysis"
    fa_dir.mkdir(exist_ok=True)
    print(f"\n[Phase 4 plots] Saving to {fa_dir}")

    _fa_plot_funnel(present, funnels, fa_dir, filtered=filtered)
    _fa_plot_overlap_heatmap(present, sym_data, top_k, fa_dir, filtered=filtered)
    _fa_plot_fp_analysis(present, compare, fa_dir, filtered=filtered)
    _fa_plot_type_breakdown(present, sym_data, fa_dir, filtered=filtered)
    _fa_plot_coeff_vs_perm(present, sym_data, top_k, fa_dir, filtered=filtered)
    _fa_plot_signed_coefficients(present, sym_data, top_k, fa_dir, filtered=filtered)
    _fa_plot_sign_split_breakdown(present, sym_data, fa_dir, filtered=filtered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-window vs AlertBERT grouping for one scenario."
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--alertbert-model-id",
        default=None,
        help="AlertBERT model ID (subdirectory inside models path). Required unless --replot is used.",
    )
    parser.add_argument(
        "--alertbert-models-path",
        default=str(_REPO / "external" / "AlertBERT" / "saved_models"),
        help="Directory containing AlertBERT saved model subdirectories.",
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        help="Mining filter YAML for the symbolic experiment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if grouping_compare results already exist (reuses grouping cache).",
    )
    parser.add_argument(
        "--force-grouping",
        action="store_true",
        help="Re-run and also discard the AlertBERT grouping cache, forcing groups to be recomputed. Implies --force.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Reload existing results JSON and regenerate confusion matrix and metrics plot without re-running experiments.",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Use detector-filtered alerts (alerts_filtered.json) instead of alerts.json.",
    )
    parser.add_argument(
        "--no-feature-analysis",
        action="store_true",
        help="Skip Phase 4 (feature overlap analysis plots). Useful for quick metric-only runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-K features for overlap/Jaccard analysis in Phase 4 (default: 25).",
    )
    args = parser.parse_args()

    scenario = args.scenario

    if args.replot:
        existing = next(
            iter(
                sorted(
                    EXPERIMENTS_DIR.glob(
                        f"*/scenario/{scenario}/grouping_compare_*.json"
                    )
                )[-1:]
            ),
            None,
        )
        if not existing:
            print(
                f"[error] No existing results found under {EXPERIMENTS_DIR}. Run without --replot first."
            )
            return
        print(f"[replot] Loading {existing}")
        with existing.open() as f:
            data = json.load(f)
        ts = data["timestamp"]
        results = {}
        for method in _ALL_METHODS:
            if method in data:
                results[method] = {
                    "baseline": _result_from_dict(data[method]["baseline"], scenario),
                    "symbolic": _result_from_dict(data[method]["symbolic"], scenario),
                }
        scenario_dir = existing.parent
        plot_dir = existing.parent.parent.parent / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrices(results, scenario, ts, scenario_dir)
        plot_data = {"scenario": scenario}
        for method in _ALL_METHODS:
            if method in data:
                plot_data[method] = {
                    "baseline": results[method]["baseline"],
                    "symbolic": results[method]["symbolic"],
                    "tx_stats": data[method]["tx_stats"],
                }
        filtered = data.get("filtered", False)
        plot_metrics(plot_data, plot_dir, filtered=filtered)
        plot_transactions(plot_data, plot_dir, filtered=filtered)
        plot_purity(plot_data, plot_dir, filtered=filtered)
        print("Done.")
        return

    if not args.alertbert_model_id:
        parser.error("--alertbert-model-id is required when not using --replot")

    existing_json = next(
        iter(
            sorted(
                EXPERIMENTS_DIR.glob(f"*/scenario/{scenario}/grouping_compare_*.json")
            )[-1:]
        ),
        None,
    )
    if existing_json and not args.force and not args.force_grouping:
        with existing_json.open() as f:
            _existing_data = json.load(f)
        if _existing_data.get("filtered", False) == args.filtered:
            print(f"[skip] Existing results: {existing_json}. Use --force to re-run.")
            return

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / f"grouping_compare_{run_ts}"
    scenario_dir = run_dir / "scenario" / scenario
    plots_dir = run_dir / "plots"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = plots_dir / f"grouping_compare_{run_ts}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")

    try:
        _main_body(args, scenario, scenario_dir, plots_dir, run_ts)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _main_body(
    args: object,
    scenario: str,
    scenario_dir: Path,
    plots_dir: Path,
    run_ts: str,
) -> None:
    # Each method gets its own groups cache dir so the alert-cache skip in
    # _process_alert_batch doesn't cause one method's groups to be reused by the
    # next method (which produced identical scores between methods in earlier runs).
    cache_dir = CACHE_DIR / scenario
    groups_base = CACHE_DIR / "groups" / scenario
    fw_groups_dir = groups_base / "fixed_window"
    fwh_groups_dir = groups_base / "fixed_window_host"
    td_groups_dir = groups_base / "time_delta"
    tdh_groups_dir = groups_base / "time_delta_host"
    alertbert_groups_dir = (
        CACHE_DIR / "alertbert_groups" / scenario / args.alertbert_model_id
    )

    if args.force_grouping:
        for d in [
            fw_groups_dir,
            fwh_groups_dir,
            td_groups_dir,
            tdh_groups_dir,
            alertbert_groups_dir,
        ]:
            if d.exists():
                shutil.rmtree(d)
                print(f"  [force-grouping] Cleared {d}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for d in [
        fw_groups_dir,
        fwh_groups_dir,
        td_groups_dir,
        tdh_groups_dir,
        alertbert_groups_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    grouping_fw = _fixed_grouping()
    grouping_fwh = _fixed_window_host_grouping()
    grouping_td = _time_delta_grouping()
    grouping_tdh = _time_delta_host_grouping()
    grouping_ab = _alertbert_grouping(
        args.alertbert_model_id, args.alertbert_models_path
    )

    alerts_json_path = (
        _REPO / "artifacts" / "processed-data" / scenario / "alerts_filtered.json"
        if getattr(args, "filtered", False)
        else None
    )

    print(f"\n{'='*60}")
    print(f" Grouping comparison: {scenario}")
    print(f"{'='*60}")

    print("\n--- [1/5] fixed_window ---")
    baseline_fw, symbolic_fw, txs_fw = _run_pair(
        scenario,
        grouping_fw,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=fw_groups_dir,
        transactions_dir=scenario_dir / "fixed_window" / "transactions",
        alerts_json_path=alerts_json_path,
    )
    gc.collect()

    print("\n--- [2/5] fixed_window_host ---")
    baseline_fwh, symbolic_fwh, txs_fwh = _run_pair(
        scenario,
        grouping_fwh,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=fwh_groups_dir,
        transactions_dir=scenario_dir / "fixed_window_host" / "transactions",
        alerts_json_path=alerts_json_path,
    )
    gc.collect()

    print("\n--- [3/5] time_delta ---")
    baseline_td, symbolic_td, txs_td = _run_pair(
        scenario,
        grouping_td,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=td_groups_dir,
        transactions_dir=scenario_dir / "time_delta" / "transactions",
        alerts_json_path=alerts_json_path,
    )
    gc.collect()

    print("\n--- [4/5] time_delta_host ---")
    baseline_tdh, symbolic_tdh, txs_tdh = _run_pair(
        scenario,
        grouping_tdh,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=tdh_groups_dir,
        transactions_dir=scenario_dir / "time_delta_host" / "transactions",
        alerts_json_path=alerts_json_path,
    )

    # Free allocations before loading torch + AlertBERT model.
    gc.collect()

    print("\n--- [5/5] alertbert ---")
    baseline_ab, symbolic_ab, txs_ab = _run_pair(
        scenario,
        grouping_ab,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=alertbert_groups_dir,
        transactions_dir=scenario_dir / "alertbert" / "transactions",
        alerts_json_path=alerts_json_path,
    )

    stats_fw = _compute_tx_stats(txs_fw)
    stats_fwh = _compute_tx_stats(txs_fwh)
    stats_td = _compute_tx_stats(txs_td)
    stats_tdh = _compute_tx_stats(txs_tdh)
    stats_ab = _compute_tx_stats(txs_ab)

    timestamp = run_ts
    combined = {
        "experiment": "grouping_compare",
        "scenario": scenario,
        "timestamp": timestamp,
        "filtered": bool(alerts_json_path is not None),
        "alertbert_config": grouping_ab.alertbert.model_dump(),
        "fixed_window": {
            "grouping": {"mode": "fixed_window", "params": None},
            "baseline": _result_to_dict(baseline_fw),
            "symbolic": _result_to_dict(symbolic_fw),
            "tx_stats": _stats_for_json(stats_fw),
        },
        "fixed_window_host": {
            "grouping": {"mode": "fixed_window_host", "params": None},
            "baseline": _result_to_dict(baseline_fwh),
            "symbolic": _result_to_dict(symbolic_fwh),
            "tx_stats": _stats_for_json(stats_fwh),
        },
        "time_delta": {
            "grouping": {"mode": "time_delta", "params": None},
            "baseline": _result_to_dict(baseline_td),
            "symbolic": _result_to_dict(symbolic_td),
            "tx_stats": _stats_for_json(stats_td),
        },
        "time_delta_host": {
            "grouping": {"mode": "time_delta_host", "params": None},
            "baseline": _result_to_dict(baseline_tdh),
            "symbolic": _result_to_dict(symbolic_tdh),
            "tx_stats": _stats_for_json(stats_tdh),
        },
        "alertbert": {
            "grouping": {
                "mode": "alertbert",
                "params": grouping_ab.alertbert.model_dump(),
            },
            "baseline": _result_to_dict(baseline_ab),
            "symbolic": _result_to_dict(symbolic_ab),
            "tx_stats": _stats_for_json(stats_ab),
        },
    }

    out_json = scenario_dir / f"grouping_compare_{timestamp}.json"
    with out_json.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results → {out_json}")

    save_confusion_matrices(
        {
            "fixed_window": {"baseline": baseline_fw, "symbolic": symbolic_fw},
            "fixed_window_host": {"baseline": baseline_fwh, "symbolic": symbolic_fwh},
            "time_delta": {"baseline": baseline_td, "symbolic": symbolic_td},
            "time_delta_host": {"baseline": baseline_tdh, "symbolic": symbolic_tdh},
            "alertbert": {"baseline": baseline_ab, "symbolic": symbolic_ab},
        },
        scenario,
        timestamp,
        scenario_dir,
    )

    # Summary table
    col_w = 16
    col_headers = [
        "fixed_window",
        "fixed_window_host",
        "time_delta",
        "time_delta_host",
        "alertbert",
    ]
    print(f"\n{'─'*100}")
    header = f"  {'':20s}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    print(header)
    print(f"{'─'*100}")
    for label, vals in [
        (
            "baseline AUC",
            [
                baseline_fw.auc,
                baseline_fwh.auc,
                baseline_td.auc,
                baseline_tdh.auc,
                baseline_ab.auc,
            ],
        ),
        (
            "symbolic AUC",
            [
                symbolic_fw.auc,
                symbolic_fwh.auc,
                symbolic_td.auc,
                symbolic_tdh.auc,
                symbolic_ab.auc,
            ],
        ),
        (
            "baseline F1",
            [
                r.metrics.get("f1", float("nan"))
                for r in [
                    baseline_fw,
                    baseline_fwh,
                    baseline_td,
                    baseline_tdh,
                    baseline_ab,
                ]
            ],
        ),
        (
            "symbolic F1",
            [
                r.metrics.get("f1", float("nan"))
                for r in [
                    symbolic_fw,
                    symbolic_fwh,
                    symbolic_td,
                    symbolic_tdh,
                    symbolic_ab,
                ]
            ],
        ),
        (
            "n_transactions",
            [
                float(s["n_transactions"])
                for s in [stats_fw, stats_fwh, stats_td, stats_tdh, stats_ab]
            ],
        ),
    ]:
        fmt = ".4f" if label != "n_transactions" else ".0f"
        row = f"  {label:<20s}" + "".join(f"{v:>{col_w}{fmt}}" for v in vals)
        print(row)
    print(f"{'─'*100}")

    print(f"\n[plots] Saving to {plots_dir}")
    plot_data = {
        "scenario": scenario,
        "fixed_window": {
            "baseline": baseline_fw,
            "symbolic": symbolic_fw,
            "tx_stats": stats_fw,
        },
        "fixed_window_host": {
            "baseline": baseline_fwh,
            "symbolic": symbolic_fwh,
            "tx_stats": stats_fwh,
        },
        "time_delta": {
            "baseline": baseline_td,
            "symbolic": symbolic_td,
            "tx_stats": stats_td,
        },
        "time_delta_host": {
            "baseline": baseline_tdh,
            "symbolic": symbolic_tdh,
            "tx_stats": stats_tdh,
        },
        "alertbert": {
            "baseline": baseline_ab,
            "symbolic": symbolic_ab,
            "tx_stats": stats_ab,
        },
    }

    filtered = bool(alerts_json_path is not None)
    plot_metrics(plot_data, plots_dir, filtered=filtered)
    plot_transactions(plot_data, plots_dir, filtered=filtered)
    plot_purity(plot_data, plots_dir, filtered=filtered)

    # Phase 4: feature overlap analysis
    if not getattr(args, "no_feature_analysis", False):
        print("\n[Phase 4] Feature overlap analysis...")
        methods = [m for m in _ALL_METHODS if m in combined]
        sym_paths = {
            "fixed_window": Path(symbolic_fw.results_file),
            "fixed_window_host": Path(symbolic_fwh.results_file),
            "time_delta": Path(symbolic_td.results_file),
            "time_delta_host": Path(symbolic_tdh.results_file),
            "alertbert": Path(symbolic_ab.results_file),
        }
        compare_dict = {
            "scenario": scenario,
            **{
                m: {
                    "baseline": {"metrics": combined[m]["baseline"]["metrics"]},
                    "symbolic": {"metrics": combined[m]["symbolic"]["metrics"]},
                }
                for m in methods
            },
        }
        _run_feature_analysis(
            methods=methods,
            sym_paths={m: sym_paths[m] for m in methods},
            compare=compare_dict,
            top_k=getattr(args, "top_k", 25),
            out_dir=plots_dir,
            filtered=filtered,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
