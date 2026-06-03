"""
Feature overlap and mining pipeline analysis across grouping methods.

Loads the most recent grouping_compare run for a scenario and produces:
  - Feature pipeline funnel: n_mined → n_after_filter → n_final → n_nonzero_coeff
  - Top-K feature overlap: Jaccard similarity heatmap across methods
  - FP analysis: baseline vs symbolic FP / precision per method
  - Mining type composition of learned features (itemset / sequence / OR)
  - Source label breakdown (attack-mined vs benign-mined survivors)
  - Generalization gap (train AUC vs test AUC)
  - Coefficient vs permutation importance agreement

Output (under artifacts/experiments/run_feature_overlap/feature_overlap_<ts>/<scenario>/):
    analysis_<ts>.txt   -- full console output
    feature_funnel.png
    feature_overlap.png
    fp_analysis.png
    feature_type_breakdown.png
    coeff_vs_perm_agreement.png

Usage:
    python src/thesis/scripts/run_feature_overlap_analysis.py <scenario> \\
        [--run-dir <path>]    # path to a specific grouping_compare scenario dir
        [--top-k 25]          # features used for Jaccard overlap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_COMPARE_BASE = _REPO / "artifacts" / "experiments" / "run_grouping_compare"
_ANALYSIS_BASE = _REPO / "artifacts" / "experiments" / "run_feature_overlap"

_ALL_METHODS = [
    "fixed_window",
    "fixed_window_host",
    "time_delta",
    "time_delta_host",
    "alertbert",
]
_LABELS = {
    "fixed_window": "Fixed-window",
    "fixed_window_host": "Fixed-window\n(host)",
    "time_delta": "Time-delta",
    "time_delta_host": "Time-delta\n(host)",
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _latest_compare_json(scenario: str) -> Path:
    candidates = sorted(
        _COMPARE_BASE.glob(f"*/scenario/{scenario}/grouping_compare_*.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No grouping_compare results found for scenario '{scenario}' under {_COMPARE_BASE}"
        )
    return candidates[-1]


def load_run(
    scenario: str, run_dir: Path | None
) -> tuple[dict, list[str], dict[str, dict]]:
    """
    Returns (compare_json_data, present_methods, {method: full_symbolic_json}).
    """
    if run_dir is not None:
        candidates = sorted(run_dir.glob("grouping_compare_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No grouping_compare JSON in {run_dir}")
        compare_path = candidates[-1]
    else:
        compare_path = _latest_compare_json(scenario)

    print(f"Loading: {compare_path}")
    with compare_path.open() as f:
        compare = json.load(f)

    methods = [m for m in _ALL_METHODS if m in compare]
    symbolic_data: dict[str, dict] = {}

    for method in methods:
        results_file = compare[method].get("symbolic", {}).get("results_file")
        if results_file and Path(results_file).exists():
            with open(results_file) as f:
                symbolic_data[method] = json.load(f)
        else:
            print(f"  [warn] No symbolic results_file for {method}")

    return compare, methods, symbolic_data


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _i(v: Any) -> int:
    return int(v) if v is not None else 0


def feature_funnel(sym: dict) -> dict:
    """Extract mining pipeline stage counts from a full symbolic result dict."""
    m = sym.get("mining", {})
    met = sym.get("metrics", {})
    top_coeff = met.get("top_feature_importances", {}).get("by_coefficient", {})
    top_perm = met.get("top_feature_importances", {}).get("by_permutation", {})

    n_nonzero_coeff = sum(1 for v in top_coeff.values() if v["importance"] > 0)
    n_nonzero_perm = sum(1 for v in top_perm.values() if v["importance"] > 0)

    n_mined = (
        _i(m.get("n_itemsets_mined"))
        + _i(m.get("n_sequences_mined"))
        + _i(m.get("n_or_mined"))
    )

    # After abstraction: use tracked counts when abstraction was applied; fall back
    # to n_mined (no change) when abstraction was skipped (all three fields are None).
    n_abs_parts = (
        _i(m.get("n_itemsets_after_abstraction"))
        + _i(m.get("n_sequences_after_abstraction"))
        + _i(m.get("n_or_after_abstraction"))
    )
    n_after_abstraction = n_abs_parts if n_abs_parts > 0 else n_mined

    # n_candidate_features is the honest post-filter total: itemsets+sequences after
    # their respective filters, PLUS OR features which bypass filtering entirely.
    # Using this avoids a misleading drop followed by a jump from unfiltered OR patterns.
    n_after_filter = _i(m.get("n_candidate_features"))

    return {
        "n_mined": n_mined,
        "n_after_abstraction": n_after_abstraction,
        "n_after_filter": n_after_filter,
        "n_final": _i(m.get("n_features_final")),
        "n_nonzero_coeff": n_nonzero_coeff,
        "n_nonzero_perm": n_nonzero_perm,
        "n_symbolic_used": _i(met.get("n_symbolic_features_used")),
        "filter_config": m.get("filter_config"),
    }


def top_feature_names(sym: dict, k: int, by: str = "by_coefficient") -> set[str]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    ranked = sorted(top.items(), key=lambda kv: kv[1]["importance"], reverse=True)
    nonzero = [(name, info) for name, info in ranked if info["importance"] > 0]
    return {name for name, _ in nonzero[:k]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def mining_type_breakdown(sym: dict, by: str = "by_coefficient") -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        if info_dict["importance"] <= 0:
            continue
        mtype = info_dict.get("feature_info", {}).get("mining_type", "base")
        counts[mtype] = counts.get(mtype, 0) + 1
    return counts


def source_label_breakdown(sym: dict, by: str = "by_coefficient") -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        if info_dict["importance"] <= 0:
            continue
        src = info_dict.get("feature_info", {}).get("source_label", "base")
        counts[src] = counts.get(src, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------


def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = round(n / total * width)
    return "█" * filled + "░" * (width - filled)


def print_funnel_table(methods: list[str], funnels: dict[str, dict]) -> None:
    stages = [
        ("n_mined", "Mined total"),
        ("n_after_abstraction", "After abstraction"),
        ("n_after_filter", "After filter (+OR)"),
        ("n_final", "Final (dedup)"),
        ("n_nonzero_coeff", "Nonzero coeff"),
        ("n_nonzero_perm", "Nonzero perm"),
    ]
    col_w = 14
    print("\n" + "═" * (26 + col_w * len(methods)))
    print("  FEATURE PIPELINE FUNNEL")
    print("─" * (26 + col_w * len(methods)))
    header = f"  {'Stage':<24}" + "".join(
        f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods
    )
    print(header)
    print("─" * (26 + col_w * len(methods)))
    for key, label in stages:
        row = f"  {label:<24}" + "".join(
            f"{funnels[m].get(key, 0):>{col_w},}" for m in methods
        )
        print(row)
    print("═" * (26 + col_w * len(methods)))
    print(f"  Keys: {', '.join(f'{_SHORT_LABELS[m]}={m}' for m in methods)}\n")


def print_fp_table(
    methods: list[str], compare: dict, sym_data: dict[str, dict]
) -> None:
    col_w = 13
    print("═" * (26 + col_w * len(methods)))
    print("  FALSE POSITIVE ANALYSIS")
    print("─" * (26 + col_w * len(methods)))
    header = f"  {'Metric':<24}" + "".join(
        f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods
    )
    print(header)
    print("─" * (26 + col_w * len(methods)))

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

    print("─" * (26 + col_w * len(methods)))

    # FP delta: symbolic - baseline
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
    print("═" * (26 + col_w * len(methods)) + "\n")


def print_generalization_table(methods: list[str], sym_data: dict[str, dict]) -> None:
    col_w = 13
    print("═" * (26 + col_w * len(methods)))
    print("  GENERALIZATION GAP (train AUC - test AUC)")
    print("─" * (26 + col_w * len(methods)))
    header = f"  {'Metric':<24}" + "".join(
        f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods
    )
    print(header)
    print("─" * (26 + col_w * len(methods)))
    for label, key in [
        ("Test AUC", "auc"),
        ("Train AUC", "train_auc"),
        ("Gap (train-test)", "performance_gap_train_vs_test"),
    ]:
        row = f"  {label:<24}"
        for m in methods:
            met = sym_data.get(m, {}).get("metrics", {})
            v = met.get(key, float("nan"))
            row += f"{v:>{col_w}.4f}" if v == v else f"{'?':>{col_w}}"
        print(row)
    print("─" * (26 + col_w * len(methods)))
    for label, key in [("Feature sparsity", "feature_sparsity")]:
        row = f"  {label:<24}"
        for m in methods:
            met = sym_data.get(m, {}).get("metrics", {})
            v = met.get(key, float("nan"))
            row += f"{v:>{col_w}.4f}" if v == v else f"{'?':>{col_w}}"
        print(row)
    print("═" * (26 + col_w * len(methods)) + "\n")


def print_type_breakdown(methods: list[str], sym_data: dict[str, dict]) -> None:
    all_types = set()
    breakdowns = {}
    for m in methods:
        bd = mining_type_breakdown(sym_data.get(m, {}))
        breakdowns[m] = bd
        all_types |= set(bd.keys())

    col_w = 10
    print("═" * (26 + col_w * len(methods)))
    print("  NONZERO-COEFF FEATURE TYPE BREAKDOWN")
    print("─" * (26 + col_w * len(methods)))
    header = f"  {'Type':<24}" + "".join(
        f"{_SHORT_LABELS[m]:>{col_w}}" for m in methods
    )
    print(header)
    print("─" * (26 + col_w * len(methods)))
    for mtype in sorted(all_types):
        row = f"  {mtype:<24}" + "".join(
            f"{breakdowns[m].get(mtype, 0):>{col_w},}" for m in methods
        )
        print(row)

    # source label breakdown
    print("─" * (26 + col_w * len(methods)))
    print("  SOURCE LABEL BREAKDOWN")
    print("─" * (26 + col_w * len(methods)))
    all_src = set()
    src_breakdowns = {}
    for m in methods:
        bd = source_label_breakdown(sym_data.get(m, {}))
        src_breakdowns[m] = bd
        all_src |= set(bd.keys())
    for src in sorted(all_src):
        row = f"  {src:<24}" + "".join(
            f"{src_breakdowns[m].get(src, 0):>{col_w},}" for m in methods
        )
        print(row)
    print("═" * (26 + col_w * len(methods)) + "\n")


def print_overlap_table(methods: list[str], sym_data: dict[str, dict], k: int) -> None:
    feature_sets = {m: top_feature_names(sym_data.get(m, {}), k) for m in methods}
    print(f"  TOP-{k} FEATURE JACCARD OVERLAP (by_coefficient, nonzero only)")
    print("─" * (10 + 9 * len(methods)))
    header = f"  {'':6}" + "".join(f"{_SHORT_LABELS[m]:>9}" for m in methods)
    print(header)
    for ma in methods:
        row = f"  {_SHORT_LABELS[ma]:<6}"
        for mb in methods:
            j = jaccard(feature_sets[ma], feature_sets[mb])
            row += f"{j:>9.3f}"
        print(row)
    print()

    # shared core
    if len(methods) > 1:
        shared = set.intersection(*feature_sets.values())
        print(f"  Shared by ALL methods ({len(shared)} features):")
        for f in sorted(shared):
            print(f"    {f}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_funnel(methods: list[str], funnels: dict[str, dict], out_dir: Path) -> None:
    stages = [
        ("n_mined", "Mined"),
        ("n_after_abstraction", "After\nabstraction"),
        ("n_after_filter", "After filter\n(+OR pass-through)"),
        ("n_final", "Final\n(dedup)"),
        ("n_nonzero_coeff", "Learned\n(coeff>0)"),
    ]
    n_stages = len(stages)
    n_methods = len(methods)
    x = np.arange(n_stages)
    w = 0.15
    offsets = [w * (i - (n_methods - 1) / 2) for i in range(n_methods)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [funnels[method].get(key, 0) for key, _ in stages]
        bars = ax.bar(
            x + offsets[i],
            vals,
            w,
            label=_LABELS[method].replace("\n", " "),
            color=_COLORS[method],
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
    out = out_dir / "feature_funnel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_overlap_heatmap(
    methods: list[str], sym_data: dict[str, dict], k: int, out_dir: Path
) -> None:
    feature_sets = {m: top_feature_names(sym_data.get(m, {}), k) for m in methods}
    n = len(methods)
    matrix = np.zeros((n, n))
    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            matrix[i, j] = jaccard(feature_sets[ma], feature_sets[mb])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    labels = [_SHORT_LABELS[m] for m in methods]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
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


def plot_fp_analysis(methods: list[str], compare: dict, out_dir: Path) -> None:
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
    out = out_dir / "fp_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_type_breakdown(
    methods: list[str], sym_data: dict[str, dict], out_dir: Path
) -> None:
    all_types_set: set[str] = set()
    breakdowns = {}
    for m in methods:
        bd = mining_type_breakdown(sym_data.get(m, {}))
        breakdowns[m] = bd
        all_types_set |= set(bd.keys())
    all_types = sorted(all_types_set)

    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    for mtype in all_types:
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
    out = out_dir / "feature_type_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_coeff_vs_perm(
    methods: list[str], sym_data: dict[str, dict], k: int, out_dir: Path
) -> None:
    """Scatter agreement between top-k by_coefficient vs by_permutation sets (Jaccard)."""
    if len(methods) < 2:
        return

    coeff_sets = {
        m: top_feature_names(sym_data.get(m, {}), k, "by_coefficient") for m in methods
    }
    perm_sets = {
        m: top_feature_names(sym_data.get(m, {}), k, "by_permutation") for m in methods
    }

    agreement = [jaccard(coeff_sets[m], perm_sets[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(
        [_SHORT_LABELS[m] for m in methods],
        agreement,
        color=[_COLORS[m] for m in methods],
        alpha=0.85,
    )
    for bar, val in zip(bars, agreement):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
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
    out = out_dir / "coeff_vs_perm_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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


def main() -> None:
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Feature overlap and pipeline analysis across grouping methods."
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific grouping_compare scenario dir. Default: latest run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top features to use for Jaccard overlap (default: 25).",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = _ANALYSIS_BASE / f"feature_overlap_{ts}" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"analysis_{ts}.txt"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        compare, methods, sym_data = load_run(args.scenario, args.run_dir)
        if not methods:
            print("[error] No grouping methods found in the compare result.")
            sys.exit(1)
        print(f"  Methods present: {methods}\n")

        funnels = {m: feature_funnel(sym_data[m]) for m in methods if m in sym_data}

        print_funnel_table(methods, funnels)
        print_fp_table(methods, compare, sym_data)
        print_generalization_table(methods, sym_data)
        print_type_breakdown(methods, sym_data)
        print_overlap_table(methods, sym_data, args.top_k)

        print(f"\n[plots] Writing to {out_dir}")
        if funnels:
            plot_funnel(methods, funnels, out_dir)
        plot_overlap_heatmap(methods, sym_data, args.top_k, out_dir)
        plot_fp_analysis(methods, compare, out_dir)
        if sym_data:
            plot_type_breakdown(methods, sym_data, out_dir)
            plot_coeff_vs_perm(methods, sym_data, args.top_k, out_dir)

        print(f"\nAnalysis written to {out_dir}")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
