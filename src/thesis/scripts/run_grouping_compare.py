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

# Fixed AlertBERT hyperparameters — change here to tune for the scenario
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
) -> tuple[ExperimentResult, ExperimentResult, list]:
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir,
            grouping=grouping,
            grouping_cache_dir=grouping_cache_dir,
            results_dir=results_dir,
            transactions_dir=transactions_dir,
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
    for method in ["fixed_window", "time_delta", "alertbert"]:
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


def plot_metrics(data: dict, out_dir: Path) -> None:
    """2×2 grid: AUC / F1 / Precision / Recall, grouped by experiment type and method."""
    metric_pairs = [
        ("auc", "AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    exp_types = [("baseline", "Baseline"), ("symbolic", "Symbolic")]
    methods = ["fixed_window", "time_delta", "alertbert"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    x = np.arange(len(exp_types))
    w = 0.25
    offsets = [-w, 0, w]

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
                        fontsize=7,
                    )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in exp_types])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Detection metrics by grouping method — {data['scenario']}", fontsize=12
    )
    fig.tight_layout()
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


def plot_transactions(data: dict, out_dir: Path) -> None:
    """2×2 grid: transaction count, alerts/tx, itemset size, unique alert types."""
    methods = ["fixed_window", "time_delta", "alertbert"]
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
    out = out_dir / "transactions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_purity(data: dict, out_dir: Path) -> None:
    """Stacked horizontal bar: pure-benign / pure-attack / mixed per grouping method."""
    methods = ["fixed_window", "time_delta", "alertbert"]

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
    out = out_dir / "purity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


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
        results = {
            "fixed_window": {
                "baseline": _result_from_dict(
                    data["fixed_window"]["baseline"], scenario
                ),
                "symbolic": _result_from_dict(
                    data["fixed_window"]["symbolic"], scenario
                ),
            },
            "time_delta": {
                "baseline": _result_from_dict(data["time_delta"]["baseline"], scenario),
                "symbolic": _result_from_dict(data["time_delta"]["symbolic"], scenario),
            },
            "alertbert": {
                "baseline": _result_from_dict(data["alertbert"]["baseline"], scenario),
                "symbolic": _result_from_dict(data["alertbert"]["symbolic"], scenario),
            },
        }
        scenario_dir = existing.parent
        plot_dir = existing.parent.parent.parent / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrices(results, scenario, ts, scenario_dir)
        plot_data = {
            "scenario": scenario,
            "fixed_window": {
                "baseline": results["fixed_window"]["baseline"],
                "symbolic": results["fixed_window"]["symbolic"],
                "tx_stats": data["fixed_window"]["tx_stats"],
            },
            "time_delta": {
                "baseline": results["time_delta"]["baseline"],
                "symbolic": results["time_delta"]["symbolic"],
                "tx_stats": data["time_delta"]["tx_stats"],
            },
            "alertbert": {
                "baseline": results["alertbert"]["baseline"],
                "symbolic": results["alertbert"]["symbolic"],
                "tx_stats": data["alertbert"]["tx_stats"],
            },
        }
        plot_metrics(plot_data, plot_dir)
        plot_transactions(plot_data, plot_dir)
        plot_purity(plot_data, plot_dir)
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
    # Both methods share the main scenario cache for alerts and groups.
    cache_dir = CACHE_DIR / scenario
    alertbert_groups_dir = (
        CACHE_DIR / "alertbert_groups" / scenario / args.alertbert_model_id
    )
    if args.force_grouping:
        if alertbert_groups_dir.exists():
            shutil.rmtree(alertbert_groups_dir)
            print(f"  [force-grouping] Cleared {alertbert_groups_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    alertbert_groups_dir.mkdir(parents=True, exist_ok=True)

    grouping_fw = _fixed_grouping()
    grouping_td = _time_delta_grouping()
    grouping_ab = _alertbert_grouping(
        args.alertbert_model_id, args.alertbert_models_path
    )

    print(f"\n{'='*60}")
    print(f" Grouping comparison: {scenario}")
    print(f"{'='*60}")

    print("\n--- [1/3] fixed_window ---")
    baseline_fw, symbolic_fw, txs_fw = _run_pair(
        scenario,
        grouping_fw,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        transactions_dir=scenario_dir / "fixed_window" / "transactions",
    )

    gc.collect()

    print("\n--- [2/3] time_delta ---")
    baseline_td, symbolic_td, txs_td = _run_pair(
        scenario,
        grouping_td,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        transactions_dir=scenario_dir / "time_delta" / "transactions",
    )

    # Free allocations before loading torch + AlertBERT model.
    gc.collect()

    print("\n--- [3/3] alertbert ---")
    baseline_ab, symbolic_ab, txs_ab = _run_pair(
        scenario,
        grouping_ab,
        cache_dir,
        args.filter_config,
        results_dir=scenario_dir,
        grouping_cache_dir=alertbert_groups_dir,
        transactions_dir=scenario_dir / "alertbert" / "transactions",
    )

    stats_fw = _compute_tx_stats(txs_fw)
    stats_td = _compute_tx_stats(txs_td)
    stats_ab = _compute_tx_stats(txs_ab)

    timestamp = run_ts
    combined = {
        "experiment": "grouping_compare",
        "scenario": scenario,
        "timestamp": timestamp,
        "alertbert_config": grouping_ab.alertbert.model_dump(),
        "fixed_window": {
            "grouping": {"mode": "fixed_window", "params": None},
            "baseline": _result_to_dict(baseline_fw),
            "symbolic": _result_to_dict(symbolic_fw),
            "tx_stats": _stats_for_json(stats_fw),
        },
        "time_delta": {
            "grouping": {"mode": "time_delta", "params": None},
            "baseline": _result_to_dict(baseline_td),
            "symbolic": _result_to_dict(symbolic_td),
            "tx_stats": _stats_for_json(stats_td),
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
            "time_delta": {"baseline": baseline_td, "symbolic": symbolic_td},
            "alertbert": {"baseline": baseline_ab, "symbolic": symbolic_ab},
        },
        scenario,
        timestamp,
        scenario_dir,
    )

    # Summary table
    print(f"\n{'─'*72}")
    print(f"  {'':20s} {'fixed_window':>10} {'time_delta':>10} {'alertbert':>10}")
    print(f"{'─'*72}")
    for label, fw_val, td_val, ab_val in [
        ("baseline AUC", baseline_fw.auc, baseline_td.auc, baseline_ab.auc),
        ("symbolic AUC", symbolic_fw.auc, symbolic_td.auc, symbolic_ab.auc),
        (
            "baseline F1",
            baseline_fw.metrics.get("f1", float("nan")),
            baseline_td.metrics.get("f1", float("nan")),
            baseline_ab.metrics.get("f1", float("nan")),
        ),
        (
            "symbolic F1",
            symbolic_fw.metrics.get("f1", float("nan")),
            symbolic_td.metrics.get("f1", float("nan")),
            symbolic_ab.metrics.get("f1", float("nan")),
        ),
        (
            "n_transactions",
            float(stats_fw["n_transactions"]),
            float(stats_td["n_transactions"]),
            float(stats_ab["n_transactions"]),
        ),
    ]:
        fmt = ".4f" if label != "n_transactions" else ".0f"
        print(f"  {label:<20s} {fw_val:>10{fmt}} {td_val:>10{fmt}} {ab_val:>10{fmt}}")
    print(f"{'─'*72}")

    print(f"\n[plots] Saving to {plots_dir}")
    plot_data = {
        "scenario": scenario,
        "fixed_window": {
            "baseline": baseline_fw,
            "symbolic": symbolic_fw,
            "tx_stats": stats_fw,
        },
        "time_delta": {
            "baseline": baseline_td,
            "symbolic": symbolic_td,
            "tx_stats": stats_td,
        },
        "alertbert": {
            "baseline": baseline_ab,
            "symbolic": symbolic_ab,
            "tx_stats": stats_ab,
        },
    }

    plot_metrics(plot_data, plots_dir)
    plot_transactions(plot_data, plots_dir)
    plot_purity(plot_data, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
