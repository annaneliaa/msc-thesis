"""
Filter configuration sweep for the symbolic experiment.

Two modes (--mode):

  params   (default) — one-factor-at-a-time sweep of continuous mining
                       hyperparameters.  Each parameter is swept independently
                       while the others stay at their strict-filter defaults.
                       Results accumulate in results.csv so re-runs add new
                       sweep points without discarding old ones.

  presets  — run a fixed set of named filter conditions (none, default,
             discriminative, strict, benign_focused) side by side to compare
             how much filtering affects model performance and feature selection.

Usage:
    # OFAT hyperparameter sweep
    python src/thesis/scripts/run_filter_sweep.py fox
    python src/thesis/scripts/run_filter_sweep.py fox --params min_support max_overlap
    python src/thesis/scripts/run_filter_sweep.py fox --no-run   # plot from cache

    # Named filter condition comparison
    python src/thesis/scripts/run_filter_sweep.py fox --mode presets
    python src/thesis/scripts/run_filter_sweep.py fox --mode presets \\
        --conditions none,strict,benign_focused

Output (under artifacts/experiments/run_filter_sweep/):
  params mode:   params_<scenario>_<ts>/
                   scenario/<scenario>/results.csv
                   plots/<scenario>_sweep_<param>.png
                   plots/<scenario>_sweep_overview.png
  presets mode:  presets_<scenario>_<ts>/
                   <scenario>/filter_effect_<ts>.json
                   <scenario>/plots/feature_funnel.png
                   <scenario>/plots/performance.png
                   <scenario>/plots/fp_analysis.png
                   <scenario>/plots/feature_overlap.png
                   <scenario>/plots/feature_type_breakdown.png
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from thesis.config import GroupingConfig
from thesis.experiments.baseline import run_baseline_experiment
from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)
from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR
from thesis.schemas.experiments import BaselineExperimentConfig, ExperimentResult


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {here}")


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO / "src"))

_OUTPUT_BASE = _REPO / "artifacts" / "experiments" / "run_filter_sweep"


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
# PARAMS MODE (OFAT hyperparameter sweep)
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "min_support": 0.05,
    "min_support_count": 50,
    "min_abs_support_diff": 0.20,
    "min_confidence_attack": 0.0,
    "max_confidence_attack": None,
    "min_confidence_benign": 0.0,
    "max_overlap": 0.3,
    "min_lift": 2.0,
}

SWEEP_GRID: dict[str, list] = {
    "min_support": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    "max_overlap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    "max_confidence_attack": [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
    "min_confidence_benign": [0.0, 0.05, 0.10, 0.20, 0.30],
}

ALL_PARAMS = list(SWEEP_GRID.keys())

PARAM_LABELS = {
    "min_support": "min_support (mining threshold)",
    "max_overlap": "max_overlap (minority/majority ratio)",
    "min_confidence_attack": "min_confidence_attack",
    "min_confidence_benign": "min_confidence_benign",
}


def _write_filter_yaml(tmp_dir: Path, params: dict) -> Path:
    def _f(v):
        return float(v) if v is not None else None

    cfg = {
        "itemsets": {
            "min_k": 2,
            "max_k": None,
            "min_support_count": int(params["min_support_count"]),
            "min_abs_support_diff": float(params["min_abs_support_diff"]),
            "min_confidence_attack": float(params["min_confidence_attack"]),
            "max_confidence_attack": _f(params["max_confidence_attack"]),
            "min_confidence_benign": float(params["min_confidence_benign"]),
            "max_overlap": _f(params["max_overlap"]),
            "remove_subsumed": True,
        },
        "item_sequences": {
            "min_k": 3,
            "min_support_count": int(params["min_support_count"]),
            "min_abs_support_diff": float(params["min_abs_support_diff"]),
            "min_confidence_attack": float(params["min_confidence_attack"]),
            "max_confidence_attack": _f(params["max_confidence_attack"]),
            "min_confidence_benign": float(params["min_confidence_benign"]),
            "min_lift": float(params["min_lift"]),
            "max_overlap": _f(params["max_overlap"]),
            "remove_subsumed": True,
        },
        "feature_selection": {"top_k": None, "min_utility_score": None},
    }
    path = tmp_dir / "filter.yaml"
    with path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


def _run_params_point(
    scenario: str,
    sweep_param: str,
    value: float,
    tmp_dir: Path,
    alerts_json_path: Path | None = None,
    mine_frac: float = 1.0,
    no_overlap: bool = False,
    random_split: bool = False,
    random_seed: int = 42,
) -> dict:
    params = {**DEFAULTS, sweep_param: value}
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        filter_yaml = _write_filter_yaml(Path(td), params)
        cfg = SymbolicExperimentConfig(
            scenario=scenario,
            min_support=params["min_support"],
            filter_config=filter_yaml,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            model_name="logreg_sweep",
            model_version="0.1.0",
            alerts_json_path=alerts_json_path,
            mine_frac=mine_frac,
            no_overlap=no_overlap,
            random_split=random_split,
            random_seed=random_seed,
        )
        result = run_symbolic_experiment(cfg)
    n_sym = result.metrics.get("n_symbolic_features_used", result.n_features - 8)
    return {
        "scenario": scenario,
        "sweep_param": sweep_param,
        "value": value,
        "n_features_total": result.n_features,
        "n_symbolic_features": n_sym,
        "auc": result.auc,
        "balanced_accuracy": result.metrics.get("balanced_accuracy", float("nan")),
        "precision": result.metrics.get("precision", float("nan")),
        "recall": result.metrics.get("recall", float("nan")),
        "f1": result.metrics.get("f1", float("nan")),
        "tp": result.metrics.get("tp", float("nan")),
        "fp": result.metrics.get("fp", float("nan")),
        "tn": result.metrics.get("tn", float("nan")),
        "fn": result.metrics.get("fn", float("nan")),
    }


def _find_latest_params_csv(scenario: str) -> Path | None:
    candidates = sorted(
        _OUTPUT_BASE.glob(f"params_{scenario}_*/scenario/{scenario}/results.csv")
    )
    return candidates[-1] if candidates else None


def _load_params_cached(csv_path: Path, scenario: str | None = None) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if scenario is not None:
        latest = _find_latest_params_csv(scenario)
        if latest is not None:
            print(f"  [cache] Loading sweep results from previous run: {latest}")
            return pd.read_csv(latest)
    return pd.DataFrame()


def _is_params_cached(cached: pd.DataFrame, sweep_param: str, value: float) -> bool:
    if cached.empty:
        return False
    mask = (cached["sweep_param"] == sweep_param) & (
        cached["value"].round(6) == round(value, 6)
    )
    return bool(mask.any())


def _append_and_save(cached: pd.DataFrame, row: dict, csv_path: Path) -> pd.DataFrame:
    updated = pd.concat([cached, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    return updated


def _plot_sweep_param(
    df: pd.DataFrame,
    sweep_param: str,
    scenario: str,
    out_dir: Path,
    ax=None,
    filtered: bool = False,
) -> None:
    sub = df[df["sweep_param"] == sweep_param].sort_values("value")
    if sub.empty:
        return

    standalone = ax is None
    if standalone:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    else:
        ax1 = ax
        fig = ax.get_figure()

    color_feat = "#2196F3"
    color_auc = "#E91E63"
    color_f1 = "#4CAF50"

    ax1.bar(
        range(len(sub)),
        sub["n_symbolic_features"],
        color=color_feat,
        alpha=0.65,
        label="symbolic features",
    )
    ax1.set_ylabel("# symbolic features (post-filter)", color=color_feat)
    ax1.tick_params(axis="y", labelcolor=color_feat)
    ax1.set_xticks(range(len(sub)))
    ax1.set_xticklabels([f"{v:.3g}" for v in sub["value"]], rotation=30, ha="right")
    ax1.set_xlabel(PARAM_LABELS.get(sweep_param, sweep_param))

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(sub)),
        sub["auc"],
        marker="o",
        color=color_auc,
        linewidth=2,
        label="AUC",
    )
    ax2.plot(
        range(len(sub)),
        sub["f1"],
        marker="s",
        color=color_f1,
        linewidth=2,
        linestyle="--",
        label="F1",
    )
    ax2.set_ylabel("Score", color="black")
    ax2.set_ylim(0, 1.05)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    title = f"{scenario} — sweep: {sweep_param}"
    if standalone:
        ax1.set_title(title)
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
        out_path = out_dir / f"{scenario}_sweep_{sweep_param}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out_path}")
    else:
        ax1.set_title(title, fontsize=10)


def _plot_all_params_sweeps(
    df: pd.DataFrame,
    scenario: str,
    out_dir: Path,
    params: list[str],
    filtered: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for param in params:
        _plot_sweep_param(df, param, scenario, out_dir, filtered=filtered)

    n = len(params)
    if n < 2:
        return
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, param in enumerate(params):
        _plot_sweep_param(
            df, param, scenario, out_dir, ax=axes_flat[i], filtered=filtered
        )
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{scenario} — hyperparameter sweep overview", fontsize=13)
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
    out_path = out_dir / f"{scenario}_sweep_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _params_body(args: argparse.Namespace) -> None:
    scenario = args.scenario
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_BASE / f"params_{scenario}_{run_ts}"
    scenario_dir = run_dir / "scenario" / scenario
    plots_dir = run_dir / "plots"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scenario_dir / "results.csv"
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    log_path = plots_dir / f"sweep_{run_ts}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")

    try:
        alerts_json_path = (
            _REPO / "artifacts" / "processed-data" / scenario / "alerts_filtered.json"
            if args.filtered
            else None
        )
        cached = _load_params_cached(
            csv_path, scenario=scenario if args.no_run else None
        )

        if not args.no_run:
            for param in args.params:
                values = SWEEP_GRID[param]
                print(f"\n{'='*60}\n Sweep: {param}  ({len(values)} values)\n{'='*60}")
                for value in values:
                    if not args.force and _is_params_cached(cached, param, value):
                        print(f"  [{param}={value:.4g}] cached — skipping.")
                        continue
                    print(f"\n  [{param}={value:.4g}] running...")
                    row = _run_params_point(
                        scenario,
                        param,
                        value,
                        tmp_dir,
                        alerts_json_path=alerts_json_path,
                        mine_frac=args.mine_frac,
                        no_overlap=args.no_overlap,
                        random_split=args.random_split,
                        random_seed=args.random_seed,
                    )
                    cached = _append_and_save(cached, row, csv_path)
                    print(
                        f"  [{param}={value:.4g}] AUC={row['auc']:.4f}  "
                        f"sym_features={row['n_symbolic_features']}"
                    )

        if cached.empty:
            print("No results available. Exiting.")
            return

        print(f"\n[results] {len(cached)} sweep points in {csv_path}")
        params_with_data = [p for p in args.params if p in cached["sweep_param"].values]
        print(f"[plots] Generating plots for: {params_with_data}")
        _plot_all_params_sweeps(
            cached, scenario, plots_dir, params_with_data, filtered=bool(args.filtered)
        )
        print("\nDone.")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


# ---------------------------------------------------------------------------
# PRESETS MODE (named filter conditions)
# ---------------------------------------------------------------------------

_CONFIGS_DIR = _REPO / "src" / "thesis" / "configs"

_FILTER_CONFIG_FILES: dict[str, Path | None] = {
    "none": None,
    "default": _CONFIGS_DIR / "mining_filters_default.yaml",
    "discriminative": _CONFIGS_DIR / "mining_filters_discriminative.yaml",
    "strict": _CONFIGS_DIR / "mining_filters_strict.yaml",
    "benign_focused": _CONFIGS_DIR / "mining_filters_benign_focused.yaml",
}

_ALL_CONDITIONS = list(_FILTER_CONFIG_FILES.keys())

_PRESET_COLORS = {
    "none": "#BBBBBB",
    "default": "#9EC8E8",
    "discriminative": "#4C72B0",
    "strict": "#1A3A6E",
    "benign_focused": "#55A868",
}

_PRESET_LABELS = {
    "none": "None\n(all mined)",
    "default": "Default\n(noise only)",
    "discriminative": "Discriminative\n(moderate)",
    "strict": "Strict\n(aggressive)",
    "benign_focused": "Benign-\nfocused",
}


def _i(v: Any) -> int:
    return int(v) if v is not None else 0


def _preset_feature_funnel(sym_path: Path) -> dict:
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
    n_abs_parts = (
        _i(m.get("n_itemsets_after_abstraction"))
        + _i(m.get("n_sequences_after_abstraction"))
        + _i(m.get("n_or_after_abstraction"))
    )
    n_after_abstraction = n_abs_parts if n_abs_parts > 0 else n_mined
    return {
        "n_mined": n_mined,
        "n_after_abstraction": n_after_abstraction,
        "n_after_filter": _i(m.get("n_candidate_features")),
        "n_final": _i(m.get("n_features_final")),
        "n_nonzero_coeff": sum(1 for v in top_coeff.values() if v["importance"] > 0),
        "n_nonzero_perm": sum(1 for v in top_perm.values() if v["importance"] > 0),
    }


def _preset_top_feature_names(
    sym_path: Path, k: int, by: str = "by_coefficient"
) -> set[str]:
    with sym_path.open() as f:
        sym = json.load(f)
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    ranked = sorted(top.items(), key=lambda kv: kv[1]["importance"], reverse=True)
    return {name for name, info in ranked[:k] if info["importance"] > 0}


def _preset_jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _preset_mining_type_breakdown(sym_path: Path) -> dict[str, int]:
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


def _result_to_dict(r: ExperimentResult) -> dict:
    return {
        "schema_name": r.schema_name,
        "schema_version": r.schema_version,
        "grouping_mode": r.grouping_mode,
        "auc": r.auc,
        "n_alert_groups": r.n_alert_groups,
        "n_mixed_dropped": r.n_mixed_dropped,
        "n_features": r.n_features,
        "metrics": r.metrics,
        "results_file": str(r.results_file),
    }


def _print_presets_funnel_table(
    conditions: list[str], funnels: dict[str, dict]
) -> None:
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
        print(
            f"  {label:<24}"
            + "".join(f"{funnels.get(c, {}).get(key, 0):>{cw},}" for c in conditions)
        )
    print("═" * w + "\n")


def _print_presets_performance_table(
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
            row += f"{int(results[c].metrics.get(key, 0)):>{cw},}"
        print(row)
    print("─" * w)
    row = f"  {'FP delta vs baseline':<24}{'':>{cw}}"
    for c in conditions:
        delta = int(results[c].metrics.get("fp", 0)) - int(
            baseline.metrics.get("fp", 0)
        )
        row += f"{delta:>+{cw},}"
    print(row)
    print("═" * w + "\n")


def _plot_presets_funnel(
    conditions: list[str],
    funnels: dict[str, dict],
    out_dir: Path,
    filtered: bool = False,
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
        bars = ax.bar(x + offsets[i], vals, w, label=cond, color=_PRESET_COLORS[cond])
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


def _plot_presets_performance(
    conditions: list[str],
    results: dict[str, ExperimentResult],
    baseline: ExperimentResult,
    out_dir: Path,
    filtered: bool = False,
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
        vals = [
            result.metrics.get(key, result.auc if key == "auc" else float("nan"))
            for key, _ in metrics
        ]
        color = "#5B9BD5" if label == "baseline" else _PRESET_COLORS[label]
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
    out = out_dir / "performance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_presets_fp_analysis(
    conditions: list[str],
    results: dict[str, ExperimentResult],
    baseline: ExperimentResult,
    out_dir: Path,
    filtered: bool = False,
) -> None:
    all_labels = ["baseline"] + conditions
    fp_vals = [baseline.metrics.get("fp", 0)] + [
        results[c].metrics.get("fp", 0) for c in conditions
    ]
    prec_vals = [baseline.metrics.get("precision", float("nan"))] + [
        results[c].metrics.get("precision", float("nan")) for c in conditions
    ]
    x = np.arange(len(all_labels))
    colors = ["#5B9BD5"] + [_PRESET_COLORS[c] for c in conditions]

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


def _plot_presets_overlap_heatmap(
    conditions: list[str],
    sym_paths: dict[str, Path],
    k: int,
    out_dir: Path,
    filtered: bool = False,
) -> None:
    feature_sets = {c: _preset_top_feature_names(sym_paths[c], k) for c in conditions}
    n = len(conditions)
    matrix = np.zeros((n, n))
    for i, ca in enumerate(conditions):
        for j, cb in enumerate(conditions):
            matrix[i, j] = _preset_jaccard(feature_sets[ca], feature_sets[cb])

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


def _plot_presets_type_breakdown(
    conditions: list[str],
    sym_paths: dict[str, Path],
    out_dir: Path,
    filtered: bool = False,
) -> None:
    all_types_set: set[str] = set()
    bds = {}
    for c in conditions:
        bd = _preset_mining_type_breakdown(sym_paths[c])
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


def _presets_body(args: argparse.Namespace) -> None:
    scenario = args.scenario
    conditions = [c.strip() for c in args.conditions.split(",")]
    unknown = [c for c in conditions if c not in _FILTER_CONFIG_FILES]
    if unknown:
        print(f"[error] Unknown conditions: {unknown}. Options: {_ALL_CONDITIONS}")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_BASE / f"presets_{scenario}_{ts}" / scenario
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"analysis_{ts}.txt"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        grouping = GroupingConfig(mode=args.grouping)
        alerts_json_path = (
            _REPO / "artifacts" / "processed-data" / scenario / "alerts_filtered.json"
            if args.filtered
            else None
        )
        method_cache_dir = CACHE_DIR / scenario / "groups" / args.grouping
        method_cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{'='*60}\n Filter preset experiment: {scenario}  grouping={args.grouping}"
        )
        print(f" Conditions: {conditions}\n{'='*60}\n")

        print("--- [baseline] ---")
        baseline = run_baseline_experiment(
            BaselineExperimentConfig(
                scenario=scenario,
                cache_dir=method_cache_dir,
                grouping=grouping,
                results_dir=run_dir,
                alerts_json_path=alerts_json_path,
                random_split=args.random_split,
                random_seed=args.random_seed,
            )
        )
        gc.collect()

        sym_results: dict[str, ExperimentResult] = {}
        sym_paths: dict[str, Path] = {}

        for i, cond in enumerate(conditions):
            print(f"\n--- [{i+1}/{len(conditions)}] condition: {cond} ---")
            result = run_symbolic_experiment(
                SymbolicExperimentConfig(
                    scenario=scenario,
                    cache_dir=method_cache_dir,
                    grouping=grouping,
                    filter_config=_FILTER_CONFIG_FILES[cond],
                    abstraction_map_path=ABSTRACTION_MAP_PATH,
                    results_dir=run_dir,
                    alerts_json_path=alerts_json_path,
                    mine_frac=args.mine_frac,
                    no_overlap=args.no_overlap,
                    random_split=args.random_split,
                    random_seed=args.random_seed,
                )
            )
            sym_results[cond] = result
            sym_paths[cond] = Path(result.results_file)
            gc.collect()

        combined: dict[str, Any] = {
            "experiment": "filter_presets",
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

        funnels = {c: _preset_feature_funnel(sym_paths[c]) for c in conditions}
        _print_presets_funnel_table(conditions, funnels)
        _print_presets_performance_table(conditions, sym_results, baseline)

        print(f"[plots] Writing to {plots_dir}")
        _plot_presets_funnel(
            conditions, funnels, plots_dir, filtered=bool(args.filtered)
        )
        _plot_presets_performance(
            conditions, sym_results, baseline, plots_dir, filtered=bool(args.filtered)
        )
        _plot_presets_fp_analysis(
            conditions, sym_results, baseline, plots_dir, filtered=bool(args.filtered)
        )
        _plot_presets_overlap_heatmap(
            conditions, sym_paths, args.top_k, plots_dir, filtered=bool(args.filtered)
        )
        _plot_presets_type_breakdown(
            conditions, sym_paths, plots_dir, filtered=bool(args.filtered)
        )

        print(f"\nDone. Output in {run_dir}")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter configuration sweep for the symbolic experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--mode",
        choices=["params", "presets"],
        default="params",
        help=(
            "Sweep mode: 'params' (OFAT continuous hyperparameter sweep, default) or "
            "'presets' (compare named filter conditions side by side)."
        ),
    )

    # Shared
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Use detector-filtered alerts (alerts_filtered.json) instead of alerts.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached results exist.",
    )

    # Params-mode specific
    parser.add_argument(
        "--params",
        nargs="+",
        choices=ALL_PARAMS,
        default=ALL_PARAMS,
        help="[params mode] Which parameters to sweep (default: all four).",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="[params mode] Skip running experiments; plot from cached results only.",
    )

    # Presets-mode specific
    parser.add_argument(
        "--conditions",
        default=",".join(_ALL_CONDITIONS),
        help=f"[presets mode] Comma-separated filter conditions (default: all). Options: {_ALL_CONDITIONS}",
    )
    parser.add_argument(
        "--grouping",
        default="fixed_window",
        choices=["fixed_window", "fixed_window_host", "time_delta", "time_delta_host"],
        help="[presets mode] Grouping method to use (default: fixed_window).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="[presets mode] Number of top features for Jaccard overlap (default: 25).",
    )
    parser.add_argument(
        "--mine-frac",
        type=float,
        default=1.0,
        dest="mine_frac",
        help="Fraction of alert_groups (sorted by time) to use for mining (default: 1.0 = all).",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        dest="no_overlap",
        help="Exclude the mining window from training data (train starts after mine_frac).",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        dest="random_split",
        help="Shuffle alert_groups randomly before any split instead of using temporal order.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        dest="random_seed",
        metavar="SEED",
        help="Random seed for --random-split (default: 42).",
    )

    args = parser.parse_args()

    if args.mode == "params":
        _params_body(args)
    else:
        _presets_body(args)


if __name__ == "__main__":
    main()
