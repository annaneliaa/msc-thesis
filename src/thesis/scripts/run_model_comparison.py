"""
Compare models (logistic regression vs MLP) across scenarios (baseline + symbolic, optional filtered data).


For each model (logreg, mlp), runs the full pipeline per scenario:
each run under two conditions: baseline (base features only) and symbolic (base + mined features from Eclat/PrefixSpan).

It then produces:
  - Performance comparison table and plots (AUC, F1, precision, recall, FP counts)
  - Per-model feature analysis: funnel, overlap heatmaps, SHAP plots (reuses run_scenario_feature_analysis functions)
  - Cross-model feature analysis (SHAP overlap, rank agreement, feature consistency)

Usage:
  python src/thesis/scripts/run_model_comparison.py fox wheeler harrison --filtered
  python src/thesis/scripts/run_model_comparison.py --all --filtered naive50
  python src/thesis/scripts/run_model_comparison.py --all --no-run --filtered

Output (under artifacts/experiments/run_model_comparison/comparison_<ts>/):
  logreg/scenario/<scenario>/       logreg compare, baseline, symbolic JSONs
  mlp/scenario/<scenario>/          mlp compare, baseline, symbolic JSONs
  plots/
    comparison_table.txt / .csv
    perf_auc.png              AUC: logreg vs mlp, baseline + symbolic
    perf_metrics.png          F1/precision/recall/ba for baseline
    logreg/                   per-model feature analysis plots
    mlp/
    cross_model/
      jaccard_spearman.png    per-scenario Jaccard + Spearman rank (logreg vs mlp)
      shap_bars_<model>.png     signed SHAP importance across scenarios per model
      feature_heatmap_logreg.png  which features appear across scenarios
      feature_heatmap_mlp.png
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.experiments.baseline import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)
from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)
from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR
from thesis.visualization.eda import SCENARIOS as ALL_SCENARIOS
from thesis.xai.overlap import jaccard as _jaccard, spearman_rank, top_k_names

import matplotlib

matplotlib.use("Agg")

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

COMPARISON_BASE = _REPO / "artifacts" / "experiments" / "run_model_comparison"
MODELS = ["logreg", "mlp", "lstm"]
_MODEL_COLORS = {"logreg": "#4C72B0", "mlp": "#DD8452", "lstm": "#55A868"}
_MODEL_LABELS = {"logreg": "LogReg", "mlp": "MLP", "lstm": "LSTM"}


def _data_label(filtered: bool, method: str | None = None) -> str:
    if not filtered:
        return "data: raw"
    return f"data: {method}" if method else "data: filtered"


# ---------------------------------------------------------------------------
# Tee logger
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
# Phase 1: Running experiments
# ---------------------------------------------------------------------------


def _run_for_model(
    scenario: str,
    model_name: str,
    model_dir: Path,
    filter_config: Path | None,
    alerts_json_path: Path | None,
    cache_dir: Path | None,
) -> dict:
    """Run baseline + symbolic for one scenario/model, write compare JSON."""
    print(f"\n{'='*60}\n  {model_name.upper()} — {scenario}\n{'='*60}")

    scenario_dir = model_dir / "scenario" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    extra = {"cache_dir": cache_dir} if cache_dir is not None else {}

    print("\n--- baseline ---")
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            model_name=model_name,
            results_dir=scenario_dir,
            alerts_json_path=alerts_json_path,
            **extra,
        )
    )

    print("\n--- symbolic ---")
    symbolic = run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            filter_config=filter_config,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            model_name=model_name,
            results_dir=scenario_dir,
            alerts_json_path=alerts_json_path,
            **extra,
        )
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _nan(v):
        return None if v != v else v

    combined = {
        "experiment": "compare",
        "scenario": scenario,
        "timestamp": ts,
        "model_name": model_name,
        "filter_config": str(filter_config),
        "filtered": alerts_json_path is not None,
        "baseline": {
            "schema_name": baseline.schema_name,
            "schema_version": baseline.schema_version,
            "auc": _nan(baseline.auc),
            "n_features": baseline.n_features,
            "n_transactions": baseline.n_transactions,
            "metrics": baseline.metrics,
            "results_file": str(baseline.results_file),
        },
        "symbolic": {
            "schema_name": symbolic.schema_name,
            "schema_version": symbolic.schema_version,
            "auc": _nan(symbolic.auc),
            "n_features": symbolic.n_features,
            "n_transactions": symbolic.n_transactions,
            "metrics": symbolic.metrics,
            "results_file": str(symbolic.results_file),
        },
    }
    out = scenario_dir / f"compare_{ts}.json"
    with out.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved → {out}")

    return {
        "scenario": scenario,
        "model_name": model_name,
        "baseline": {
            **baseline.metrics,
            "n_features": baseline.n_features,
            "n_transactions": baseline.n_transactions,
        },
        "symbolic": {
            **symbolic.metrics,
            "n_features": symbolic.n_features,
            "n_transactions": symbolic.n_transactions,
        },
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_compare_json(model_dir: Path, scenario: str) -> dict | None:
    candidates = sorted((model_dir / "scenario" / scenario).glob("compare_*.json"))
    if not candidates:
        return None
    with candidates[-1].open() as f:
        return json.load(f)


def _load_importance(
    results_file: str | Path, key: str = "by_shap"
) -> dict[str, float]:
    """Load {feature: importance} from a results JSON. Handles enriched dict format."""
    path = Path(results_file)
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    imp = data.get("metrics", {}).get("top_feature_importances", {}).get(key, {})
    if not imp:
        return {}
    first = next(iter(imp.values()))
    if isinstance(first, dict):
        return {k: v["importance"] for k, v in imp.items()}
    return imp


def _collect_importances(
    run_dir: Path, scenarios: list[str], experiment: str = "baseline"
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns {model: {scenario: {feature: importance}}}.
    Prefers by_shap; falls back to by_permutation if empty.
    """
    result: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        result[model] = {}
        model_dir = run_dir / model
        for scenario in scenarios:
            cmp = _load_compare_json(model_dir, scenario)
            if cmp is None:
                result[model][scenario] = {}
                continue
            rf = cmp.get(experiment, {}).get("results_file")
            if not rf:
                result[model][scenario] = {}
                continue
            imp = _load_importance(rf, "by_shap")
            if not imp:
                imp = _load_importance(rf, "by_permutation")
                if imp:
                    print(f"  [info] {model}/{scenario}: SHAP empty, using permutation")
            result[model][scenario] = imp
    return result


def _load_all_results(run_dir: Path, scenarios: list[str]) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {m: [] for m in MODELS}
    for model in MODELS:
        model_dir = run_dir / model
        for scenario in scenarios:
            cmp = _load_compare_json(model_dir, scenario)
            if cmp is None:
                continue
            all_results[model].append(
                {
                    "scenario": scenario,
                    "model_name": model,
                    "baseline": {
                        **cmp.get("baseline", {}).get("metrics", {}),
                        "n_features": cmp.get("baseline", {}).get("n_features", 0),
                        "n_transactions": cmp.get("baseline", {}).get(
                            "n_transactions", 0
                        ),
                    },
                    "symbolic": {
                        **cmp.get("symbolic", {}).get("metrics", {}),
                        "n_features": cmp.get("symbolic", {}).get("n_features", 0),
                        "n_transactions": cmp.get("symbolic", {}).get(
                            "n_transactions", 0
                        ),
                    },
                    "filtered": cmp.get("filtered", False),
                }
            )
    return all_results


# ---------------------------------------------------------------------------
# Phase 2: Performance comparison table & plots
# ---------------------------------------------------------------------------

_PERF_METRICS = [
    "auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "fp",
    "tp",
    "fn",
    "n_features",
]


def _build_comparison_df(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model, results in all_results.items():
        for r in results:
            for exp in ("baseline", "symbolic"):
                m = r[exp]
                row = {"model": model, "scenario": r["scenario"], "experiment": exp}
                for k in _PERF_METRICS:
                    row[k] = m.get(k, float("nan"))
                rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["auc", "f1", "precision", "recall", "balanced_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def _format_text_table(df: pd.DataFrame) -> str:
    base = df[df["experiment"] == "baseline"]
    sym = df[df["experiment"] == "symbolic"]
    scenarios = sorted(df["scenario"].unique())

    w = 9
    sep = "─" * (16 + (w * 3 + 4) * len(MODELS) * 2)
    parts = [sep]
    header = f"  {'scenario':<14}"
    for exp_label, src in [("base", base), ("sym", sym)]:
        for m in MODELS:
            header += (
                f"  {('[' + _MODEL_LABELS[m] + '|' + exp_label + ']'):<{w * 3 + 2}}"
            )
    parts.append(header)

    subheader = f"  {'':14}"
    for _ in range(2):
        for _ in MODELS:
            subheader += f"  {'AUC':>{w}}{'F1':>{w}}{'FP':>{w}}"
    parts.append(subheader)
    parts.append(sep)

    for scenario in scenarios:
        row = f"  {scenario:<14}"
        for src in (base, sym):
            for m in MODELS:
                r = src[(src["scenario"] == scenario) & (src["model"] == m)]
                if r.empty:
                    row += f"  {'?':>{w}}{'?':>{w}}{'?':>{w}}"
                else:
                    auc = r.iloc[0]["auc"]
                    f1 = r.iloc[0]["f1"]
                    fp = r.iloc[0].get("fp", float("nan"))
                    row += f"  {auc:>{w}.4f}{f1:>{w}.4f}{int(fp) if not np.isnan(fp) else '?':>{w}}"
        parts.append(row)
    parts.append(sep)
    return "\n".join(parts)


def plot_perf_auc(
    df: pd.DataFrame, out_dir: Path, filtered: bool, method: str | None = None
) -> None:
    scenarios = sorted(df["scenario"].unique())
    n_sc = len(scenarios)
    n_bars = len(MODELS) * 2
    x = np.arange(n_sc)
    w = 0.18
    offsets = [w * (i - (n_bars - 1) / 2) for i in range(n_bars)]

    fig, ax = plt.subplots(figsize=(max(8, n_sc * 2.2), 5))
    bar_idx = 0
    for model in MODELS:
        for exp, suffix, alpha, hatch in [
            ("baseline", "base", 0.55, "//"),
            ("symbolic", "sym", 0.85, ""),
        ]:
            src = df[df["experiment"] == exp]
            vals = []
            for s in scenarios:
                r = src[(src["scenario"] == s) & (src["model"] == model)]["auc"]
                vals.append(float(r.iloc[0]) if len(r) else float("nan"))
            bars = ax.bar(
                x + offsets[bar_idx],
                vals,
                w,
                label=f"{_MODEL_LABELS[model]} ({suffix})",
                color=_MODEL_COLORS[model],
                alpha=alpha,
                hatch=hatch,
            )
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=45,
                    )
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC")
    ax.set_title("AUC comparison: LogReg vs MLP")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(filtered, method),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "perf_auc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_perf_metrics(
    df: pd.DataFrame, out_dir: Path, filtered: bool, method: str | None = None
) -> None:
    """Precision / recall / F1 / balanced_accuracy for baseline, models side by side."""
    base = df[df["experiment"] == "baseline"]
    scenarios = sorted(df["scenario"].unique())
    n_sc = len(scenarios)
    x = np.arange(n_sc)
    w = 0.35

    metrics = [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, n_sc * 2.2), 8))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        for i, model in enumerate(MODELS):
            offset = w * (i - 0.5)
            vals = []
            for s in scenarios:
                r = base[(base["scenario"] == s) & (base["model"] == model)][key]
                vals.append(float(r.iloc[0]) if len(r) else float("nan"))
            bars = ax.bar(
                x + offset,
                vals,
                w,
                label=_MODEL_LABELS[model],
                color=_MODEL_COLORS[model],
                alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline metrics: LogReg vs MLP")
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(filtered, method),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "perf_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Phase 3: Per-model feature analysis (reuses run_scenario_feature_analysis)
# ---------------------------------------------------------------------------


def _load_sfa():
    spec = importlib.util.spec_from_file_location(
        "run_scenario_feature_analysis",
        _HERE.parent / "run_scenario_feature_analysis.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_per_model_feature_analysis(
    model_name: str, model_dir: Path, scenarios: list[str], top_k: int, out_dir: Path
) -> None:
    sfa = _load_sfa()
    compare, sym_data = sfa.load_all(model_dir, scenarios)
    present = [s for s in scenarios if s in compare]
    if not present:
        print(f"  [warn] No loaded data for {model_name}.")
        return
    filtered = any(compare[s].get("filtered", False) for s in present)
    funnels = {s: sfa.feature_funnel(sym_data[s]) for s in present if s in sym_data}

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    sfa.print_funnel_table(present, funnels)
    sfa.print_overlap_table(present, sym_data, top_k)

    if funnels:
        sfa.plot_funnel(present, funnels, filtered, model_out)
    sfa.plot_fp_analysis(present, compare, filtered, model_out)
    if sym_data:
        sfa.plot_type_breakdown(present, sym_data, filtered, model_out)
        if len(present) > 1:
            sfa.plot_overlap_heatmap(present, sym_data, top_k, filtered, model_out)
        sfa.plot_coeff_vs_perm(present, sym_data, top_k, filtered, model_out)
        if model_name in ("logreg", "logreg_l1", "logreg_sweep"):
            sfa.plot_signed_coefficients(present, sym_data, top_k, filtered, model_out)
            sfa.plot_sign_split_breakdown(present, sym_data, filtered, model_out)


# ---------------------------------------------------------------------------
# Phase 4: Cross-model feature analysis
# ---------------------------------------------------------------------------


def _print_overlap_table(
    all_imps: dict[str, dict[str, dict[str, float]]],
    scenarios: list[str],
    top_k: int,
) -> None:
    pairs = list(itertools.combinations(MODELS, 2))
    if not pairs:
        return
    cw = 12
    for ma, mb in pairs:
        print(
            f"\n  TOP-{top_k} FEATURE OVERLAP: {_MODEL_LABELS[ma]} vs {_MODEL_LABELS[mb]}"
        )
        print("─" * (26 + cw * 3))
        print(
            f"  {'scenario':<24}{'Jaccard':>{cw}}{'Shared N':>{cw}}{'Spearman ρ':>{cw}}"
        )
        print("─" * (26 + cw * 3))
        for s in scenarios:
            a = all_imps[ma].get(s, {})
            b = all_imps[mb].get(s, {})
            ta = top_k_names(a, top_k)
            tb = top_k_names(b, top_k)
            jac = _jaccard(ta, tb)
            n_sh = len(ta & tb)
            sp = spearman_rank(a, b)
            jac_s = f"{jac:>{cw}.4f}" if not np.isnan(jac) else f"{'—':>{cw}}"
            sp_s = f"{sp:.4f}" if not np.isnan(sp) else "?"
            print(f"  {s:<24}{jac_s}{n_sh:>{cw}}{sp_s:>{cw}}")
        print("─" * (26 + cw * 3))

    for model in MODELS:
        tops = [
            top_k_names(all_imps[model].get(s, {}), top_k)
            for s in scenarios
            if all_imps[model].get(s)
        ]
        nonempty = [t for t in tops if t]
        if nonempty:
            core = set.intersection(*nonempty)
            print(
                f"\n  Core (all scenarios with features, {_MODEL_LABELS[model]}): {len(core)} features"
            )
            for f in sorted(core):
                print(f"    {f}")
        else:
            print(
                f"\n  Core ({_MODEL_LABELS[model]}): no learned features found across scenarios"
            )
    print()


def plot_cross_model_overlap(
    all_imps: dict[str, dict[str, dict[str, float]]],
    scenarios: list[str],
    top_k: int,
    filtered: bool,
    out_dir: Path,
    method: str | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
) -> None:
    ma = model_a or MODELS[0]
    mb = model_b or MODELS[1]

    sc = [
        s
        for s in scenarios
        if all_imps.get(ma, {}).get(s) or all_imps.get(mb, {}).get(s)
    ]
    jacs = [
        _jaccard(
            top_k_names(all_imps[ma].get(s, {}), top_k),
            top_k_names(all_imps[mb].get(s, {}), top_k),
        )
        for s in sc
    ]
    shared = [
        len(
            top_k_names(all_imps[ma].get(s, {}), top_k)
            & top_k_names(all_imps[mb].get(s, {}), top_k)
        )
        for s in sc
    ]
    spears = [
        spearman_rank(all_imps[ma].get(s, {}), all_imps[mb].get(s, {})) for s in sc
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(9, len(sc) * 1.6), 4))

    ax1.bar(
        sc, [v if not np.isnan(v) else 0 for v in jacs], color="#8172B3", alpha=0.85
    )
    for i, (v, c) in enumerate(zip(jacs, shared)):
        txt = f"—\n(n={c})" if np.isnan(v) else f"{v:.2f}\n(n={c})"
        ax1.text(
            i,
            (v if not np.isnan(v) else 0) + 0.02,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("Jaccard similarity")
    ax1.set_title(
        f"Top-{top_k} feature overlap\n({_MODEL_LABELS[ma]} vs {_MODEL_LABELS[mb]})"
    )
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    ax2.bar(sc, spears, color="#55A868", alpha=0.85)
    for i, v in enumerate(spears):
        if not np.isnan(v):
            ax2.text(
                i, max(v, 0) + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8
            )
    ax2.set_ylim(-0.15, 1.2)
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title(
        f"Feature rank agreement\n({_MODEL_LABELS[ma]} vs {_MODEL_LABELS[mb]})"
    )
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(filtered, method),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / f"jaccard_spearman_{ma}_vs_{mb}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_shap_importance_bars(
    all_imps: dict[str, dict[str, dict[str, float]]],
    scenarios: list[str],
    top_k: int,
    filtered: bool,
    out_dir: Path,
    method: str | None = None,
) -> None:
    """Per model: one figure with all scenarios as subplots, showing signed mean SHAP."""
    from matplotlib.patches import Patch

    for model in MODELS:
        present = [s for s in scenarios if all_imps.get(model, {}).get(s)]
        if not present:
            continue

        n = len(present)
        n_cols = min(n, 3)
        n_rows = (n + n_cols - 1) // n_cols
        fig_h = max(top_k * 0.28 + 1.5, 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, fig_h * n_rows))
        axes_flat: list = np.array(axes).flatten().tolist() if n > 1 else [axes]

        for ax_idx, sc in enumerate(present):
            ax = axes_flat[ax_idx]
            imp = all_imps[model][sc]
            ranked = sorted(imp.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top = [(name, val) for name, val in ranked if val != 0][:top_k][::-1]
            names = [r[0][:50] for r in top]
            values = [r[1] for r in top]
            colors = ["#4C72B0" if v > 0 else "#C94040" for v in values]
            y = np.arange(len(names))
            ax.barh(y, values, color=colors, alpha=0.85, height=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(sc, fontsize=9)
            ax.set_xlabel("Mean SHAP value", fontsize=8)
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
            f"Top-{top_k} features by |SHAP| per scenario — {_MODEL_LABELS[model]}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.text(
            0.99,
            0.01,
            _data_label(filtered, method),
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )
        out = out_dir / f"shap_bars_{model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_cross_scenario_heatmap(
    imps_by_scenario: dict[str, dict[str, float]],
    model_name: str,
    top_k: int,
    filtered: bool,
    out_dir: Path,
    method: str | None = None,
) -> None:
    """Presence/absence heatmap: which features are in top-K across scenarios."""
    scenarios = [s for s, imp in imps_by_scenario.items() if imp]
    if not scenarios:
        return

    top_per_sc = {s: top_k_names(imps_by_scenario[s], top_k) for s in scenarios}
    all_features = set().union(*top_per_sc.values())
    if not all_features:
        return

    features = sorted(
        all_features,
        key=lambda f: sum(1 for tops in top_per_sc.values() if f in tops),
        reverse=True,
    )

    matrix = np.array(
        [[1.0 if f in top_per_sc[s] else 0.0 for s in scenarios] for f in features]
    )

    fig_h = max(len(features) * 0.24 + 2, 5)
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.3), fig_h))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f[:60] for f in features], fontsize=7)
    ax.set_title(
        f"Feature consistency across scenarios — {_MODEL_LABELS[model_name]}\n(top-{top_k} SHAP/permutation)"
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Feature (sorted by frequency)")

    freqs = [sum(1 for tops in top_per_sc.values() if f in tops) for f in features]
    for i, freq in enumerate(freqs):
        ax.text(
            len(scenarios) - 0.4,
            i,
            f"{freq}/{len(scenarios)}",
            ha="left",
            va="center",
            fontsize=7,
            color="#555555",
        )

    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(filtered, method),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / f"feature_heatmap_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LogReg vs MLP across scenarios (baseline + symbolic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_model_comparison.py fox wheeler --filtered
  python run_model_comparison.py --all --filtered naive50
  python run_model_comparison.py --all --no-run --filtered
""",
    )
    parser.add_argument(
        "scenarios", nargs="*", help="Scenario names (e.g. fox wheeler)."
    )
    parser.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help=f"Run all scenarios: {', '.join(ALL_SCENARIOS)}.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if results exist."
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running; plot from the latest existing comparison run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the latest existing comparison run dir. Skips model/scenario pairs that already have results; only runs missing ones.",
    )
    parser.add_argument(
        "--filter-config", type=Path, default=None, help="Path to mining filter YAML."
    )
    parser.add_argument(
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help="Use filtered alerts. Optionally pass a balancing method (e.g. naive50, type_stratified).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-K features for overlap/heatmap plots (default: 25).",
    )
    args = parser.parse_args()

    if args.all_scenarios:
        args.scenarios = list(ALL_SCENARIOS)
    elif not args.scenarios:
        parser.error("Specify at least one scenario name or use --all.")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    COMPARISON_BASE.mkdir(parents=True, exist_ok=True)

    if args.no_run or args.resume:
        candidates = sorted(
            p
            for p in COMPARISON_BASE.iterdir()
            if p.is_dir() and p.name.startswith("comparison_")
        )
        if not candidates:
            print("[error] No existing comparison runs found under", COMPARISON_BASE)
            sys.exit(1)
        run_dir = candidates[-1]
        print(f"  Using existing run: {run_dir.name}")
    else:
        run_dir = COMPARISON_BASE / f"comparison_{run_ts}"

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = plots_dir / f"comparison_{run_ts}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}\n")

    try:
        _run_main(args, run_dir, plots_dir)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _run_main(args, run_dir: Path, plots_dir: Path) -> None:
    scenarios = args.scenarios
    filtered = args.filtered is not None
    method: str | None = args.filtered if args.filtered else None

    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )
    cache_subdir = f"filtered_{method}" if method else "filtered"

    # ── Phase 1: Run ──────────────────────────────────────────────────────────
    if not args.no_run:
        cache_dir = CACHE_DIR / cache_subdir if filtered else None
        for model in MODELS:
            model_dir = run_dir / model
            for scenario in scenarios:
                existing = _load_compare_json(model_dir, scenario)
                if existing is not None and not args.force:
                    print(f"[skip] {model}/{scenario} — exists. Use --force to re-run.")
                    continue
                alerts_path = (
                    _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
                    if filtered
                    else None
                )
                try:
                    _run_for_model(
                        scenario=scenario,
                        model_name=model,
                        model_dir=model_dir,
                        filter_config=args.filter_config,
                        alerts_json_path=alerts_path,
                        cache_dir=cache_dir,
                    )
                except Exception as exc:
                    print(f"\n[{model}/{scenario}] FAILED: {exc}")
                    traceback.print_exc()

    # ── Phase 2: Performance comparison ───────────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 2: PERFORMANCE COMPARISON\n{'='*60}")
    all_results = _load_all_results(run_dir, scenarios)
    df = _build_comparison_df(all_results)

    if df.empty:
        print("[error] No results loaded. Nothing to plot.")
        return

    text_table = _format_text_table(df)
    print("\n" + text_table)
    (plots_dir / "comparison_table.txt").write_text(text_table, encoding="utf-8")
    df.to_csv(plots_dir / "comparison_table.csv", index=False)
    print(f"\n  Saved → {plots_dir / 'comparison_table.txt'}")
    print(f"  Saved → {plots_dir / 'comparison_table.csv'}")

    print("\n[plots]")
    plot_perf_auc(df, plots_dir, filtered, method)
    plot_perf_metrics(df, plots_dir, filtered, method)

    # ── Phase 3: Per-model feature analysis ───────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 3: PER-MODEL FEATURE ANALYSIS\n{'='*60}")
    for model in MODELS:
        print(f"\n  [{model}]")
        try:
            run_per_model_feature_analysis(
                model_name=model,
                model_dir=run_dir / model,
                scenarios=scenarios,
                top_k=args.top_k,
                out_dir=plots_dir,
            )
        except Exception as exc:
            print(f"  [{model}] feature analysis failed: {exc}")
            traceback.print_exc()

    # ── Phase 4: Cross-model feature analysis ─────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 4: CROSS-MODEL FEATURE ANALYSIS\n{'='*60}")
    all_imps = _collect_importances(run_dir, scenarios, experiment="baseline")

    _print_overlap_table(all_imps, scenarios, args.top_k)

    cross_dir = plots_dir / "cross_model"
    cross_dir.mkdir(parents=True, exist_ok=True)

    for ma, mb in itertools.combinations(MODELS, 2):
        plot_cross_model_overlap(
            all_imps, scenarios, args.top_k, filtered, cross_dir, method, ma, mb
        )
    plot_shap_importance_bars(
        all_imps, scenarios, args.top_k, filtered, cross_dir, method
    )
    for model in MODELS:
        plot_cross_scenario_heatmap(
            all_imps.get(model, {}), model, args.top_k, filtered, cross_dir, method
        )

    print(f"\nAll output written to {run_dir}")


if __name__ == "__main__":
    main()
