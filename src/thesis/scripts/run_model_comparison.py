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
  python src/thesis/scripts/run_model_comparison.py fox --models logreg
  python src/thesis/scripts/run_model_comparison.py fox --models logreg mlp

Mining scope (--mine-frac / --no-overlap):
  Transactions are sorted chronologically before any split is applied.

  --mine-frac 1.0  (default)
    Mine: [0%, 100%)   Train: [0%, 70%)   Test: [70%, 100%)

  --mine-frac 0.7
    Mine: [0%, 70%)    Train: [0%, 70%)   Test: [70%, 100%)
    Mining and training cover the same window; no test-period data
    leaks into the mined patterns.

  --mine-frac 0.5
    Mine: [0%, 50%)    Train: [0%, 70%)   Test: [70%, 100%)
    Mining uses only the earliest 50%; training still uses [0%, 70%).

  --mine-frac 0.3 --no-overlap
    Mine: [0%, 30%)    Train: [30%, 70%)  Test: [70%, 100%)
    Mining and training are strictly disjoint.

  --mine-frac 0.5 --no-overlap
    Mine: [0%, 50%)    Train: [50%, 70%)  Test: [70%, 100%)

  --no-overlap without --mine-frac is a no-op (mine_frac=1.0).

Random split (--random-split / --random-seed):
  By default transactions are sorted chronologically before any split is
  applied (temporal holdout).  Pass --random-split to shuffle the full
  transaction list with a fixed random seed before mining, training, and
  testing are performed.  This distributes attack traffic more evenly across
  all splits, which is useful for checking whether the temporal ordering is
  the main driver of model performance.

  --random-split                  (seed defaults to 42)
    Shuffle: random   Mine: random 100%   Train: random 70%   Test: random 30%

  --random-split --random-seed 123
    Same, but with seed 123 for a different shuffle.

  --random-split --mine-frac 0.7
    Shuffle first, then mine on the first 70% of the shuffled list.
    Train/test are still 70/30 of the full shuffled list.

  --random-split runs are stored in a separate directory (tagged _rs<seed>)
  so they never collide with temporal-split runs of the same configuration.


Output (under artifacts/experiments/run_model_comparison/comparison_<ts>/):
  logreg/scenario/<scenario>/       logreg compare, baseline, symbolic JSONs
  mlp/scenario/<scenario>/          mlp compare, baseline, symbolic JSONs
  plots/
    comparison_table.txt / .csv
    logreg/                   per-model feature analysis plots
    mlp/
    logreg/
      shap_bars_logreg.png    signed SHAP importance across scenarios (symbolic experiment)
    mlp/
      shap_bars_mlp.png
    cross_model/
      perf_auc.png              AUC: logreg vs mlp, baseline + symbolic
      perf_metrics.png          F1/precision/recall/ba for baseline
      jaccard_spearman.png    per-scenario Jaccard + Spearman rank (logreg vs mlp)
      common_features_logreg.png  signed SHAP per scenario for features shared across scenarios
      common_features_mlp.png
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

from thesis.config import GroupingConfig
from thesis.experiments.baseline import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)
from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)
from thesis.experiments.anomaly import run_anomaly_experiment
from thesis.schemas.experiments import AnomalyExperimentConfig
from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR
from thesis.visualization.eda import SCENARIOS as ALL_SCENARIOS
from thesis.xai.overlap import jaccard as _jaccard, spearman_rank, top_k_names

import matplotlib

matplotlib.use("Agg")

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

COMPARISON_BASE = _REPO / "artifacts" / "experiments" / "run_model_comparison"
MODELS = ["logreg", "mlp", "lstm", "iforest", "ocsvm"]
ANOMALY_MODELS = {"iforest", "ocsvm"}
_MODEL_COLORS = {
    "logreg": "#4C72B0",
    "mlp": "#DD8452",
    "lstm": "#55A868",
    "iforest": "#9B59B6",
    "ocsvm": "#E74C3C",
}
_MODEL_LABELS = {
    "logreg": "LogReg",
    "mlp": "MLP",
    "lstm": "LSTM",
    "iforest": "IForest",
    "ocsvm": "OC-SVM",
}


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
    cache_dir: Path,
    grouping: GroupingConfig | None = None,
    mine_frac: float = 1.0,
    no_overlap: bool = False,
    random_split: bool = False,
    random_seed: int = 42,
    schema_cache: dict | None = None,
) -> dict:
    """Run baseline + symbolic for one scenario/model, write compare JSON.

    schema_cache maps scenario -> Path of the already-mined symbolic schema.
    When populated, the mining step is skipped and the cached schema is reused.
    After a successful symbolic run the cache is updated so subsequent models
    in the same script invocation can skip mining too.
    """
    print(f"\n{'='*60}\n  {model_name.upper()} — {scenario}\n{'='*60}")

    scenario_dir = model_dir / "scenario" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {"cache_dir": cache_dir}
    if grouping is not None:
        extra["grouping"] = grouping

    if schema_cache is None:
        schema_cache = {}
    prebuilt = schema_cache.get(scenario)

    def _nan(v):
        return None if v != v else v

    if model_name in ANOMALY_MODELS:
        print("\n--- anomaly (base features) ---")
        baseline = run_anomaly_experiment(
            AnomalyExperimentConfig(
                scenario=scenario,
                model_name=model_name,
                schema_name="base",
                results_dir=scenario_dir,
                alerts_json_path=alerts_json_path,
                mine_frac=mine_frac,
                filter_config=filter_config,
                abstraction_map_path=ABSTRACTION_MAP_PATH,
                **extra,
            )
        )

        print("\n--- anomaly (base + symbolic features) ---")
        symbolic = run_anomaly_experiment(
            AnomalyExperimentConfig(
                scenario=scenario,
                model_name=model_name,
                schema_name="base+symbolic",
                results_dir=scenario_dir,
                alerts_json_path=alerts_json_path,
                mine_frac=mine_frac,
                filter_config=filter_config,
                abstraction_map_path=ABSTRACTION_MAP_PATH,
                prebuilt_symbolic_schema_path=prebuilt,
                **extra,
            )
        )
    else:
        print("\n--- baseline ---")
        baseline = run_baseline_experiment(
            BaselineExperimentConfig(
                scenario=scenario,
                model_name=model_name,
                results_dir=scenario_dir,
                alerts_json_path=alerts_json_path,
                random_split=random_split,
                random_seed=random_seed,
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
                mine_frac=mine_frac,
                no_overlap=no_overlap,
                random_split=random_split,
                random_seed=random_seed,
                prebuilt_symbolic_schema_path=prebuilt,
                **extra,
            )
        )

    # Cache the schema path so subsequent models in this run skip mining
    if symbolic.symbolic_schema_path is not None:
        schema_cache[scenario] = symbolic.symbolic_schema_path

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

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
    run_dir: Path,
    scenarios: list[str],
    experiment: str = "baseline",
    models: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns {model: {scenario: {feature: importance}}}.
    Prefers by_shap; falls back to by_permutation if empty.
    """
    if models is None:
        models = list(MODELS)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
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
            if not imp:
                imp = _load_importance(rf, "by_coefficient")
                if imp:
                    print(
                        f"  [info] {model}/{scenario}: SHAP+permutation empty, using coefficient/MDI"
                    )
            result[model][scenario] = imp
    return result


def _load_all_results(
    run_dir: Path, scenarios: list[str], models: list[str] | None = None
) -> dict[str, list[dict]]:
    if models is None:
        models = list(MODELS)
    all_results: dict[str, list[dict]] = {m: [] for m in models}
    for model in models:
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


def _format_text_table(df: pd.DataFrame, models: list[str] | None = None) -> str:
    if models is None:
        models = list(MODELS)
    base = df[df["experiment"] == "baseline"]
    sym = df[df["experiment"] == "symbolic"]
    scenarios = sorted(df["scenario"].unique())

    w = 9
    sep = "─" * (16 + (w * 3 + 4) * len(models) * 2)
    parts = [sep]
    header = f"  {'scenario':<14}"
    for exp_label, src in [("base", base), ("sym", sym)]:
        for m in models:
            header += (
                f"  {('[' + _MODEL_LABELS[m] + '|' + exp_label + ']'):<{w * 3 + 2}}"
            )
    parts.append(header)

    subheader = f"  {'':14}"
    for _ in range(2):
        for _ in models:
            subheader += f"  {'AUC':>{w}}{'F1':>{w}}{'FP':>{w}}"
    parts.append(subheader)
    parts.append(sep)

    for scenario in scenarios:
        row = f"  {scenario:<14}"
        for src in (base, sym):
            for m in models:
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
    df: pd.DataFrame,
    out_dir: Path,
    filtered: bool,
    method: str | None = None,
    models: list[str] | None = None,
) -> None:
    if models is None:
        models = list(MODELS)
    scenarios = sorted(df["scenario"].unique())
    n_sc = len(scenarios)
    n_bars = len(models) * 2
    x = np.arange(n_sc)
    w = 0.18
    offsets = [w * (i - (n_bars - 1) / 2) for i in range(n_bars)]

    fig, ax = plt.subplots(figsize=(max(8, n_sc * 2.2), 5))
    bar_idx = 0
    for model in models:
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
    df: pd.DataFrame,
    out_dir: Path,
    filtered: bool,
    method: str | None = None,
    models: list[str] | None = None,
) -> None:
    """Precision / recall / F1 / balanced_accuracy for baseline, models side by side."""
    if models is None:
        models = list(MODELS)
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
        for i, model in enumerate(models):
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

    # Recover symbolic results files whose stored absolute path is stale (e.g. after
    # a directory rename or a run resumed into a different folder).  We look for the
    # file by name in the scenario's local directory before giving up.
    for s in present:
        if s not in sym_data:
            stored = compare[s].get("symbolic", {}).get("results_file")
            if stored:
                local = model_dir / "scenario" / s / Path(stored).name
                if local.exists():
                    with local.open() as _f:
                        sym_data[s] = json.load(_f)
                    print(
                        f"  [fix] Recovered symbolic results for '{s}' from local path"
                    )

    filtered = any(compare[s].get("filtered", False) for s in present)
    funnels = {s: sfa.feature_funnel(sym_data[s]) for s in present if s in sym_data}
    # Only pass scenarios that have funnel data to functions that do funnels[s] lookups.
    funnel_present = [s for s in present if s in funnels]

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    sfa.print_funnel_table(funnel_present, funnels)
    sfa.print_overlap_table(present, sym_data, top_k)

    if funnels:
        sfa.plot_funnel(funnel_present, funnels, filtered, model_out)
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
    pairs = list(itertools.combinations(all_imps.keys(), 2))
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

    for model in all_imps:
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

    for model in all_imps:
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
            raw_values = [r[1] for r in top]
            max_abs = max(abs(v) for v in raw_values) if raw_values else 1.0
            norm_values = [v / max_abs for v in raw_values]
            colors = ["#4C72B0" if v > 0 else "#C94040" for v in raw_values]
            y = np.arange(len(names))
            ax.barh(y, norm_values, color=colors, alpha=0.85, height=0.7)
            for yi, (norm, raw) in enumerate(zip(norm_values, raw_values)):
                if abs(norm) < 0.01:
                    continue
                x_tip = norm + (0.04 if norm >= 0 else -0.04)
                ha = "left" if norm >= 0 else "right"
                ax.text(
                    x_tip,
                    yi,
                    f"{raw:.3f}",
                    ha=ha,
                    va="center",
                    fontsize=7,
                    color="black",
                    clip_on=False,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlim(-1.3, 1.3)
            ax.set_title(sc, fontsize=9)
            ax.set_xlabel("Normalised SHAP (labels = raw value)", fontsize=7)
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
            f"Top-{top_k} features by |SHAP| per scenario — {_MODEL_LABELS[model]}\n"
            f"Bars normalised per scenario; annotated with raw SHAP value",
            fontsize=10,
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


def plot_common_feature_importance(
    imps_by_scenario: dict[str, dict[str, float]],
    model_name: str,
    top_k: int,
    filtered: bool,
    out_dir: Path,
    method: str | None = None,
    min_freq: int = 2,
) -> None:
    """
    Grouped bar chart for features shared across scenarios.

    Bars are normalised per scenario (each bar = SHAP / max |SHAP| for that
    scenario) so that scenarios with very different scales are visually comparable.
    The raw SHAP value is annotated on every bar whose normalised height exceeds 5%
    of the scenario maximum.

    Only features in the top-K of at least `min_freq` scenarios are shown.
    Mined (symbolic) features are marked with '*' in their label.
    """
    scenarios = [s for s, imp in imps_by_scenario.items() if imp]
    if not scenarios:
        return

    top_per_sc = {s: top_k_names(imps_by_scenario[s], top_k) for s in scenarios}

    # Keep only features that appear in at least min_freq scenarios' top-K
    all_features = set().union(*top_per_sc.values())
    common = [
        f
        for f in all_features
        if sum(1 for tops in top_per_sc.values() if f in tops) >= min_freq
    ]
    if not common:
        print(
            f"  [warn] {model_name}: no features appear in ≥{min_freq} scenarios' top-{top_k}; skipping common-feature plot"
        )
        return

    # Sort by mean |SHAP| across scenarios, descending
    common.sort(
        key=lambda f: sum(abs(imps_by_scenario[s].get(f, 0.0)) for s in scenarios),
        reverse=True,
    )

    # Per-scenario normalisation denominator: max |SHAP| across all features in that scenario
    max_abs = {
        sc: max((abs(v) for v in imps_by_scenario[sc].values()), default=1.0) or 1.0
        for sc in scenarios
    }

    sc_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_sc = len(scenarios)
    n_feat = len(common)
    bar_width = 0.8 / n_sc
    x = np.arange(n_feat)

    fig_w = max(8, n_feat * (0.5 + 0.15 * n_sc))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    for i, sc in enumerate(scenarios):
        offsets = x + (i - n_sc / 2 + 0.5) * bar_width
        raw_vals = [imps_by_scenario[sc].get(f, 0.0) for f in common]
        norm_vals = [v / max_abs[sc] for v in raw_vals]
        bars = ax.bar(
            offsets,
            norm_vals,
            width=bar_width * 0.9,
            label=sc,
            color=sc_colors[i % len(sc_colors)],
            alpha=0.85,
        )

        for rect, raw, norm in zip(bars, raw_vals, norm_vals):
            if abs(norm) < 0.05:
                continue
            x_center = rect.get_x() + rect.get_width() / 2
            y_tip = norm + (0.02 if norm >= 0 else -0.02)
            va = "bottom" if norm >= 0 else "top"
            ax.text(
                x_center,
                y_tip,
                f"{raw:.3f}",
                ha="center",
                va=va,
                fontsize=5,
                rotation=90,
                color="black",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    display_names = [
        ("* " + f[5:][:40] if f.startswith("sym__") else f[:40]) for f in common
    ]
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(
        "Normalised SHAP value (per-scenario, relative to scenario max |SHAP|)\n"
        "positive → predicts attack, negative → predicts benign; labels = raw SHAP",
        fontsize=8,
    )
    ax.set_xlabel(
        f"Feature (top-{top_k} in ≥{min_freq}/{len(scenarios)} scenarios, sorted by mean |SHAP|)",
        fontsize=8,
    )
    ax.set_ylim(-1.25, 1.25)

    n_sym = sum(1 for f in common if f.startswith("sym__"))
    sym_note = f" — {n_sym} mined feature(s) marked *" if n_sym else ""
    ax.set_title(
        f"Shared feature importance across scenarios — {_MODEL_LABELS[model_name]}{sym_note}\n"
        f"Bars normalised per scenario so scale differences don't hide pattern; annotated with raw SHAP.",
        fontsize=10,
    )
    ax.legend(title="Scenario", fontsize=8, title_fontsize=8, loc="best")
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
    out = out_dir / f"common_features_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_distinctive_feature_importance(
    imps_by_scenario: dict[str, dict[str, float]],
    model_name: str,
    top_k: int,
    filtered: bool,
    out_dir: Path,
    method: str | None = None,
    max_freq: int = 1,
    top_n_per_scenario: int = 5,
) -> None:
    """
    Horizontal grouped bar chart for features that are distinctive to specific scenarios.

    Shows features in the top-K of at most `max_freq` scenarios, grouped by their
    primary scenario (where they rank highest). Within each group features are sorted
    by |SHAP| in their primary scenario. Bars for other scenarios are shown for
    contrast — they will be near-zero for truly distinctive features.

    This is the complement of plot_common_feature_importance.
    Mined (symbolic) features are marked with '*' in their label.
    """
    scenarios = [s for s, imp in imps_by_scenario.items() if imp]
    if not scenarios:
        return

    top_per_sc = {s: top_k_names(imps_by_scenario[s], top_k) for s in scenarios}

    all_features = set().union(*top_per_sc.values())
    distinctive = {
        f
        for f in all_features
        if sum(1 for tops in top_per_sc.values() if f in tops) <= max_freq
    }
    if not distinctive:
        print(
            f"  [warn] {model_name}: no features appear in ≤{max_freq} scenarios' "
            f"top-{top_k}; skipping distinctive-feature plot"
        )
        return

    # Group each feature under the scenario where it has highest |SHAP|
    sc_groups: dict[str, list[tuple[str, float]]] = {s: [] for s in scenarios}
    for f in distinctive:
        primary = max(scenarios, key=lambda s: abs(imps_by_scenario[s].get(f, 0.0)))
        sc_groups[primary].append((f, imps_by_scenario[primary].get(f, 0.0)))

    # Per scenario: keep top N by |SHAP|, sort ascending so most important is at top
    ordered_features: list[tuple[str, str]] = []  # (feature_name, primary_scenario)
    for s in scenarios:
        top_feats = sorted(sc_groups[s], key=lambda x: abs(x[1]), reverse=True)[
            :top_n_per_scenario
        ]
        for f, _ in reversed(
            top_feats
        ):  # reversed → most important at top in horizontal bar
            ordered_features.append((f, s))

    if not ordered_features:
        return

    max_abs = {
        sc: max((abs(v) for v in imps_by_scenario[sc].values()), default=1.0) or 1.0
        for sc in scenarios
    }

    sc_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sc_color_map = {s: sc_colors[i % len(sc_colors)] for i, s in enumerate(scenarios)}

    n_feat = len(ordered_features)
    n_sc = len(scenarios)
    bar_h = 0.8 / n_sc
    y = np.arange(n_feat)

    fig_h = max(6, n_feat * (0.3 + 0.08 * n_sc))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    for i, sc in enumerate(scenarios):
        offsets = y + (i - n_sc / 2 + 0.5) * bar_h
        raw_vals = [imps_by_scenario[sc].get(f, 0.0) for f, _ in ordered_features]
        norm_vals = [v / max_abs[sc] for v in raw_vals]
        bars = ax.barh(
            offsets,
            norm_vals,
            height=bar_h * 0.9,
            label=sc,
            color=sc_color_map[sc],
            alpha=0.85,
        )

        for rect, raw, norm in zip(bars, raw_vals, norm_vals):
            if abs(norm) < 0.05:
                continue
            x_tip = norm + (0.03 if norm >= 0 else -0.03)
            ha = "left" if norm >= 0 else "right"
            ax.text(
                x_tip,
                rect.get_y() + rect.get_height() / 2,
                f"{raw:.3f}",
                ha=ha,
                va="center",
                fontsize=5,
                color="black",
            )

    # Dashed separators between scenario groups
    group_sizes = [sum(1 for _, ps in ordered_features if ps == s) for s in scenarios]
    boundary = 0
    for size in group_sizes[:-1]:
        boundary += size
        ax.axhline(
            boundary - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5
        )

    # Scenario labels on the right margin
    boundary = 0
    for s, size in zip(scenarios, group_sizes):
        if size:
            mid = boundary + size / 2 - 0.5
            ax.text(
                1.32,
                mid,
                s,
                va="center",
                ha="left",
                fontsize=7,
                color=sc_color_map[s],
                fontweight="bold",
                transform=ax.get_yaxis_transform(),
            )
        boundary += size

    ax.set_yticks(y)
    display_names = [
        ("* " + f[5:][:50] if f.startswith("sym__") else f[:50])
        for f, _ in ordered_features
    ]
    ax.set_yticklabels(display_names, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel(
        "Normalised SHAP value (per-scenario, relative to scenario max |SHAP|)\n"
        "positive → predicts attack, negative → predicts benign; labels = raw SHAP",
        fontsize=8,
    )

    n_sym = sum(1 for f, _ in ordered_features if f.startswith("sym__"))
    sym_note = f" — {n_sym} mined feature(s) marked *" if n_sym else ""
    ax.set_title(
        f"Scenario-distinctive features — {_MODEL_LABELS[model_name]}{sym_note}\n"
        f"Top-{top_n_per_scenario} features in top-{top_k} of ≤{max_freq} scenario(s), "
        f"grouped and labelled by primary scenario",
        fontsize=10,
    )
    ax.legend(title="Scenario", fontsize=8, title_fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

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
    out = out_dir / f"distinctive_features_{model_name}.png"
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
        "--force",
        action="store_true",
        help="Start a new run directory and re-run everything, ignoring any existing results.",
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
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        metavar="MODEL",
        help=f"Models to run (default: all). Choices: {', '.join(MODELS)}. Anomaly models: {', '.join(sorted(ANOMALY_MODELS))}.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        metavar="W",
        help="Fixed-window size in seconds for alert grouping (default: 2).",
    )
    parser.add_argument(
        "--mine-frac",
        type=float,
        default=1.0,
        dest="mine_frac",
        help="Fraction of transactions (sorted by time) to use for mining (default: 1.0 = all).",
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
        help="Shuffle transactions randomly before any split (mining, train, test) instead of using temporal order.",
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

    if args.models is None:
        args.models = list(MODELS)

    if args.all_scenarios:
        args.scenarios = list(ALL_SCENARIOS)
    elif not args.scenarios:
        parser.error("Specify at least one scenario name or use --all.")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    COMPARISON_BASE.mkdir(parents=True, exist_ok=True)

    # Encode filter config in the directory name so filtered/unfiltered runs never share a dir
    _filter = args.filtered
    if _filter is None:
        filter_tag = "raw"
    elif _filter:
        filter_tag = f"filtered_{_filter}"
    else:
        filter_tag = "filtered"

    window_tag = f"_w{args.window_size}" if args.window_size != 2 else ""
    mine_tag = ""
    if args.mine_frac != 1.0:
        mine_tag = f"_mf{args.mine_frac}".replace(".", "p")
    if args.no_overlap:
        mine_tag += "_nool"
    if args.random_split:
        mine_tag += f"_rs{args.random_seed}"
    dir_prefix = f"comparison_{filter_tag}{window_tag}{mine_tag}_"
    existing_runs = sorted(
        p
        for p in COMPARISON_BASE.iterdir()
        if p.is_dir()
        and p.name.startswith(dir_prefix)
        # Exclude subdirectories that have extra tags beyond the expected prefix.
        # e.g. prefix "comparison_raw_mf0p7_" must not match "comparison_raw_mf0p7_rs42_..."
        and p.name[len(dir_prefix) : len(dir_prefix) + 2].isdigit()
    )

    if args.no_run or args.resume:
        if not existing_runs:
            print(
                f"[error] No existing '{filter_tag}' comparison runs found under",
                COMPARISON_BASE,
            )
            sys.exit(1)
        run_dir = existing_runs[-1]
        print(f"  Using existing run: {run_dir.name}")
    elif existing_runs and not args.force:
        run_dir = existing_runs[-1]
        print(
            f"  Resuming latest run: {run_dir.name}  (use --force to start a new run)"
        )
    else:
        run_dir = COMPARISON_BASE / f"{dir_prefix}{run_ts}"

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
    models: list[str] = args.models
    filtered = args.filtered is not None
    method: str | None = args.filtered if args.filtered else None
    window_size: int = args.window_size

    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )

    grouping = GroupingConfig(window_size=window_size)
    window_tag = f"_w{window_size}" if window_size != 2 else ""

    if filtered:
        method_tag = (
            f"filtered_{method}{window_tag}" if method else f"filtered{window_tag}"
        )
    elif window_tag:
        method_tag = f"w{window_size}"
    else:
        method_tag = "fixed_window"

    # ── Phase 1: Run ──────────────────────────────────────────────────────────
    if not args.no_run:
        # Shared cache of scenario → symbolic schema path, populated on first mine
        # and reused by subsequent models to skip redundant mining.
        schema_cache: dict[str, Path] = {}
        for model in models:
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
                scenario_cache_dir = CACHE_DIR / scenario / "groups" / method_tag
                scenario_cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _run_for_model(
                        scenario=scenario,
                        model_name=model,
                        model_dir=model_dir,
                        filter_config=args.filter_config,
                        alerts_json_path=alerts_path,
                        cache_dir=scenario_cache_dir,
                        grouping=grouping,
                        mine_frac=args.mine_frac,
                        no_overlap=args.no_overlap,
                        random_split=args.random_split,
                        random_seed=args.random_seed,
                        schema_cache=schema_cache,
                    )
                except Exception as exc:
                    print(f"\n[{model}/{scenario}] FAILED: {exc}")
                    traceback.print_exc()

    # ── Phase 2: Performance comparison ───────────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 2: PERFORMANCE COMPARISON\n{'='*60}")
    all_results = _load_all_results(run_dir, scenarios, models=models)
    df = _build_comparison_df(all_results)

    if df.empty:
        print("[error] No results loaded. Nothing to plot.")
        return

    text_table = _format_text_table(df, models=models)
    print("\n" + text_table)
    (plots_dir / "comparison_table.txt").write_text(text_table, encoding="utf-8")
    df.to_csv(plots_dir / "comparison_table.csv", index=False)
    print(f"\n  Saved → {plots_dir / 'comparison_table.txt'}")
    print(f"  Saved → {plots_dir / 'comparison_table.csv'}")

    cross_dir = plots_dir / "cross_model"
    cross_dir.mkdir(parents=True, exist_ok=True)

    print("\n[plots]")
    plot_perf_auc(df, cross_dir, filtered, method, models=models)
    plot_perf_metrics(df, cross_dir, filtered, method, models=models)

    # ── Phase 3: Per-model feature analysis ───────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 3: PER-MODEL FEATURE ANALYSIS\n{'='*60}")
    for model in models:
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
    all_imps = _collect_importances(
        run_dir, scenarios, experiment="baseline", models=models
    )
    all_imps_sym = _collect_importances(
        run_dir, scenarios, experiment="symbolic", models=models
    )

    # Prefer symbolic importances (base + mined features) for all cross-model analysis; fall back to baseline
    all_imps_combined: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        all_imps_combined[model] = {}
        for sc in scenarios:
            sym = all_imps_sym.get(model, {}).get(sc, {})
            all_imps_combined[model][sc] = (
                sym if sym else all_imps.get(model, {}).get(sc, {})
            )

    _print_overlap_table(all_imps_combined, scenarios, args.top_k)

    for ma, mb in itertools.combinations(models, 2):
        plot_cross_model_overlap(
            all_imps_combined,
            scenarios,
            args.top_k,
            filtered,
            cross_dir,
            method,
            ma,
            mb,
        )
    for model in models:
        model_plot_dir = plots_dir / model
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_shap_importance_bars(
            {model: all_imps_combined.get(model, {})},
            scenarios,
            args.top_k,
            filtered,
            model_plot_dir,
            method,
        )
        plot_common_feature_importance(
            all_imps_combined.get(model, {}),
            model,
            args.top_k,
            filtered,
            cross_dir,
            method,
        )
        plot_distinctive_feature_importance(
            all_imps_combined.get(model, {}),
            model,
            args.top_k,
            filtered,
            cross_dir,
            method,
            max_freq=1,
        )

    print(f"\nAll output written to {run_dir}")


if __name__ == "__main__":
    main()
