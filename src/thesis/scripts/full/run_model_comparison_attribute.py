"""
Compare models (logreg vs MLP vs LSTM) across scenarios using ATTRIBUTE
mining (per-alert-group contrast-set + decision-tree rule mining) instead of
the old Eclat/PrefixSpan cross-signature co-occurrence mining.

This is the attribute-mining counterpart to run_model_comparison.py, which
covers the eclat/prefixspan cross-signature path. Phase 1 (running the
baseline + symbolic experiments) is reimplemented here with
mining_strategy="attribute"; Phases 2-4 (performance comparison, per-model
feature analysis, cross-model feature analysis) are identical between the
two mining strategies and are reused directly from run_model_comparison.py
so the plotting/analysis code isn't duplicated.

Anomaly models (bernoulli_oc, autoencoder_oc, ocsvm) aren't wired for
attribute mining yet (run_anomaly_experiment always mines cooccurrence
schemas) and are not offered here.

Usage:
  python src/thesis/scripts/run_model_comparison_attribute.py fox wheeler harrison
  python src/thesis/scripts/run_model_comparison_attribute.py --all --filtered naive50
  python src/thesis/scripts/run_model_comparison_attribute.py cscas --models logreg mlp
  python src/thesis/scripts/run_model_comparison_attribute.py fox \
    --min-growth-rate 4.0 --max-depth 5 --min-samples-leaf 10
  python src/thesis/scripts/full/run_model_comparison_attribute.py cscas \
    --train-frac 0.1 --test-frac 0.9 --mine-frac 0.1  # reproduce CSCAS paper split: 6 of 60 days train, rest test, mine on first 6 days

Mining scope (--mine-frac / --no-overlap) and random split
(--random-split / --random-seed) behave exactly as in run_model_comparison.py.

Attribute mining thresholds (Step 1 contrast-set filter / Step 2 decision
tree) replace --filter-config, which does not apply to this mining strategy:
  --min-attack-coverage / --min-benign-coverage / --min-growth-rate / --max-p-value
  --max-depth / --min-samples-leaf / --class-weight

Train/test split (--test-frac / --train-frac):
  Alert_groups are sorted chronologically (or shuffled if --random-split) before
  any split is applied. --test-frac holds out that fraction from the end for
  testing, exactly as in run_model_comparison.py (default 0.3). --train-frac is
  new here: it caps how much of the timeline immediately before the test split
  is actually used for training, in the same units as --test-frac.

  --test-frac 0.3                          (default)
    Train: [0%, 70%)     Test: [70%, 100%)   -- train on everything before test

  --train-frac 0.1 --test-frac 0.9
    Train: [0%, 10%)     Test: [90%, 100%)   -- CSCAS paper's 6-of-60-day split

  --train-frac 0.1 --test-frac 0.7
    Train: [60%, 70%)    Test: [70%, 100%)   -- fixed-size train window, with an
                                                 unused gap [0%, 60%) before it

  --train-frac composes with --mine-frac/--no-overlap's own exclusion of the
  mining window from training (whichever pushes the train start later wins);
  leaving --train-frac unset preserves the old behavior exactly (train on
  everything from the mine-window boundary, or 0, up to the test split).

Output (under artifacts/experiments/run_model_comparison_attribute/comparison_<ts>/):
  Same layout as run_model_comparison.py's output.
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

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.experiments.baseline import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)
from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.paths import CACHE_DIR
from thesis.schemas.mining import AttributeMiningConfig
from thesis.visualization.eda import SCENARIOS as ALL_SCENARIOS

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

COMPARISON_BASE = _REPO / "artifacts" / "experiments" / "run_model_comparison_attribute"
MODELS = ["logreg", "rf", "mlp", "xgboost", "torch_nn", "rf_gpu"]

# Training-pool conditions (see training/pool_sampling.py). "guided" is
# CSCAS-only -- AIT-ADS has no SCAS-equivalent outlier signal (see
# baselines/_sampling.py's module docstring; generalizing it is an explicit
# non-goal). Iterated as the outer loop in main() below: each condition gets
# its own run_dir/<condition>/ subtree, so a scenario mix spanning both
# CSCAS and AIT-ADS just skips "guided" for the AIT-ADS scenarios within
# that pass rather than needing a separate invocation.
ALL_POOL_CONDITIONS = ["random", "class_weighted", "guided"]


def _load_rmc():
    """Dynamically load run_model_comparison.py (scripts/ has no __init__.py,
    so it can't be imported as a normal package module) to reuse its Phase
    2-4 plotting/analysis functions unchanged."""
    spec = importlib.util.spec_from_file_location(
        "run_model_comparison", _HERE.parent / "run_model_comparison.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rmc = _load_rmc()


# ---------------------------------------------------------------------------
# Phase 1: Running experiments (attribute mining)
# ---------------------------------------------------------------------------


def _run_for_model(
    scenario: str,
    model_name: str,
    model_dir: Path,
    attribute_mining_config: AttributeMiningConfig,
    alerts_json_path: Path | None,
    cache_dir: Path,
    grouping: GroupingConfig | None = None,
    mine_frac: float = 1.0,
    no_overlap: bool = False,
    random_split: bool = False,
    random_seed: int = 42,
    test_frac: float = 0.3,
    train_frac: float | None = None,
    schema_cache: dict | None = None,
    pool_condition: str = "none",
) -> dict:
    """Run baseline + attribute-mined symbolic for one scenario/model, write
    compare JSON. Mirrors run_model_comparison._run_for_model but always
    trains the binary classifiers (no anomaly branch) with
    mining_strategy="attribute". schema_cache behaves as in the co-occurrence
    version: populated after the first successful mine so later models in the
    same run reuse it instead of re-mining.

    pool_condition: training-pool construction strategy (see
    training/pool_sampling.py), applied identically to both the baseline and
    symbolic passes below."""
    print(
        f"\n{'='*60}\n  {model_name.upper()} — {scenario} (attribute mining)\n{'='*60}"
    )

    scenario_dir = rmc._dataset_scenario_path(model_dir / "scenario", scenario)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {"cache_dir": cache_dir}
    if grouping is not None:
        extra["grouping"] = grouping

    if schema_cache is None:
        schema_cache = {}
    prebuilt = schema_cache.get(scenario)

    def _nan(v):
        return None if v != v else v

    print("\n--- baseline ---")
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            model_name=model_name,
            results_dir=scenario_dir,
            alerts_json_path=alerts_json_path,
            random_split=random_split,
            random_seed=random_seed,
            test_frac=test_frac,
            train_frac=train_frac,
            pool_condition=pool_condition,
            **extra,
        )
    )

    print("\n--- symbolic (attribute mining) ---")
    symbolic = run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            mining_strategy="attribute",
            attribute_mining_config=attribute_mining_config,
            model_name=model_name,
            results_dir=scenario_dir,
            alerts_json_path=alerts_json_path,
            mine_frac=mine_frac,
            no_overlap=no_overlap,
            random_split=random_split,
            random_seed=random_seed,
            test_frac=test_frac,
            train_frac=train_frac,
            prebuilt_symbolic_schema_path=prebuilt,
            pool_condition=pool_condition,
            **extra,
        )
    )

    if symbolic.symbolic_schema_path is not None:
        schema_cache[scenario] = symbolic.symbolic_schema_path

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    combined = {
        "experiment": "compare",
        "scenario": scenario,
        "timestamp": ts,
        "model_name": model_name,
        "mining_strategy": "attribute",
        "filtered": alerts_json_path is not None,
        "baseline": {
            "schema_name": baseline.schema_name,
            "schema_version": baseline.schema_version,
            "auc": _nan(baseline.auc),
            "n_features": baseline.n_features,
            "n_alert_groups": baseline.n_alert_groups,
            "metrics": baseline.metrics,
            "results_file": str(baseline.results_file),
        },
        "symbolic": {
            "schema_name": symbolic.schema_name,
            "schema_version": symbolic.schema_version,
            "auc": _nan(symbolic.auc),
            "n_features": symbolic.n_features,
            "n_alert_groups": symbolic.n_alert_groups,
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
            "n_alert_groups": baseline.n_alert_groups,
        },
        "symbolic": {
            **symbolic.metrics,
            "n_features": symbolic.n_features,
            "n_alert_groups": symbolic.n_alert_groups,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LogReg vs MLP vs LSTM across scenarios (baseline + attribute-mined symbolic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_model_comparison_attribute.py fox wheeler --filtered
  python run_model_comparison_attribute.py --all --filtered naive50
  python run_model_comparison_attribute.py --all --no-run --filtered
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
        help=f"Models to run (default: all). Choices: {', '.join(MODELS)}.",
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
        help="Shuffle alert_groups randomly before any split (mining, train, test) instead of using temporal order.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        dest="random_seed",
        metavar="SEED",
        help="Random seed for --random-split (default: 42).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        dest="test_frac",
        metavar="FRAC",
        help=(
            "Fraction of alert_groups (chronologically last, or last after "
            "--random-split) held out as the test set. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        dest="train_frac",
        metavar="FRAC",
        help=(
            "Fraction of the full alert_group timeline used for training, "
            "immediately preceding the test split -- same units as --test-frac, "
            "so e.g. --train-frac 0.1 --test-frac 0.9 reproduces a published "
            "'first N / rest' split (CSCAS's paper: 6 of 60 days train, "
            "remainder test). Default: unset, meaning train on everything "
            "before the test split (i.e. train_frac = 1 - test_frac)."
        ),
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
    args = parser.parse_args()

    if args.models is None:
        args.models = list(MODELS)

    if args.all_scenarios:
        args.scenarios = list(ALL_SCENARIOS)
    elif not args.scenarios:
        parser.error("Specify at least one scenario name or use --all.")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    COMPARISON_BASE.mkdir(parents=True, exist_ok=True)

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
    if args.test_frac != 0.3:
        mine_tag += f"_tf{args.test_frac}".replace(".", "p")
    if args.train_frac is not None:
        mine_tag += f"_trf{args.train_frac}".replace(".", "p")
    dir_prefix = f"comparison_{filter_tag}{window_tag}{mine_tag}_"
    existing_runs = sorted(
        p
        for p in COMPARISON_BASE.iterdir()
        if p.is_dir()
        and p.name.startswith(dir_prefix)
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

    # This run_dir's own "plots" holds only the shared log across every pool
    # condition below -- each condition gets its own run_dir/<condition>/plots
    # subtree for its actual comparison tables/plots (see ALL_POOL_CONDITIONS).
    log_dir = run_dir / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"comparison_{run_ts}.log"
    tee = rmc._Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}\n")

    try:
        for condition in ALL_POOL_CONDITIONS:
            print(f"\n{'#' * 70}\n  POOL CONDITION: {condition}\n{'#' * 70}")
            condition_run_dir = run_dir / condition
            condition_plots_dir = condition_run_dir / "plots"
            condition_plots_dir.mkdir(parents=True, exist_ok=True)
            _run_main(args, condition_run_dir, condition_plots_dir, condition)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _run_main(args, run_dir: Path, plots_dir: Path, condition: str) -> None:
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

    attribute_mining_config = AttributeMiningConfig()
    attribute_mining_config.contrast.min_attack_coverage = args.min_attack_coverage
    attribute_mining_config.contrast.min_benign_coverage = args.min_benign_coverage
    attribute_mining_config.contrast.min_growth_rate = args.min_growth_rate
    attribute_mining_config.contrast.max_p_value = args.max_p_value
    attribute_mining_config.tree.max_depth = args.max_depth
    attribute_mining_config.tree.min_samples_leaf = args.min_samples_leaf
    attribute_mining_config.tree.class_weight = args.class_weight

    # ── Phase 1: Run ──────────────────────────────────────────────────────────
    if not args.no_run:
        schema_cache: dict[str, Path] = {}
        for model in models:
            model_dir = run_dir / model
            for scenario in scenarios:
                existing = rmc._load_compare_json(model_dir, scenario)
                if existing is not None and not args.force:
                    print(f"[skip] {model}/{scenario} — exists. Use --force to re-run.")
                    continue
                is_cscas = dataset_for_scenario(scenario) == "cscas"
                if condition == "guided" and not is_cscas:
                    print(
                        f"[skip] {model}/{scenario} — 'guided' pool condition is "
                        f"CSCAS-only, {scenario} has no SCAS-equivalent signal."
                    )
                    continue
                alerts_path = (
                    _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
                    if filtered and not is_cscas
                    else None
                )
                scenario_method_tag = (
                    CSCAS_PREGROUPED_METHOD if is_cscas else method_tag
                )
                scenario_grouping = (
                    GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
                    if is_cscas
                    else grouping
                )
                scenario_cache_dir = (
                    CACHE_DIR / scenario / "groups" / scenario_method_tag
                )
                scenario_cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _run_for_model(
                        scenario=scenario,
                        model_name=model,
                        model_dir=model_dir,
                        attribute_mining_config=attribute_mining_config,
                        alerts_json_path=alerts_path,
                        cache_dir=scenario_cache_dir,
                        grouping=scenario_grouping,
                        mine_frac=args.mine_frac,
                        no_overlap=args.no_overlap,
                        random_split=args.random_split,
                        random_seed=args.random_seed,
                        test_frac=args.test_frac,
                        train_frac=args.train_frac,
                        schema_cache=schema_cache,
                        pool_condition=condition,
                    )
                except Exception as exc:
                    print(f"\n[{model}/{scenario}] FAILED: {exc}")
                    traceback.print_exc()

    # ── Phase 2: Performance comparison ───────────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 2: PERFORMANCE COMPARISON\n{'='*60}")
    all_results = rmc._load_all_results(run_dir, scenarios, models=models)
    df = rmc._build_comparison_df(all_results)

    if df.empty:
        print("[error] No results loaded. Nothing to plot.")
        return

    text_table = rmc._format_text_table(df, models=models)
    print("\n" + text_table)
    (plots_dir / "comparison_table.txt").write_text(text_table, encoding="utf-8")
    df.to_csv(plots_dir / "comparison_table.csv", index=False)
    print(f"\n  Saved → {plots_dir / 'comparison_table.txt'}")
    print(f"  Saved → {plots_dir / 'comparison_table.csv'}")

    cross_dir = plots_dir / "cross_model"
    cross_dir.mkdir(parents=True, exist_ok=True)

    print("\n[plots]")
    rmc.plot_perf_auc(df, cross_dir, filtered, method, models=models)
    rmc.plot_perf_metrics(df, cross_dir, filtered, method, models=models)

    workload_df = rmc._build_workload_df(all_results)
    if not workload_df.empty:
        workload_df.to_csv(plots_dir / "workload_reduction_table.csv", index=False)
        print(f"  Saved → {plots_dir / 'workload_reduction_table.csv'}")
    rmc.plot_workload_reduction(workload_df, cross_dir, filtered, method, models=models)

    for model in models:
        model_plot_dir = plots_dir / model
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        rmc.plot_confusion_matrices(
            df, model_plot_dir, filtered, method, models=[model]
        )

    # ── Phase 3: Per-model feature analysis ───────────────────────────────────
    print(f"\n{'='*60}\n  PHASE 3: PER-MODEL FEATURE ANALYSIS\n{'='*60}")
    for model in models:
        print(f"\n  [{model}]")
        try:
            rmc.run_per_model_feature_analysis(
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
    all_imps = rmc._collect_importances(
        run_dir, scenarios, experiment="baseline", models=models
    )
    all_imps_sym = rmc._collect_importances(
        run_dir, scenarios, experiment="symbolic", models=models
    )

    all_imps_combined: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        all_imps_combined[model] = {}
        for sc in scenarios:
            sym = all_imps_sym.get(model, {}).get(sc, {})
            all_imps_combined[model][sc] = (
                sym if sym else all_imps.get(model, {}).get(sc, {})
            )

    rmc._print_overlap_table(all_imps_combined, scenarios, args.top_k)

    for ma, mb in itertools.combinations(models, 2):
        rmc.plot_cross_model_overlap(
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
        rmc.plot_shap_importance_bars(
            {model: all_imps_combined.get(model, {})},
            scenarios,
            args.top_k,
            filtered,
            model_plot_dir,
            method,
        )
        rmc.plot_common_feature_importance(
            all_imps_combined.get(model, {}),
            model,
            args.top_k,
            filtered,
            cross_dir,
            method,
        )
        rmc.plot_distinctive_feature_importance(
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
