"""
Temporal Generalization (Rolling-Horizon Decay) CLI.

Runs experiments.temporal_decay.run_temporal_decay_experiment for one
scenario: for each shortlisted (feature_set, mining_setting, granularity,
model) config, mines a schema + fits a model once on window 0's train split,
then walks the frozen schema/model/threshold forward one window at a time to
the end of the timeline, tracking SHAP/LIME importances alongside the metric
decay at every step. See experiments/temporal_decay.py's module docstring
for the full experiment design, and thesis.metrics.shortlist for the
shortlist file format (feature_set,mining_setting,granularity,model columns).

Two ways to supply the shortlist:
  --shortlist CSV            A pre-built shortlist file in that format.
  --structural-configs CSV   Build it directly from
                              attribute_mining_sweep_eda.ipynb's structural
                              shortlist (e.g. feasible_configs_all.csv --
                              growth_rate/max_depth/max_depth_attack/etc, one
                              row per config that clears the mining-only
                              precision/recall floors), crossed with
                              --granularities x --models. This is the single
                              source of truth going forward -- no separate
                              real-evaluation ranking step
                              (notebooks/config_selection.ipynb is no longer
                              part of this pipeline). Requires --granularities.

Usage:
  # From a pre-built shortlist
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --shortlist artifacts/experiments/screening_sweep/cscas/shortlist.csv

  # Directly from the mining notebook's structural shortlist
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --structural-configs artifacts/experiments/attribute_mining_parameter_grid/feasible_configs_all.csv \\
      --granularities 0.1 0.2 0.25 0.5

  # Calibrated-recall threshold instead of flat 0.5, metrics only (no SHAP/LIME)
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --shortlist shortlist.csv \\
      --threshold-mode calibrated_recall --calibrated-recall-target 0.9 \\
      --no-explanations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.system_eval.temporal_decay import run_temporal_decay_experiment
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.schemas.experiments import TemporalDecayConfig
from thesis.scripts.system_eval._common import (
    build_shortlist_from_structural_configs,
    cache_dir_for,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "screening_mining_settings.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal decay: freeze a schema+model mined/trained on window 0's "
        "train split, evaluate + explain one window at a time to the end of the timeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenario", help="Scenario name (e.g. cscas, fox).")
    shortlist_source = parser.add_mutually_exclusive_group(required=True)
    shortlist_source.add_argument(
        "--shortlist",
        type=Path,
        metavar="CSV",
        help="Pre-built shortlist CSV (feature_set,mining_setting,granularity,model columns).",
    )
    shortlist_source.add_argument(
        "--structural-configs",
        type=Path,
        dest="structural_configs",
        metavar="CSV",
        help=(
            "Build the shortlist directly from attribute_mining_sweep_eda.ipynb's structural "
            "shortlist (e.g. feasible_configs_all.csv), crossed with --granularities x --models. "
            "Requires --granularities."
        ),
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Granularities to cross with --structural-configs (ignored with --shortlist).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],
        metavar="MODEL",
        help="Models to cross with --structural-configs (ignored with --shortlist). Default: logreg",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_false",
        dest="include_baseline",
        help="Don't add a baseline row per granularity when using --structural-configs.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        dest="train_frac",
        metavar="FRAC",
        help="W_src (window 0) internal train/test split fraction. Default: 0.7",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "calibrated_recall"],
        default="fixed",
        dest="threshold_mode",
        help="How the frozen decision threshold is chosen from W_src's own train-split data. Default: fixed (0.5)",
    )
    parser.add_argument(
        "--calibrated-recall-target",
        type=float,
        default=0.90,
        dest="calibrated_recall_target",
        metavar="RECALL",
        help="Recall target for --threshold-mode calibrated_recall. Default: 0.90",
    )
    parser.add_argument(
        "--no-explanations",
        action="store_false",
        dest="compute_explanations",
        help="Skip SHAP/LIME importance tracking (metrics only, much faster).",
    )
    parser.add_argument(
        "--explain-background-n",
        type=int,
        default=100,
        dest="explain_background_n",
        metavar="N",
        help="Rows sampled from W_src's train split as the SHAP/LIME background set. Default: 100",
    )
    parser.add_argument(
        "--explain-sample-n",
        type=int,
        default=50,
        dest="explain_sample_n",
        metavar="N",
        help="Rows sampled from each horizon window to explain. Default: 50",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=1000,
        dest="lime_num_samples",
        metavar="N",
        help="Perturbed samples LIME draws per explained row. Default: 1000",
    )
    parser.add_argument(
        "--top-n-importances",
        type=int,
        default=30,
        dest="top_n_importances",
        metavar="N",
        help="Top-N features (by |importance|) kept per horizon per method. Default: 30",
    )
    parser.add_argument(
        "--mining-settings",
        type=Path,
        default=_DEFAULT_MINING_SETTINGS,
        dest="mining_settings",
        metavar="YAML",
        help=f"Named mining-setting grid axis (same file Experiment 1 used). Default: {_DEFAULT_MINING_SETTINGS}",
    )
    parser.add_argument(
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help="AIT-ADS only: use filtered alerts. Optionally a balancing method (e.g. naive50).",
    )
    parser.add_argument(
        "--window-size", type=int, default=2, metavar="W", dest="window_size"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any cached mined schema and re-mine.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        dest="n_jobs",
        metavar="N",
        help="Shortlisted configs to run concurrently (thread pool). Default: 4",
    )
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    if args.structural_configs is not None:
        if not args.granularities:
            parser.error("--structural-configs requires --granularities")
        derived = build_shortlist_from_structural_configs(
            args.structural_configs,
            args.granularities,
            args.models,
            args.include_baseline,
        )
        derived_dir = (
            _REPO / "artifacts" / "experiments" / "temporal_decay" / args.scenario
        )
        derived_dir.mkdir(parents=True, exist_ok=True)
        shortlist_path = derived_dir / "_derived_shortlist.csv"
        derived.to_csv(shortlist_path, index=False)
        print(
            f"  Derived shortlist ({len(derived)} configs) from {args.structural_configs} "
            f"→ {shortlist_path}"
        )
    else:
        shortlist_path = args.shortlist

    scenario = args.scenario
    is_cscas = dataset_for_scenario(scenario) == "cscas"
    grouping = (
        GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
        if is_cscas
        else GroupingConfig(window_size=args.window_size)
    )
    method = args.filtered if args.filtered else None
    filtered = args.filtered is not None
    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )
    alerts_path = (
        _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
        if filtered and not is_cscas
        else None
    )
    cache_dir = cache_dir_for(scenario, filtered, method, args.window_size)
    results_dir = args.output_dir / scenario if args.output_dir is not None else None

    config = TemporalDecayConfig(
        scenario=scenario,
        shortlist_path=shortlist_path,
        train_frac_within_window=args.train_frac,
        mining_settings_path=args.mining_settings,
        threshold_mode=args.threshold_mode,
        calibrated_recall_target=args.calibrated_recall_target,
        compute_explanations=args.compute_explanations,
        explain_background_n=args.explain_background_n,
        explain_sample_n=args.explain_sample_n,
        lime_num_samples=args.lime_num_samples,
        top_n_importances=args.top_n_importances,
        cache_dir=cache_dir,
        grouping=grouping,
        alerts_json_path=alerts_path,
        results_dir=results_dir,
        force_remine=args.force,
        n_jobs=args.n_jobs,
    )
    out_dir = run_temporal_decay_experiment(config)
    print(f"\n[{scenario}] Temporal decay results → {out_dir}")


if __name__ == "__main__":
    main()
