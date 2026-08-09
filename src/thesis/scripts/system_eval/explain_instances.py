"""
On-demand instance-level SHAP/LIME case studies.

Mines/fits one shortlisted config on window 0 -- exactly the way
run_temporal_decay.py's sweep does (thesis.system_eval.temporal_decay.
fit_source_window) -- then scores one horizon window, picks its most
confidently wrong predictions (false positives and/or false negatives,
ranked by |proba - threshold|), and explains each individually with SHAP +
LIME. Meant to answer "why is the model wrong about this specific alert
group at this horizon", which the aggregate drift plots in
temporal_decay_eda.ipynb can't -- run by hand, per case study, not as part
of the sweep.

Usage:
  python src/thesis/scripts/system_eval/explain_instances.py cscas \\
      --shortlist artifacts/experiments/screening_sweep/cscas/shortlist.csv \\
      --feature-set symbolic --mining-setting gr3.0_md4 --granularity 0.1 \\
      --model logreg --horizon 5 --kind fp --top-n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.experiments._shared import load_scenario_context
from thesis.system_eval.instance_explain import explain_instances_for_config
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.metrics.shortlist import load_shortlist
from thesis.scripts.system_eval._common import cache_dir_for

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "screening_mining_settings.yaml"
)


def _print_report(explanations, cfg_label: str, horizon: int) -> None:
    if not explanations:
        print(f"\nNo matching instances for {cfg_label} at horizon {horizon}.")
        return

    print(f"\n{'=' * 70}\n{cfg_label} -- horizon {horizon}\n{'=' * 70}")
    for exp in explanations:
        print(
            f"\n[{exp.error_kind.upper()}] row_index={exp.row_index} "
            f"y_true={exp.y_true} proba={exp.proba:.4f} threshold={exp.threshold:.4f} "
            f"lime_fidelity={exp.lime_fidelity:.3f}"
        )
        print("  top SHAP:")
        for feat, val in exp.shap_importances.items():
            print(f"    {feat:40s} {val:+.4f}")
        print("  top LIME:")
        for feat, val in exp.lime_importances.items():
            print(f"    {feat:40s} {val:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain a shortlisted config's most confident FP/FN instances "
        "at one horizon window, with SHAP + LIME.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenario", help="Scenario name (e.g. cscas, fox).")
    parser.add_argument(
        "--shortlist",
        type=Path,
        required=True,
        metavar="CSV",
        help="Shortlist CSV -- the one matching row is selected by "
        "--feature-set/--mining-setting/--granularity/--model.",
    )
    parser.add_argument(
        "--feature-set",
        required=True,
        dest="feature_set",
        choices=["baseline", "symbolic"],
    )
    parser.add_argument("--mining-setting", default=None, dest="mining_setting")
    parser.add_argument("--granularity", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Horizon window index (0 = W_src's own held-out test split).",
    )
    parser.add_argument(
        "--kind",
        choices=["fp", "fn", "both"],
        default="both",
        help="Which error type to explain. Default: both",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        dest="top_n",
        help="Instances to explain per error kind, ranked by |proba - threshold|. Default: 5",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, dest="train_frac")
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "calibrated_recall"],
        default="fixed",
        dest="threshold_mode",
    )
    parser.add_argument(
        "--calibrated-recall-target",
        type=float,
        default=0.90,
        dest="calibrated_recall_target",
        metavar="RECALL",
    )
    parser.add_argument(
        "--explain-background-n", type=int, default=100, dest="explain_background_n"
    )
    parser.add_argument(
        "--lime-num-samples", type=int, default=1000, dest="lime_num_samples"
    )
    parser.add_argument(
        "--top-n-importances", type=int, default=15, dest="top_n_importances"
    )
    parser.add_argument(
        "--mining-settings",
        type=Path,
        default=_DEFAULT_MINING_SETTINGS,
        dest="mining_settings",
        metavar="YAML",
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
    parser.add_argument("--random-seed", type=int, default=42, dest="random_seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the explained instances as a long-format CSV "
        "(same shape as temporal_decay.py's explanations.csv).",
    )
    args = parser.parse_args()

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

    shortlist = load_shortlist(args.shortlist)
    matches = [
        cfg
        for cfg in shortlist
        if cfg.feature_set == args.feature_set
        and cfg.mining_setting == args.mining_setting
        and cfg.granularity == args.granularity
        and cfg.model == args.model
    ]
    if not matches:
        raise SystemExit(
            f"No shortlist row matches feature_set={args.feature_set!r}, "
            f"mining_setting={args.mining_setting!r}, granularity={args.granularity}, "
            f"model={args.model!r} in {args.shortlist}"
        )
    cfg = matches[0]

    ctx = load_scenario_context(
        scenario=scenario,
        cache_dir=cache_dir,
        grouping=grouping,
        alerts_json_path=alerts_path,
        mining_settings_path=args.mining_settings,
    )

    explanations = explain_instances_for_config(
        cfg=cfg,
        scenario=scenario,
        alert_groups=ctx.alert_groups,
        alert_groups_path=ctx.alert_groups_path,
        n_total=ctx.n_total,
        base_schema=ctx.base_schema,
        mining_settings_by_name=ctx.mining_settings_by_name,
        mining_settings_path=ctx.mining_settings_path,
        horizon_window_index=args.horizon,
        kind=args.kind,
        top_n=args.top_n,
        train_frac_within_window=args.train_frac,
        threshold_mode=args.threshold_mode,
        calibrated_recall_target=args.calibrated_recall_target,
        explain_background_n=args.explain_background_n,
        lime_num_samples=args.lime_num_samples,
        top_n_importances=args.top_n_importances,
        random_seed=args.random_seed,
        force_remine=args.force,
    )

    cfg_label = (
        f"{cfg.feature_set}/{cfg.mining_setting} g={cfg.granularity:g} {cfg.model}"
    )
    _print_report(explanations, cfg_label, args.horizon)

    if args.output is not None:
        from thesis.system_eval.instance_explain import explanations_to_long_dataframe

        base_row = {
            "scenario": scenario,
            "feature_set": cfg.feature_set,
            "mining_setting": cfg.mining_setting,
            "granularity": cfg.granularity,
            "model": cfg.model,
        }
        df = explanations_to_long_dataframe(explanations, base_row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
