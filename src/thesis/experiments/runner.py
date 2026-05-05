"""
Experiment runner.

Usage:
    python -m thesis.experiments.runner <experiment> <scenario> [scenario2 ...]

Experiments:
    baseline   Preprocess → encode (base features) → train → evaluate
    symbolic   Preprocess → mine → build symbolic schema → encode → train → evaluate
    compare    Run both baseline and symbolic for the same scenario and save
               a side-by-side result to artifacts/experiments/<scenario>/

Examples:
    python -m thesis.experiments.runner baseline fox
    python -m thesis.experiments.runner symbolic fox
    python -m thesis.experiments.runner symbolic fox bear wolf
    python -m thesis.experiments.runner symbolic fox --filter-config src/thesis/configs/mining_filters_discriminative.yaml
    python -m thesis.experiments.runner compare fox
    python -m thesis.experiments.runner compare fox --filter-config src/thesis/configs/mining_filters_strict.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from thesis.experiments.baseline import (
    BaselineExperimentConfig,
    _EXPERIMENTS_DIR,
    run_baseline_experiment,
)
from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)


def run_baseline(scenario: str) -> None:
    config = BaselineExperimentConfig(scenario=scenario)
    result = run_baseline_experiment(config)
    print(
        f"\n[{scenario}] baseline done — "
        f"AUC={result.auc:.4f}  "
        f"features={result.n_features}  "
        f"transactions={result.n_transactions}"
    )


def run_symbolic(scenario: str, filter_config: Path | None = None) -> None:
    config = SymbolicExperimentConfig(
        scenario=scenario,
        filter_config=filter_config,
    )
    result = run_symbolic_experiment(config)
    print(
        f"\n[{scenario}] symbolic done — "
        f"AUC={result.auc:.4f}  "
        f"features={result.n_features}  "
        f"transactions={result.n_transactions}"
    )


def run_compare(scenario: str, filter_config: Path | None = None) -> None:
    print("\n--- Phase 1/2: baseline ---")
    baseline_config = BaselineExperimentConfig(scenario=scenario)
    baseline = run_baseline_experiment(baseline_config)

    print("\n--- Phase 2/2: symbolic ---")
    symbolic_config = SymbolicExperimentConfig(
        scenario=scenario,
        filter_config=filter_config,
    )
    symbolic = run_symbolic_experiment(symbolic_config)

    auc_delta = symbolic.auc - baseline.auc
    feat_delta = symbolic.n_features - baseline.n_features

    print(f"\n{'─' * 50}")
    print(f"  {'':20s}  {'baseline':>10s}  {'symbolic':>10s}  {'delta':>8s}")
    print(f"  {'─' * 52}")
    print(
        f"  {'AUC':20s}  {baseline.auc:>10.4f}  {symbolic.auc:>10.4f}  {auc_delta:>+8.4f}"
    )
    print(
        f"  {'features':20s}  {baseline.n_features:>10d}  {symbolic.n_features:>10d}  {feat_delta:>+8d}"
    )
    print(
        f"  {'transactions':20s}  {baseline.n_transactions:>10d}  {symbolic.n_transactions:>10d}"
    )
    print(f"{'─' * 50}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = _EXPERIMENTS_DIR / scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"compare_{timestamp}.json"

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "compare",
                "scenario": scenario,
                "timestamp": timestamp,
                "filter_config": str(filter_config) if filter_config else None,
                "baseline": {
                    "schema_name": baseline.schema_name,
                    "schema_version": baseline.schema_version,
                    "auc": baseline.auc,
                    "n_features": baseline.n_features,
                    "n_transactions": baseline.n_transactions,
                    "metrics": baseline.metrics,
                    "results_file": str(baseline.results_file),
                },
                "symbolic": {
                    "schema_name": symbolic.schema_name,
                    "schema_version": symbolic.schema_version,
                    "symbolic_schema_path": str(symbolic.symbolic_schema_path),
                    "auc": symbolic.auc,
                    "n_features": symbolic.n_features,
                    "n_transactions": symbolic.n_transactions,
                    "metrics": symbolic.metrics,
                    "results_file": str(symbolic.results_file),
                },
                "delta": {
                    "auc": round(auc_delta, 6),
                    "n_features": feat_delta,
                },
            },
            f,
            indent=2,
        )

    print(f"  Compare results → {results_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a thesis experiment for one or more scenarios."
    )
    parser.add_argument(
        "experiment",
        choices=["baseline", "symbolic", "compare"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="One or more scenario names (e.g. fox bear).",
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        help="Path to a YAML mining filter config (symbolic and compare only).",
    )

    args = parser.parse_args()

    for scenario in args.scenarios:
        print(f"\n{'=' * 60}")
        print(f" Experiment : {args.experiment}")
        print(f" Scenario   : {scenario}")
        print(f"{'=' * 60}")

        if args.experiment == "baseline":
            run_baseline(scenario)
        elif args.experiment == "symbolic":
            run_symbolic(scenario, filter_config=args.filter_config)
        elif args.experiment == "compare":
            run_compare(scenario, filter_config=args.filter_config)


if __name__ == "__main__":
    main()
