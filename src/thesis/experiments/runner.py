"""
Experiment runner.

Usage:
    python -m thesis.experiments.runner <experiment> <scenario> [scenario2 ...]

Experiments:
    baseline   Preprocess → encode (base features) → train → evaluate
    symbolic   Preprocess → mine → build symbolic schema → encode → train → evaluate

Examples:
    python -m thesis.experiments.runner baseline fox
    python -m thesis.experiments.runner symbolic fox
    python -m thesis.experiments.runner symbolic fox bear wolf
    python -m thesis.experiments.runner symbolic fox --filter-config src/thesis/configs/mining_filters_discriminative.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.experiments.baseline import (
    BaselineExperimentConfig,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a thesis experiment for one or more scenarios."
    )
    parser.add_argument(
        "experiment",
        choices=["baseline", "symbolic"],
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
        help="Path to a YAML mining filter config (symbolic experiment only).",
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


if __name__ == "__main__":
    main()
