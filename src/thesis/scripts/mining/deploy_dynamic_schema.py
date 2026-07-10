"""
Mine a window of an existing (already-ingested) alert_groups file and deploy
the result as the next Vk dynamic schema for a scenario -- the deployment-
scoped registry a drift monitor compares live alert traffic against
(thesis.features.dynamic_schema_registry), distinct from the experiment-
scoped SymbolicFeatureSchema versions mine_attribute_schema.py produces.

This is a deliberate, explicit action: unlike attribute_mining_job.py
(called at high frequency by sweeps/screening/walk-forward experiments
purely for evaluation), nothing else calls
thesis.features.dynamic_schema_service.mine_and_deploy_dynamic_schema
automatically, so routine evaluation mining never floods the deployment
history.

Usage:
  python src/thesis/scripts/mining/deploy_dynamic_schema.py \
      --scenario cscas \
      --alert-groups-path artifacts/cache/cscas/groups/cscas_pregrouped/alert_groups/alert_groups_raw.json

  # Deploy from only the first 50% of the (chronologically-sorted) timeline:
  python src/thesis/scripts/mining/deploy_dynamic_schema.py \
      --scenario cscas --alert-groups-path ... --win-end-frac 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.features.dynamic_schema_service import (  # noqa: E402
    _load_sorted_labeled_alert_groups,
    mine_and_deploy_dynamic_schema,
)
from thesis.schemas.mining import AttributeMiningConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine a window of alert_groups and deploy it as the next Vk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", required=True, help="Scenario name.")
    parser.add_argument(
        "--alert-groups-path",
        required=True,
        type=Path,
        help="Path to an already-ingested alert_groups JSON file.",
    )
    parser.add_argument(
        "--win-start-frac",
        type=float,
        default=0.0,
        help="Fraction of the chronologically-sorted timeline to start mining at.",
    )
    parser.add_argument(
        "--win-end-frac",
        type=float,
        default=1.0,
        help="Fraction of the chronologically-sorted timeline to stop mining at.",
    )
    parser.add_argument("--min-attack-coverage", type=float, default=0.05)
    parser.add_argument("--min-benign-coverage", type=float, default=0.05)
    parser.add_argument("--min-growth-rate", type=float, default=3.0)
    parser.add_argument("--max-p-value", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--class-weight", type=str, default="balanced")
    args = parser.parse_args()

    config = AttributeMiningConfig()
    config.contrast.min_attack_coverage = args.min_attack_coverage
    config.contrast.min_benign_coverage = args.min_benign_coverage
    config.contrast.min_growth_rate = args.min_growth_rate
    config.contrast.max_p_value = args.max_p_value
    config.tree.max_depth = args.max_depth
    config.tree.min_samples_leaf = args.min_samples_leaf
    config.tree.class_weight = args.class_weight

    n_total = len(_load_sorted_labeled_alert_groups(args.alert_groups_path))
    win_start_idx = int(args.win_start_frac * n_total)
    win_end_idx = int(args.win_end_frac * n_total)

    print(
        f"Mining {args.scenario} window [{win_start_idx}:{win_end_idx}) "
        f"of {n_total} labeled alert_groups..."
    )
    schema_path = mine_and_deploy_dynamic_schema(
        alert_groups_path=args.alert_groups_path,
        scenario_name=args.scenario,
        win_start_idx=win_start_idx,
        win_end_idx=win_end_idx,
        config=config,
    )
    print(f"Deployed dynamic schema -> {schema_path}")


if __name__ == "__main__":
    main()
