"""
Ingest AIT-ADS scenarios (alert-by-alert stream) into alert_groups_raw.json.

Mirrors run_ingest_cscas.py for the AIT-ADS dataset. For each scenario this
converts data/alerts_csv/<scenario>_alerts.txt to alerts.json, tokenises and
groups alerts into the TokenCache, ensures the feature manifest exists, and
builds alert_groups_raw.json — the prerequisite artifact for scripts that
only need alert_groups (not a trained model), e.g.:

    python src/thesis/scripts/run_mining_window_sweep.py fox harrison ...

Previously the documented prerequisite was `python -m thesis.experiments.runner
baseline <scenario>`, which also trains and persists a model. This script
does only the ingestion prefix, so it's faster when all you need is the
alert_groups cache.

Usage:
    python src/thesis/scripts/run_ingest_ait_ads.py fox harrison
    python src/thesis/scripts/run_ingest_ait_ads.py --all
    python src/thesis/scripts/run_ingest_ait_ads.py --all --window-size 5

Output (per scenario):
    artifacts/cache/<scenario>/groups/fixed_window/alert_groups/alert_groups_raw.json
"""

from __future__ import annotations

import argparse

from thesis.config import GroupingConfig
from thesis.configs import load_scenarios
from thesis.pipeline.pipeline import ingest_ait_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest AIT-ADS scenarios into alert_groups_raw.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenarios", nargs="*", help="Scenario names (e.g. fox harrison)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all AIT-ADS scenarios (see src/thesis/configs/scenarios.json).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        metavar="SECONDS",
        help="Fixed-window size in seconds for alert grouping (default: 2).",
    )
    args = parser.parse_args()

    if args.all:
        scenarios = load_scenarios("ait-ads")
        if args.scenarios:
            print(f"[warn] --all overrides positional scenarios {args.scenarios}")
    else:
        scenarios = args.scenarios
    if not scenarios:
        parser.error("Provide at least one scenario name or use --all.")

    grouping = GroupingConfig(window_size=args.window_size)

    for scenario in scenarios:
        print(f"\n[scenario={scenario}]")
        out_path = ingest_ait_scenario(scenario, grouping=grouping)
        print(f"  Done → {out_path}")


if __name__ == "__main__":
    main()
