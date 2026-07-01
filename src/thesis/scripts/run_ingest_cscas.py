"""
Ingest the CSCAS dataset (pre-grouped Suricata rows) into alert_groups_raw.json.

CSCAS has no alert-by-alert stream to tokenise/group like AIT-ADS does, so
there's no "baseline experiment" prerequisite here — this is the entire
ingestion step. Run it once before any script that consumes cscas
alert_groups, e.g.:

    python src/thesis/scripts/run_mining_window_sweep.py cscas \\
        --grouping-method suricata_grouped ...

Usage:
    python src/thesis/scripts/run_ingest_cscas.py
    python src/thesis/scripts/run_ingest_cscas.py --csv-path data/cscas/dataset-labeled-anon-ip.csv

Output:
    artifacts/cache/cscas/groups/suricata_grouped/alert_groups/alert_groups_raw.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.pipeline.pipeline import ingest_cscas_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the CSCAS CSV into alert_groups_raw.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        metavar="CSV",
        help="Path to the CSCAS CSV. Default: data/cscas/dataset-labeled-anon-ip.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output cache directory. Default: "
            "artifacts/cache/cscas/groups/suricata_grouped"
        ),
    )
    args = parser.parse_args()

    out_path = ingest_cscas_scenario(csv_path=args.csv_path, cache_dir=args.cache_dir)
    print(f"\nDone. alert_groups_raw.json → {out_path}")


if __name__ == "__main__":
    main()
