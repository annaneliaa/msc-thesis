"""
Ingest the CSCAS dataset (pre-grouped Suricata rows) into alert_groups_raw.json.

CSCAS has no alert-by-alert stream to tokenise/group like AIT-ADS does, so
there's no "baseline experiment" prerequisite here — this is the entire
ingestion step. Run it once per grouping method before any script that
consumes cscas alert_groups, e.g.:

    python src/thesis/scripts/run_mining_window_sweep.py cscas \\
        --grouping-method cscas_pregrouped ...

Usage:
    python src/thesis/scripts/run_ingest_cscas.py
    python src/thesis/scripts/run_ingest_cscas.py --csv-path data/cscas/dataset-labeled-anon-ip.csv

    # Group by (internal target IP, 1h window) instead of one basket per
    # signature — see group_cscas_rows_by_target_window for why this is
    # needed for itemset/sequence mining to find cross-signature patterns:
    python src/thesis/scripts/run_ingest_cscas.py \\
        --grouping-method cscas_target_window --window-seconds 3600

    # Same idea, but a session-gap scheme (closes a basket after a quiet
    # gap, caps worst case at session_length) instead of a fixed window --
    # see group_cscas_rows_by_target_session for why this bounds detection
    # latency more tightly:
    python src/thesis/scripts/run_ingest_cscas.py \\
        --grouping-method cscas_target_session --session-timeout 1800 --session-length 21600

Output:
    artifacts/cache/cscas/groups/<grouping-method>/alert_groups/alert_groups_raw.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.grouping.group_alerts import (
    CSCAS_PREGROUPED_METHOD,
    CSCAS_TARGET_WINDOW_METHOD,
    CSCAS_TARGET_WINDOW_SECONDS,
    CSCAS_TARGET_SESSION_METHOD,
    CSCAS_TARGET_SESSION_TIMEOUT_SECONDS,
    CSCAS_TARGET_SESSION_LENGTH_SECONDS,
)
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
            "artifacts/cache/cscas/groups/<grouping-method>"
        ),
    )
    parser.add_argument(
        "--grouping-method",
        type=str,
        default=CSCAS_PREGROUPED_METHOD,
        choices=[
            CSCAS_PREGROUPED_METHOD,
            CSCAS_TARGET_WINDOW_METHOD,
            CSCAS_TARGET_SESSION_METHOD,
        ],
        help=(
            f"'{CSCAS_PREGROUPED_METHOD}' (default): one basket per CSV row/signature. "
            f"'{CSCAS_TARGET_WINDOW_METHOD}': baskets span multiple signatures "
            "grouped by (internal target IP, fixed time window) — required for "
            "itemset mining to find cross-signature co-occurrence and for "
            "sequence mining to have any sorted_items to work with. "
            f"'{CSCAS_TARGET_SESSION_METHOD}': same idea, but grouped by "
            "(internal target IP, session-gap) instead of a fixed window — "
            "closes a basket after a quiet gap rather than always waiting "
            "for the window to close, bounding detection latency more "
            "tightly for the common case of a target that goes quiet."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=CSCAS_TARGET_WINDOW_SECONDS,
        metavar="SECONDS",
        help=(
            f"Time window size for --grouping-method {CSCAS_TARGET_WINDOW_METHOD}. "
            f"Default: {CSCAS_TARGET_WINDOW_SECONDS:.0f}s. Ignored otherwise."
        ),
    )
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=CSCAS_TARGET_SESSION_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            f"Quiet-gap timeout for --grouping-method {CSCAS_TARGET_SESSION_METHOD}: "
            "a basket closes once no new row for a target has arrived for this "
            f"long. Default: {CSCAS_TARGET_SESSION_TIMEOUT_SECONDS:.0f}s. Ignored otherwise."
        ),
    )
    parser.add_argument(
        "--session-length",
        type=float,
        default=CSCAS_TARGET_SESSION_LENGTH_SECONDS,
        metavar="SECONDS",
        help=(
            f"Hard cap on basket span for --grouping-method {CSCAS_TARGET_SESSION_METHOD}: "
            "forces a basket closed after this long even if the target is still "
            f"active. Default: {CSCAS_TARGET_SESSION_LENGTH_SECONDS:.0f}s. Ignored otherwise."
        ),
    )
    args = parser.parse_args()

    out_path = ingest_cscas_scenario(
        csv_path=args.csv_path,
        cache_dir=args.cache_dir,
        grouping_method=args.grouping_method,
        window_seconds=args.window_seconds,
        session_timeout=args.session_timeout,
        session_length=args.session_length,
    )
    print(f"\nDone. alert_groups_raw.json → {out_path}")


if __name__ == "__main__":
    main()
