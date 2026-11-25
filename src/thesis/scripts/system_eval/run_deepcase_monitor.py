"""
Our monitor's signals on frozen DeepCASE, observe-only (DeepCASE
implementation plan B5). Reads a deepcase_comparison run (step 5.2); writes
per_window_signals.csv and summary.txt.

  python src/thesis/scripts/system_eval/run_deepcase_monitor.py
  python src/thesis/scripts/system_eval/run_deepcase_monitor.py --deepcase-run <dir>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_monitor import (  # noqa: E402
    AUDIT_RATE,
    MIN_SAMPLES,
    run_deepcase_monitor,
    summarise,
)

_EXP = _REPO / "artifacts" / "experiments"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--deepcase-run",
        type=Path,
        default=_EXP / "deepcase_comparison" / "cscas" / "20261123_200913",
    )
    p.add_argument("--audit-rate", type=float, default=AUDIT_RATE)
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.output_dir or (
        _EXP
        / "deepcase_monitor"
        / "cscas"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    signals = run_deepcase_monitor(
        args.deepcase_run, out_dir, args.audit_rate, args.min_samples
    )
    text = "\n".join(
        [
            "Our monitor on frozen DeepCASE (observe-only)",
            f"DeepCASE run: {args.deepcase_run}",
            f"audit_rate={args.audit_rate}, min_samples={args.min_samples}",
            "cluster precision: "
            + (
                "on"
                if signals["has_cluster_ids"].all()
                else "OFF (run has no per-row cluster ids -- rerun deepcase_comparison)"
            ),
            "",
            summarise(signals),
        ]
    )
    (out_dir / "summary.txt").write_text(text)
    print(text)
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
