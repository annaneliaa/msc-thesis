"""
Synthetic drift test for the monitor on frozen DeepCASE: inject controlled
drift into the CSCAS test events (labels unchanged) and check which monitor
signals fire. Reuses a deepcase_comparison run's tuned settings and cached
W0 ContextBuilders; one attention query per (seed, injection) on the GPU.

  python src/thesis/scripts/system_eval/run_deepcase_synthetic_drift.py
  python src/thesis/scripts/system_eval/run_deepcase_synthetic_drift.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_synthetic_drift import (  # noqa: E402
    KINDS,
    DeepCaseDriftConfig,
    run_deepcase_synthetic_drift,
)

_EXP = _REPO / "artifacts" / "experiments"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--base-run",
        type=Path,
        default=_EXP / "deepcase_comparison" / "cscas" / "20261125_150901",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--inject-from-h", type=int, default=6)
    p.add_argument("--audit-rate", type=float, default=0.05)
    p.add_argument("--kinds", nargs="+", choices=KINDS[1:], default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    run_deepcase_synthetic_drift(
        DeepCaseDriftConfig(
            base_run=str(args.base_run),
            seeds=tuple(args.seeds),
            inject_from_h=args.inject_from_h,
            audit_rate=args.audit_rate,
            kinds=tuple(args.kinds) if args.kinds else None,
            out_dir=str(args.output_dir) if args.output_dir else None,
        )
    )


if __name__ == "__main__":
    main()
