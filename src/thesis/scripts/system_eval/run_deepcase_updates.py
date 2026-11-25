"""
DeepCASE kept up to date over the CSCAS test windows (DeepCASE
implementation plan B6): never / schedule / combined (our monitor), with the
same label accounting as our system. Reuses a deepcase_comparison run's
tuned settings and cached W0 ContextBuilders.

  python src/thesis/scripts/system_eval/run_deepcase_updates.py --policy schedule
  python src/thesis/scripts/system_eval/run_deepcase_updates.py --policy combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_updates import (  # noqa: E402
    ACTIONS,
    POLICIES,
    UpdateConfig,
    run_deepcase_updates,
)

_EXP = _REPO / "artifacts" / "experiments"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--base-run",
        type=Path,
        default=_EXP / "deepcase_comparison" / "cscas" / "20261123_200913",
        help="deepcase_comparison run whose tuned settings (config.json) to reuse",
    )
    p.add_argument("--policy", choices=POLICIES, default="never")
    p.add_argument("--schedule-action", choices=ACTIONS, default="REMINE_AND_RETRAIN")
    p.add_argument("--audit", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--audit-rate", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument(
        "--update-epochs",
        type=int,
        default=None,
        help="ContextBuilder epochs per retrain (default: the base run's)",
    )
    p.add_argument(
        "--no-remine",
        action="store_true",
        help="Turn every REMINE_AND_RETRAIN into RETRAIN_ONLY (keep the ContextBuilder)",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    run_deepcase_updates(
        UpdateConfig(
            base_run=str(args.base_run),
            policy=args.policy,
            schedule_action=args.schedule_action,
            audit=args.audit,
            audit_rate=args.audit_rate,
            seeds=tuple(args.seeds) if args.seeds else None,
            update_epochs=args.update_epochs,
            allow_remine=not args.no_remine,
            out_dir=str(args.output_dir) if args.output_dir else None,
        )
    )


if __name__ == "__main__":
    main()
