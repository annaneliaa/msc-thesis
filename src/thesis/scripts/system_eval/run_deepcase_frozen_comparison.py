"""
Frozen comparison table, ours vs DeepCASE on CSCAS (DeepCASE implementation
plan B4). Reads a finished deepcase_comparison run (step 5.2) and our B3
runs (run_deepcase_ours.sh, POLICY=never); writes per_seed.csv, table.csv,
per_horizon.csv, workload_curve.csv, workload_vs_recall_<subset>.png and
summary.txt.

  python src/thesis/scripts/system_eval/run_deepcase_frozen_comparison.py
  python src/thesis/scripts/system_eval/run_deepcase_frozen_comparison.py \\
      --ours-root artifacts/experiments/deepcase_comparison_ours/cscas/never_audit-auto
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_frozen_comparison import (  # noqa: E402
    SUBSETS,
    format_table,
    plot_workload_curve,
    run_frozen_comparison,
)

_EXP = _REPO / "artifacts" / "experiments"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--deepcase-run",
        type=Path,
        default=_EXP / "deepcase_comparison" / "cscas" / "20261123_200913",
    )
    p.add_argument(
        "--ours-root",
        type=Path,
        default=_EXP / "deepcase_comparison_ours" / "cscas" / "never_audit-auto",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or (
        _EXP
        / "deepcase_frozen_comparison"
        / "cscas"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    res = run_frozen_comparison(args.deepcase_run, args.ours_root, out_dir)
    table = res["table"]

    lines = [
        "Frozen comparison: ours vs DeepCASE on CSCAS (test period, windows 1-10)",
        f"DeepCASE run: {args.deepcase_run}",
        f"Ours: {args.ours_root}",
        "",
    ]
    for subset in SUBSETS:
        for op in ("native", "matched_workload"):
            t = table[(table.subset == subset) & (table.operating_point == op)]
            if t.empty:
                continue
            lines += [f"== {subset} signatures, {op}", format_table(t), ""]
        if not res["curves"].empty:
            plot_workload_curve(
                res["curves"],
                res["per_seed"],
                out_dir / f"workload_vs_recall_{subset}.png",
                subset,
            )
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print(text)
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
