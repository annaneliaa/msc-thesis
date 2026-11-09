"""
Paper Check 1: re-select the two-tree mining settings on W0 only.

Splits W0 (CSCAS baseline train period) 80/20 in time order, sweeps
max_depth x max_depth_attack x min_samples_leaf x max_diverse_rounds
(2 x 3 x 3 x 3 = 54 mines on tuning-train, RF x 5 seeds each), scores every
point on tuning-val, and compares the thesis-rule and PR-AUC picks with
Table E.13. Writes stage1_survivors.csv, stage2_grid.csv (checkpointed per
grid point), selection.csv, thesis_rule_trace.csv under
artifacts/experiments/mining_setting_selection/<ts>/ (mirrored to
results/paper/mining-setting-selection/<ts>/ for a full run). Analysis:
notebooks/paper/01_mining_setting_selection.ipynb.

Usage:
    python src/thesis/scripts/system_eval/run_mining_setting_selection.py
    python src/thesis/scripts/system_eval/run_mining_setting_selection.py --smoke   # 2 grid points, 1 seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.mining_setting_selection import (  # noqa: E402
    PAPER_GRANULARITY,
    THESIS_SETTING,
    MiningSettingSelectionConfig,
    run_mining_setting_selection,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gran", type=float, default=PAPER_GRANULARITY)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--min-val-attacks", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--stage1-full-timeline",
        action="store_true",
        help="also compute the Stage-1 survivor curve on the full timeline (slow, read-only)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2 grid points (thesis setting + md2), seeds as given",
    )
    p.add_argument("--force", action="store_true", help="ignore cached mined schemas")
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()

    grid = None
    if a.smoke:
        base = {k: v for k, v in THESIS_SETTING.items() if k != "min_growth_rate"}
        grid = [base, {**base, "max_depth": 2}]

    out = run_mining_setting_selection(
        MiningSettingSelectionConfig(
            gran=a.gran,
            train_frac=a.train_frac,
            min_val_attacks=a.min_val_attacks,
            seeds=tuple(a.seeds),
            stage1_full_timeline=a.stage1_full_timeline,
            grid=grid,
            force_remine=a.force,
            out_dir=a.output_dir,
        )
    )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
