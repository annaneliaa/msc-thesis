"""
Paper Check 3, steps 2-3: walk-forward (a)/(b)/(c) ablation on CSCAS.

  a  Base(5), no mining
  b  Base(5) + mining on all items
  c  Base(5) + mining without signature-derived items

RF (class-weighted), one fit per seed, baseline_split windows at gran 0.05.
Writes per_step_results.csv, predictions.csv.gz (attack + FP rows, with
seen/unseen-signature flags), ablation_summary.csv under
artifacts/experiments/signature_leakage/<ts>/ (mirrored to
results/paper/signature-leakage/<ts>/ for a full run). Analysis:
notebooks/paper/03_signature_leakage.ipynb.

Usage:
    python src/thesis/scripts/system_eval/run_signature_leakage.py
    python src/thesis/scripts/system_eval/run_signature_leakage.py --max-steps 1 --seeds 0   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.signature_leakage import (  # noqa: E402
    ARMS,
    PAPER_GRANULARITY,
    SignatureAblationConfig,
    run_signature_ablation,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gran", type=float, default=PAPER_GRANULARITY)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--model", default="rf")
    p.add_argument(
        "--mining-settings",
        type=Path,
        default=SignatureAblationConfig.mining_settings_path,
    )
    p.add_argument("--mining-setting", default=SignatureAblationConfig.mining_setting)
    p.add_argument(
        "--threshold-mode", default="fixed", choices=["fixed", "calibrated_recall"]
    )
    p.add_argument(
        "--max-steps", type=int, default=None, help="cap the walk (smoke tests)"
    )
    p.add_argument("--force", action="store_true", help="ignore cached mined schemas")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--experiment",
        default="signature_leakage",
        help="output folder name under artifacts/experiments/ (and results/paper/)",
    )
    a = p.parse_args()

    out = run_signature_ablation(
        SignatureAblationConfig(
            gran=a.gran,
            seeds=tuple(a.seeds),
            arms=tuple(a.arms),
            model=a.model,
            mining_setting=a.mining_setting,
            mining_settings_path=a.mining_settings,
            threshold_mode=a.threshold_mode,
            max_steps=a.max_steps,
            force_remine=a.force,
            out_dir=a.output_dir,
            experiment=a.experiment,
        )
    )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
