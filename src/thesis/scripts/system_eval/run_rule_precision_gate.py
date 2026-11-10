"""
Paper Check 2: precision gate on mined rules -- walk-forward gate off vs. on.

Arms (same mined schema per step, RF x seeds, baseline_split windows at
gran 0.05): b (gate off) and b_g1.5 / b_g2 / b_g3 (rules failing
attack rate >= g*pi / <= pi/g on their mining window dropped before the fit).
Writes per_step_results.csv, rules.csv (every mined rule per step with its
attack rate on Wi and Wi+1 and the gate verdicts), predictions.csv.gz,
ablation_summary.csv under artifacts/experiments/rule_precision_gate/<ts>/
(mirrored to results/paper/rule-precision-gate/<ts>/ for a full run).
Analysis: notebooks/paper/02_rule_precision_gate.ipynb.

Usage:
    python src/thesis/scripts/system_eval/run_rule_precision_gate.py
    python src/thesis/scripts/system_eval/run_rule_precision_gate.py --max-steps 1 --seeds 0   # smoke test
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

DEFAULT_ARMS = ["b", "b_g1.5", "b_g2", "b_g3"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gran", type=float, default=PAPER_GRANULARITY)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
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
            experiment="rule_precision_gate",
        )
    )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
