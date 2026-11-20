"""
Settings sensitivity for the mining comparison (HANDOFF "Deferred" item 2):
instead of tuning, every method -- the contrast tree included -- runs at three
fixed settings of its main knob, with its default in the middle, and we check
whether the paper's conclusions hold across all of them.

Variants (SETTINGS_GRID): "default" is the main comparison; "low" / "high"
move every method's knob down / up at once (each method's mining is
independent, so one run per variant covers all five):

  method          knob                          low     default   high
  Eclat           support threshold*            0.02    0.05      0.10
  PrefixSpan      support threshold*            0.02    0.05      0.10
  RIPPER          prune-set share (prune_size)  0.20    0.33      0.50
  MCTSExtent      search iterations             500     1000      2000
  Contrast tree   diverse mining rounds         1       2         3

* min support and min support difference together: the miners keep a
  pattern only if its support beats the other class's by the difference,
  which floors the support at that value too, so lowering the min support
  alone below 0.05 changes nothing (found 2026-11-20: identical output).

Scores: the comparison's own metric (PR-AUC; F1 alongside), seed-averaged per
step / fold, then averaged. Per method, its worst and best variant are picked
by that mean -- with test knowledge, so they are never reported as a method's
score; the headline stays at the default.

Stress test per contrast (A vs. B): pair, fold by fold, one method at its
best variant against the other at its worst, in both directions. A
conclusion "B > A" is robust to settings if B's worst still beats A's best.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from thesis.paths import ROOT
from thesis.system_eval.mining_comparison import NO_MINING, per_step_means
from thesis.system_eval.mining_mini_experiment import MiniConfig

SETTINGS_GRID_FILE = ROOT / "src/thesis/configs/mining_comparison_settings_grid.yaml"

SETTINGS_GRID: dict[str, dict] = {
    "low": {
        "eclat_min_support": 0.02,
        "eclat_min_support_diff": 0.02,
        "prefixspan_min_support": 0.02,
        "prefixspan_min_support_diff": 0.02,
        "ripper_prune_size": 0.20,
        "mctsextent_iterations": 500,
        "contrast_tree_setting": "gr3_md1_mda4",
        "mining_settings_path": SETTINGS_GRID_FILE,
    },
    "default": {},
    "high": {
        "eclat_min_support": 0.10,
        "eclat_min_support_diff": 0.10,
        "prefixspan_min_support": 0.10,
        "prefixspan_min_support_diff": 0.10,
        "ripper_prune_size": 0.50,
        "mctsextent_iterations": 2000,
        "contrast_tree_setting": "gr3_md1_mda4_rounds3",
        "mining_settings_path": SETTINGS_GRID_FILE,
    },
}
VARIANTS = tuple(SETTINGS_GRID)

# The knob each method varies, for labelling (MiniConfig field per method).
KNOB = {
    "eclat": "eclat_min_support",
    "prefixspan": "prefixspan_min_support",
    "ripper": "ripper_prune_size",
    "mctsextent": "mctsextent_iterations",
    "contrast_tree": "contrast_tree_setting",
}

# The paper's contrasts as (name, A, B): "B vs. A".
CONTRASTS = (
    ("label after → during (atemporal)", "eclat", "contrast_tree"),
    ("label after → during (temporal)", "prefixspan", "mctsextent"),
    ("two-step → rule set", "contrast_tree", "ripper"),
    ("atemporal → temporal", "eclat", "prefixspan"),
)


def apply_variant(mini: MiniConfig, variant: str) -> MiniConfig:
    """`mini` with the variant's settings. 'default' returns it unchanged."""
    if variant not in SETTINGS_GRID:
        raise ValueError(f"unknown settings variant {variant!r}; use {VARIANTS}")
    return replace(mini, **SETTINGS_GRID[variant])


def knob_value(method: str, variant: str) -> object:
    field = KNOB[method]
    return SETTINGS_GRID[variant].get(field, getattr(MiniConfig(), field))


# --- Summary ----------------------------------------------------------------


def _seed_means(run_dirs: dict[str, Path], metric: str) -> pd.DataFrame:
    """(variant, arm, step_i) -> seed-averaged metric, long format."""
    frames = []
    for variant, d in run_dirs.items():
        m = per_step_means(pd.read_csv(Path(d) / "per_step_results.csv"))
        frames.append(m[["arm", "step_i", metric]].assign(variant=variant))
    return pd.concat(frames, ignore_index=True)


def settings_spread(run_dirs: dict[str, Path], metric: str = "pr_auc") -> pd.DataFrame:
    """Per method: the mean metric at each variant, and which variant is its
    worst / best (by mean over steps or folds)."""
    long = _seed_means(run_dirs, metric)
    wide = long.groupby(["arm", "variant"])[metric].mean().unstack("variant")
    wide = wide.reindex(columns=[v for v in VARIANTS if v in wide.columns])
    out = wide.copy()
    mined = out.index != NO_MINING
    out["worst"] = np.where(mined, wide.min(axis=1), np.nan)
    out["best"] = np.where(mined, wide.max(axis=1), np.nan)
    out["worst_variant"] = np.where(mined, wide.idxmin(axis=1), "")
    out["best_variant"] = np.where(mined, wide.idxmax(axis=1), "")
    out["range"] = out["best"] - out["worst"]
    out.insert(0, "metric", metric)
    return out.reset_index()


def stress_contrasts(
    run_dirs: dict[str, Path],
    metric: str = "pr_auc",
    contrasts=CONTRASTS,
) -> pd.DataFrame:
    """Per contrast and direction: one method at its worst variant against the
    other at its best, paired over steps / folds (mean Δ, wins, Wilcoxon p).
    Plus the default-vs-default row for reference."""
    long = _seed_means(run_dirs, metric)
    spread = settings_spread(run_dirs, metric).set_index("arm")
    piv = long.pivot_table(index="step_i", columns=["arm", "variant"], values=metric)
    rows = []
    for name, a, b in contrasts:
        cases = [
            ("default vs default", (b, "default"), (a, "default")),
            (
                f"{b} worst vs {a} best",
                (b, spread.loc[b, "worst_variant"]),
                (a, spread.loc[a, "best_variant"]),
            ),
            (
                f"{a} worst vs {b} best",
                (a, spread.loc[a, "worst_variant"]),
                (b, spread.loc[b, "best_variant"]),
            ),
        ]
        for case, (x, xv), (y, yv) in cases:
            if (x, xv) not in piv or (y, yv) not in piv:
                continue
            dx, dy = piv[(x, xv)], piv[(y, yv)]
            diff = (dx - dy).dropna()
            rows.append(
                {
                    "contrast": name,
                    "metric": metric,
                    "case": case,
                    "x": f"{x}@{xv}",
                    "y": f"{y}@{yv}",
                    "mean x": dx.mean(),
                    "mean y": dy.mean(),
                    "mean Δ (x−y)": diff.mean(),
                    "x wins": f"{int((diff > 0).sum())}/{len(diff)}",
                    "wilcoxon_p": (
                        sps.wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_settings_summary(out_dir: Path, run_dirs: dict[str, Path]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spread = pd.concat(
        [settings_spread(run_dirs, m) for m in ("pr_auc", "f1")], ignore_index=True
    )
    stress = pd.concat(
        [stress_contrasts(run_dirs, m) for m in ("pr_auc", "f1")], ignore_index=True
    )
    spread.to_csv(out_dir / "settings_spread.csv", index=False)
    stress.to_csv(out_dir / "settings_stress.csv", index=False)
    pd.Series({k: str(v) for k, v in run_dirs.items()}).to_json(
        out_dir / "settings_runs.json", indent=2
    )
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        text = (
            spread.round(4).to_string(index=False)
            + "\n\n"
            + stress.round(4).to_string(index=False)
        )
    (out_dir / "settings_summary.txt").write_text(text + "\n")
    print(text)
