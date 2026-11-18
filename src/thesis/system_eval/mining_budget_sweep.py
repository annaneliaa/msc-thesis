"""
Budget robustness for the mining comparison (HANDOFF "Deferred" item 1):
does the ranking of mining methods hold when the feature budget K changes,
and with one pooled budget instead of K/2 per class?

Variants (default): K in {10, 25, 50, 60} split per class, plus K = 60
pooled. K = 60 per class is the main comparison's setting and the reference.

Cheap by construction: the budget is applied after mining
(mining_comparison's two-stage split), so the sweep mines once into the
mining cache (stage "mine"; a no-op for everything already cached, e.g. the
AIT-ADS full run's candidates) and then only re-runs training per variant
(stage "train"). Every variant sees exactly the same mined candidates, so any
difference between variants comes from the budget alone.

Outputs under <out_root>/: one comparison run per variant (<label>/, with the
usual per_step_results.csv, summary.csv, stats.json, ...), and
  budget_summary.csv    variant x arm: mean F1, PR-AUC, features (over steps)
  budget_ranks.csv      variant x metric x arm: Friedman average rank, plus
                        the Friedman p and Nemenyi CD per variant x metric
  budget_agreement.csv  per variant x metric: Kendall tau of its average-rank
                        order with the reference's, and whether the top-ranked
                        arm is the same
  budget_summary.txt    all of the above as text
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from thesis.paths import ROOT
from thesis.system_eval.mining_comparison import (
    COMPARISON_K,
    ComparisonConfig,
    friedman_nemenyi,
    run_comparison,
    summarize_comparison,
)
from thesis.system_eval.mining_mini_experiment import _repo_commit

DEFAULT_KS = (10, 25, 50, COMPARISON_K)
METRICS = ("pr_auc", "f1")


@dataclass(frozen=True)
class BudgetVariant:
    label: str
    k: int
    per_class: bool


def budget_variants(
    ks: tuple[int, ...] = DEFAULT_KS, pooled_at: tuple[int, ...] = (COMPARISON_K,)
) -> list[BudgetVariant]:
    """Per-class variants for every K in `ks`, then pooled variants for every
    K in `pooled_at`. Labels: k<K> and k<K>_pooled."""
    out = [BudgetVariant(f"k{k}", k, True) for k in ks]
    out += [BudgetVariant(f"k{k}_pooled", k, False) for k in pooled_at]
    labels = [v.label for v in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate budget variants: {labels}")
    return out


def default_out_root(dataset: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "experiments" / "mining_budget_sweep" / dataset / ts


def variant_config(base: ComparisonConfig, v: BudgetVariant, out_dir: Path):
    return replace(
        base,
        stage="train",
        mining=replace(base.mining, k=v.k, per_class_budget=v.per_class),
        out_dir=out_dir,
    )


def run_budget_sweep(
    base: ComparisonConfig,
    variants: list[BudgetVariant],
    out_root: Path,
    mine: bool = True,
    reference: str = f"k{COMPARISON_K}",
) -> dict[str, Path]:
    """Mine once (unless mine=False: everything must already be cached), then
    train every variant from the same cached candidates, then summarise."""
    if base.mining_cache_dir is None:
        raise ValueError("the budget sweep needs a mining_cache_dir")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep.json").write_text(
        json.dumps(
            {
                "base": asdict(base),
                "variants": [asdict(v) for v in variants],
                "reference": reference,
                "repo_commit": _repo_commit(),
            },
            indent=2,
            default=str,
        )
    )
    if mine:
        print(f"== mine (cached candidates are reused) → {base.mining_cache_dir}")
        run_comparison(replace(base, stage="mine", out_dir=out_root / "_mine"))
    dirs: dict[str, Path] = {}
    for v in variants:
        print(f"== train {v.label} (K={v.k}, per_class={v.per_class})")
        dirs[v.label] = run_comparison(variant_config(base, v, out_root / v.label))
    write_budget_summary(out_root, dirs, reference)
    return dirs


# --- Summary ----------------------------------------------------------------


def summarize_budget_sweep(
    run_dirs: dict[str, Path], reference: str = f"k{COMPARISON_K}"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(summary, ranks, agreement) over the variants' per_step_results.csv."""
    summary_rows, rank_rows, agree_rows = [], [], []
    avg_ranks: dict[tuple[str, str], pd.Series] = {}
    for label, d in run_dirs.items():
        per_step = pd.read_csv(Path(d) / "per_step_results.csv")
        s = summarize_comparison(per_step).set_index("arm")
        for arm, r in s.iterrows():
            summary_rows.append(
                {
                    "variant": label,
                    "arm": arm,
                    "n_steps": int(r["n_steps"]),
                    "pr_auc": r.get("pr_auc_mean", np.nan),
                    "f1": r.get("f1_mean", np.nan),
                    "n_symbolic": r.get("n_symbolic_mean", np.nan),
                }
            )
        for metric in METRICS:
            fn = friedman_nemenyi(per_step, metric)
            ranks = pd.Series(fn.get("avg_rank", {}), dtype=float)
            avg_ranks[(label, metric)] = ranks
            for arm, rank in ranks.items():
                rank_rows.append(
                    {
                        "variant": label,
                        "metric": metric,
                        "arm": arm,
                        "avg_rank": rank,
                        "friedman_p": fn.get("friedman_p", np.nan),
                        "nemenyi_cd": fn.get("nemenyi_cd", np.nan),
                    }
                )
    for (label, metric), ranks in avg_ranks.items():
        ref = avg_ranks.get((reference, metric))
        if ref is None or ranks.empty:
            continue
        arms = sorted(set(ranks.index) & set(ref.index))
        tau = (
            sps.kendalltau(ranks[arms], ref[arms]).statistic
            if len(arms) >= 2
            else np.nan
        )
        agree_rows.append(
            {
                "variant": label,
                "metric": metric,
                "kendall_tau_vs_reference": float(tau),
                "top_arm": ranks.idxmin(),
                "reference_top_arm": ref.idxmin(),
                "same_top_arm": bool(ranks.idxmin() == ref.idxmin()),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(agree_rows),
    )


def write_budget_summary(
    out_root: Path, run_dirs: dict[str, Path], reference: str = f"k{COMPARISON_K}"
) -> None:
    summary, ranks, agreement = summarize_budget_sweep(run_dirs, reference)
    out_root = Path(out_root)
    summary.to_csv(out_root / "budget_summary.csv", index=False)
    ranks.to_csv(out_root / "budget_ranks.csv", index=False)
    agreement.to_csv(out_root / "budget_agreement.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        text = "\n\n".join(
            [
                summary.pivot(index="arm", columns="variant", values="pr_auc")
                .round(3)
                .to_string()
                if not summary.empty
                else "(no runs)",
                ranks.pivot_table(
                    index="arm", columns=["metric", "variant"], values="avg_rank"
                )
                .round(2)
                .to_string()
                if not ranks.empty
                else "",
                agreement.to_string(index=False),
            ]
        )
    (out_root / "budget_summary.txt").write_text(text + "\n")
    print(text)
