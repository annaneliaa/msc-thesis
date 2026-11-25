"""
Frozen comparison, ours vs DeepCASE on CSCAS (DeepCASE implementation plan
B4 / comparison plan step 5.4).

Inputs (both already on disk, nothing is retrained here):
* DeepCASE: a deepcase_comparison run (step 5.2) -- horizon_metrics.csv
  (its own decision metrics per seed / horizon / seen-unseen subset),
  budget.csv, and seed<k>/test_predictions.parquet (per test row: group_id,
  ts, signature_id, seen).
* Ours: run_deepcase_ours.sh outputs (B3) --
  <root>/<variant>/seed<k>/cscas/<ts>/predictions.parquet, the newest <ts>
  per (variant, seed); one row per (model, test row): label, score,
  escalated. Joined to DeepCASE's test rows on group_id (identical rows,
  verified in the smoke test), which gives both systems the same timestamps,
  signatures and seen/unseen split.

Metrics use deepcase_comparison.decision_metrics for both systems, so the
formulas are identical. For ours every row is decided automatically
(coverage 1): escalated -> analyst item. For DeepCASE an analyst item is an
auto-flagged attack OR a manual-queue event.

Two operating points for ours (plan §2.3):
* native: our fixed threshold (escalated as logged);
* matched_workload: our top-k scores escalated, k = DeepCASE's analyst items
  for the same seed and subset -- recall/precision at equal analyst work.
Plus a workload-vs-recall curve (recall at each analyst-items/day level), with
DeepCASE as one point per seed.

Windows: the test period's horizons 1..10 (granularity 0.1); "all" = the whole
test period (horizons 1..10 together), as in DeepCASE's horizon_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.system_eval.deepcase_comparison import _days, decision_metrics

SUBSETS = ("all", "seen", "unseen")
METRICS = (
    "e2e_f1",
    "e2e_precision",
    "e2e_recall",
    "missed_attacks",
    "analyst_items_per_day",
    "fp_per_day",
    "workload_reduction",
    "coverage",
    "n",
    "n_attack",
)
CURVE_ITEMS_PER_DAY = np.unique(np.concatenate([[0.0], np.geomspace(1, 50000, 240)]))


@dataclass(frozen=True, slots=True)
class OurRun:
    variant: str
    seed: int
    path: Path  # predictions.parquet


def find_our_runs(root: Path) -> list[OurRun]:
    """Newest predictions.parquet per (variant, seed) under a
    run_deepcase_ours.sh policy folder."""
    out = []
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sdir in sorted(vdir.glob("seed*")):
            files = sorted(sdir.glob("*/2*/predictions.parquet"))
            if files:
                out.append(OurRun(vdir.name, int(sdir.name[4:]), files[-1]))
    return out


def load_test_rows(deepcase_run: Path, seed: int) -> pd.DataFrame:
    tp = pd.read_parquet(
        deepcase_run / f"seed{seed}" / "test_predictions.parquet",
        columns=["group_id", "ts", "horizon", "label", "signature_id", "seen", "code"],
    )
    return tp.reset_index(drop=True)


def _subset_masks(seen: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(seen), dtype=bool), "seen": seen, "unseen": ~seen}


def _ours_metrics(label: np.ndarray, escalated: np.ndarray, days: float) -> dict:
    codes = escalated.astype(int)  # 1 = auto attack, 0 = auto benign
    return decision_metrics(label, codes, days)


def matched_workload_escalation(score: np.ndarray, k: int) -> np.ndarray:
    """Escalate the k highest scores (stable order breaks ties)."""
    esc = np.zeros(len(score), dtype=bool)
    if k > 0:
        esc[np.argsort(-score, kind="stable")[: min(k, len(score))]] = True
    return esc


def workload_curve(
    label: np.ndarray, score: np.ndarray, days: float, items_per_day=CURVE_ITEMS_PER_DAY
) -> pd.DataFrame:
    """Recall and precision when escalating the top items_per_day * days
    scores, for each workload level."""
    order = np.argsort(-score, kind="stable")
    cum_tp = np.concatenate([[0], np.cumsum(label[order].astype(int))])
    n_pos = max(int(label.sum()), 1)
    k = np.clip(np.round(np.asarray(items_per_day) * days).astype(int), 0, len(score))
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(k > 0, cum_tp[k] / np.maximum(k, 1), np.nan)
    return pd.DataFrame(
        {
            "items_per_day": items_per_day,
            "k": k,
            "recall": cum_tp[k] / n_pos,
            "precision": precision,
        }
    )


def compare_seed(
    ours: pd.DataFrame, rows: pd.DataFrame, deepcase_items: dict[str, int]
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    """Metrics for one (variant, seed) of ours against DeepCASE's test rows.
    `ours` = one model's predictions; `deepcase_items` = DeepCASE's analyst
    items per subset for the same seed. Returns (overall rows, curves,
    per-horizon rows)."""
    m = rows.merge(
        ours[["group_id", "score", "escalated", "label"]].rename(
            columns={"label": "label_ours"}
        ),
        on="group_id",
        how="left",
        validate="one_to_one",
    )
    if m["score"].isna().any():
        raise ValueError(f"{int(m['score'].isna().sum())} test rows missing from ours")
    if not (m["label"].to_numpy() == m["label_ours"].to_numpy()).all():
        raise ValueError("labels disagree between ours and DeepCASE")
    label = m["label"].to_numpy().astype(bool)
    score = m["score"].to_numpy()
    escalated = m["escalated"].to_numpy().astype(bool)
    ts = m["ts"].to_numpy()
    seen = m["seen"].to_numpy().astype(bool)
    days = _days(ts)

    overall, curves = [], []
    for subset, mask in _subset_masks(seen).items():
        nat = _ours_metrics(label[mask], escalated[mask], days)
        overall.append({"subset": subset, "operating_point": "native", **nat})
        k = deepcase_items.get(subset, 0)
        esc_k = matched_workload_escalation(score[mask], k)
        mw = _ours_metrics(label[mask], esc_k, days)
        overall.append({"subset": subset, "operating_point": "matched_workload", **mw})
        c = workload_curve(label[mask], score[mask], days)
        c.insert(0, "subset", subset)
        curves.append(c)

    per_horizon = []
    for h, idx in m.groupby("horizon").indices.items():
        hd = _days(ts[idx])
        for subset, mask in _subset_masks(seen[idx]).items():
            r = _ours_metrics(label[idx][mask], escalated[idx][mask], hd)
            per_horizon.append({"horizon": int(h), "subset": subset, **r})
    return overall, curves, per_horizon


def run_frozen_comparison(
    deepcase_run: Path, ours_root: Path, out_dir: Path
) -> dict[str, pd.DataFrame]:
    dc_metrics = pd.read_csv(deepcase_run / "horizon_metrics.csv")
    dc_all = dc_metrics[dc_metrics.horizon.astype(str) == "all"]
    budget = pd.read_csv(deepcase_run / "budget.csv").set_index("seed")

    per_seed, curves, per_horizon = [], [], []
    for _, r in dc_all.iterrows():
        per_seed.append(
            {
                "system": "deepcase",
                "variant": "deepcase",
                "model": "deepcase",
                "seed": int(r.seed),
                "subset": r.subset,
                "operating_point": "native",
                **{k: r[k] for k in METRICS},
                "budget_b0": int(budget.loc[int(r.seed), "budget_b0"]),
            }
        )
    for _, r in dc_metrics[dc_metrics.horizon.astype(str) != "all"].iterrows():
        per_horizon.append(
            {
                "system": "deepcase",
                "variant": "deepcase",
                "model": "deepcase",
                "seed": int(r.seed),
                "horizon": int(r.horizon),
                "subset": r.subset,
                **{k: r[k] for k in METRICS},
            }
        )

    rows_cache: dict[int, pd.DataFrame] = {}
    for run in find_our_runs(ours_root):
        if run.seed not in rows_cache:
            rows_cache[run.seed] = load_test_rows(deepcase_run, run.seed)
        rows = rows_cache[run.seed]
        dc_items = (
            dc_all[dc_all.seed == run.seed]
            .set_index("subset")["analyst_items"]
            .to_dict()
        )
        preds = pd.read_parquet(run.path)
        for model, p in preds[preds.horizon_window_index > 0].groupby("model"):
            overall, cv, ph = compare_seed(p, rows, dc_items)
            base = {
                "system": "ours",
                "variant": run.variant,
                "model": model,
                "seed": run.seed,
            }
            for o in overall:
                per_seed.append(
                    {
                        **base,
                        **{k: o[k] for k in ("subset", "operating_point")},
                        **{k: o[k] for k in METRICS},
                        "budget_b0": None,
                    }
                )
            for c in cv:
                c = c.copy()
                for k, v in base.items():
                    c[k] = v
                curves.append(c)
            for h in ph:
                per_horizon.append(
                    {
                        **base,
                        **{k: h[k] for k in ("horizon", "subset")},
                        **{k: h[k] for k in METRICS},
                    }
                )

    per_seed_df = pd.DataFrame(per_seed)
    keys = ["system", "variant", "model", "subset", "operating_point"]
    agg = per_seed_df.groupby(keys)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    table = agg.reset_index()
    table.insert(len(keys), "n_seeds", per_seed_df.groupby(keys).size().to_numpy())

    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    per_horizon_df = pd.DataFrame(per_horizon)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(out_dir / "per_seed.csv", index=False)
    table.to_csv(out_dir / "table.csv", index=False)
    curves_df.to_csv(out_dir / "workload_curve.csv", index=False)
    per_horizon_df.to_csv(out_dir / "per_horizon.csv", index=False)
    return {
        "per_seed": per_seed_df,
        "table": table,
        "curves": curves_df,
        "per_horizon": per_horizon_df,
    }


def format_table(table: pd.DataFrame) -> str:
    """Compact mean ± std view of the main metrics."""
    cols = [
        "e2e_f1",
        "e2e_recall",
        "e2e_precision",
        "analyst_items_per_day",
        "fp_per_day",
    ]
    view = table[
        ["system", "variant", "model", "subset", "operating_point", "n_seeds"]
    ].copy()
    for c in cols:
        prec = 0 if c.endswith("per_day") else 3
        view[c] = [
            f"{m:.{prec}f} ± {s:.{prec}f}" if pd.notna(s) else f"{m:.{prec}f}"
            for m, s in zip(table[f"{c}_mean"], table[f"{c}_std"])
        ]
    return view.to_string(index=False)


def plot_workload_curve(
    curves: pd.DataFrame, per_seed: pd.DataFrame, path: Path, subset: str = "all"
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    c = curves[curves.subset == subset]
    for (variant, model), g in c.groupby(["variant", "model"]):
        mean = g[g.items_per_day > 0].groupby("items_per_day")["recall"].mean()
        ax.plot(mean.index, mean.values, label=f"ours {model} / {variant}", lw=1.4)
    dc = per_seed[(per_seed.system == "deepcase") & (per_seed.subset == subset)]
    ax.scatter(
        dc["analyst_items_per_day"],
        dc["e2e_recall"],
        color="black",
        marker="x",
        zorder=5,
        label="DeepCASE (per seed)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("analyst items per day (log)")
    ax.set_ylabel("recall")
    ax.set_title(f"Workload vs recall, frozen, test period ({subset} signatures)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
