"""
Our monitor's signals on frozen DeepCASE, observe-only (DeepCASE
implementation plan B5 / comparison plan step 5.5).

Question: does the same monitor (the combined trigger) make sense for a
different model? DeepCASE runs frozen over the test windows (step 5.2); per
window we compute the combined trigger's signals on DeepCASE's decisions and
log them without acting. Only the labels an analyst would have are used:

* analyst items = what DeepCASE escalates: events auto-labelled attack
  (code > 0) plus its manual queue (code < 0). Reviewed anyway, weight 1;
* audit = a random `audit_rate` sample of the events DeepCASE auto-labels
  benign (code == 0), weight 1/audit_rate -- the same hash draw on group_id
  as our system's audit (pattern_validity.audit_mask_ids), so both systems
  audit the same CSCAS rows;
* "oracle" mode uses every true label (upper bound).

Signal mapping (ours -> DeepCASE):

| ours                     | DeepCASE                                                  |
|--------------------------|-----------------------------------------------------------|
| feedback precision       | attack share among analyst items (escalated + manual)     |
| pattern coverage         | attack coverage: weighted share of attacks DeepCASE       |
|                          | auto-labels attack; benign coverage likewise              |
| pattern precision        | per-cluster precision: weighted share of a cluster's      |
|                          | automatic decisions that are correct                      |
| (label-free, logged)     | unmatched rate = 1 - coverage (DeepCASE's own "coverage") |

Reference = window 1, the first window the frozen model hasn't seen (the
warm-up our system also uses when trained on all 6 days). Tests are the same
as ours: feedback precision's one-sided 99% two-proportion test; coverage
and cluster precision via pattern_validity._drop_significant (99%, and at
least 0.05 lower, Kish effective n). Decision (logged): pattern-validity
analogue = cluster precision OR coverage drop -> REMINE_AND_RETRAIN; else
feedback precision -> RETRAIN_ONLY; else NO_ACTION.

Per-cluster precision needs the matched training cluster per test row
(`cluster` column, written by deepcase_comparison since 2026-11-25); runs
without it skip that part.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis.monitor.feedback import FeedbackCounts, feedback_signal
from thesis.monitor.pattern_validity import (
    AUDIT_RATE,
    _drop_significant,
    _n_eff,
    audit_mask_ids,
)

MIN_SAMPLES = 30
LABEL_MODES = ("oracle", "audit")


def label_weights(
    code: np.ndarray, audited: np.ndarray, mode: str, audit_rate: float
) -> np.ndarray:
    """Weight per row of the labels the system has in `mode`."""
    if mode == "oracle":
        return np.ones(len(code))
    analyst = code != 0  # auto-attack or manual queue
    return np.where(analyst, 1.0, np.where(audited, 1.0 / audit_rate, 0.0))


def window_stats(
    label: np.ndarray, code: np.ndarray, cluster: np.ndarray | None, w: np.ndarray
) -> dict:
    """Weighted per-window statistics for one label mode."""
    att = label.astype(bool)
    auto_attack = code > 0
    auto_benign = code == 0
    wa, wb = w * att, w * ~att
    A, B = wa.sum(), wb.sum()
    out = {
        "cov_attack": float((wa * auto_attack).sum() / A) if A > 0 else np.nan,
        "cov_benign": float((wb * auto_benign).sum() / B) if B > 0 else np.nan,
        "n_eff_attack": _n_eff(wa),
        "n_eff_benign": _n_eff(wb),
        "cluster_prec": {},
    }
    if cluster is not None:
        auto = (code >= 0) & (cluster >= 0)
        correct = (label == code).astype(float)
        df = pd.DataFrame(
            {"cluster": cluster[auto], "w": w[auto], "wc": (w * correct)[auto]}
        )
        df = df[df.w > 0]
        if len(df):
            g = df.groupby("cluster")
            sums = g[["w", "wc"]].sum()
            n_eff = g["w"].apply(lambda x: _n_eff(x.to_numpy()))
            out["cluster_prec"] = {
                int(c): (float(r.wc / r.w), float(n_eff[c])) for c, r in sums.iterrows()
            }
    return out


def compare_windows(win: dict, ref: dict, min_samples: int = MIN_SAMPLES) -> dict:
    cov = _drop_significant(
        win["cov_attack"],
        win["n_eff_attack"],
        ref["cov_attack"],
        ref["n_eff_attack"],
        min_samples,
    ) or _drop_significant(
        win["cov_benign"],
        win["n_eff_benign"],
        ref["cov_benign"],
        ref["n_eff_benign"],
        min_samples,
    )
    drops, worst = [], np.nan
    for c, (p_ref, n_ref) in ref["cluster_prec"].items():
        if c not in win["cluster_prec"]:
            continue
        p_win, n_win = win["cluster_prec"][c]
        d = p_ref - p_win
        worst = d if np.isnan(worst) else max(worst, d)
        if _drop_significant(p_win, n_win, p_ref, n_ref, min_samples):
            drops.append(c)
    return {
        "cov_attack": win["cov_attack"],
        "cov_attack_ref": ref["cov_attack"],
        "cov_benign": win["cov_benign"],
        "cov_benign_ref": ref["cov_benign"],
        "n_eff_attack": win["n_eff_attack"],
        "n_eff_benign": win["n_eff_benign"],
        "n_clusters_checked": sum(
            1
            for c, (_, n) in ref["cluster_prec"].items()
            if c in win["cluster_prec"]
            and n >= min_samples
            and win["cluster_prec"][c][1] >= min_samples
        ),
        "n_cluster_prec_drop": len(drops),
        "cluster_prec_drop_ids": ";".join(map(str, drops)),
        "worst_cluster_prec_drop": worst,
        "coverage_elevated": bool(cov),
        "precision_elevated": bool(drops),
        "elevated": bool(cov or drops),
    }


def _feedback(label: np.ndarray, flagged: np.ndarray) -> FeedbackCounts:
    return FeedbackCounts(int(flagged.sum()), int((flagged & label.astype(bool)).sum()))


def monitor_seed(
    rows: pd.DataFrame,
    audit_rate: float = AUDIT_RATE,
    min_samples: int = MIN_SAMPLES,
    ref_horizon: int = 1,
) -> pd.DataFrame:
    """Per-window signals for one seed's test_predictions.parquet."""
    has_cluster = "cluster" in rows.columns
    audited_all = audit_mask_ids(rows["group_id"].astype(str), audit_rate) > 0
    out, ref_stats, ref_fb, ref_fb_auto = [], {}, None, None
    for h, idx in sorted(rows.groupby("horizon").indices.items()):
        r = rows.iloc[idx]
        label = r["label"].to_numpy().astype(int)
        code = r["code"].to_numpy()
        cluster = r["cluster"].to_numpy() if has_cluster else None
        audited = audited_all[idx] & (code == 0)
        analyst = code != 0
        row = {
            "horizon": int(h),
            "is_reference": int(h) == ref_horizon,
            "has_cluster_ids": has_cluster,
            "n": len(r),
            "n_attack": int(label.sum()),
            "unmatched_rate": float((code < 0).mean()),
            "n_analyst_items": int(analyst.sum()),
            "n_audit_labels": int(audited.sum()),
        }
        fb = _feedback(label, analyst)
        fb_auto = _feedback(label, code > 0)
        if int(h) == ref_horizon:
            ref_fb, ref_fb_auto = fb, fb_auto
        row.update(feedback_signal(fb, ref_fb))
        row.update(
            {f"auto_{k}": v for k, v in feedback_signal(fb_auto, ref_fb_auto).items()}
        )
        for mode in LABEL_MODES:
            w = label_weights(code, audited, mode, audit_rate)
            st = window_stats(label, code, cluster, w)
            if int(h) == ref_horizon:
                ref_stats[mode] = st
            if mode in ref_stats:
                cmp = compare_windows(st, ref_stats[mode], min_samples)
                row.update({f"pv_{mode}_{k}": v for k, v in cmp.items()})
        pv = bool(row.get("pv_audit_elevated"))
        fbe = bool(row.get("fb_precision_elevated"))
        row["combined_action"] = (
            "REMINE_AND_RETRAIN" if pv else "RETRAIN_ONLY" if fbe else "NO_ACTION"
        )
        out.append(row)
    return pd.DataFrame(out)


def run_deepcase_monitor(
    deepcase_run: Path,
    out_dir: Path,
    audit_rate: float = AUDIT_RATE,
    min_samples: int = MIN_SAMPLES,
) -> pd.DataFrame:
    """Signals for every seed of a deepcase_comparison run, joined with
    DeepCASE's own per-window metrics (horizon_metrics.csv, subset all)."""
    metrics = pd.read_csv(deepcase_run / "horizon_metrics.csv")
    metrics = metrics[
        (metrics.subset == "all") & (metrics.horizon.astype(str) != "all")
    ].assign(horizon=lambda d: d.horizon.astype(int))
    keep = ["seed", "horizon", "e2e_f1", "e2e_precision", "e2e_recall", "coverage"]
    frames = []
    for seed_dir in sorted(deepcase_run.glob("seed*")):
        seed = int(seed_dir.name[4:])
        rows = pd.read_parquet(seed_dir / "test_predictions.parquet")
        sig = monitor_seed(rows, audit_rate, min_samples)
        sig.insert(0, "seed", seed)
        frames.append(sig)
    signals = pd.concat(frames, ignore_index=True).merge(
        metrics[keep], on=["seed", "horizon"], how="left"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_dir / "per_window_signals.csv", index=False)
    return signals


def summarise(signals: pd.DataFrame) -> str:
    """Firing rates per signal (windows after the reference) and how each
    signal's value tracks DeepCASE's per-window F1."""
    walk = signals[~signals.is_reference]
    flags = {
        "feedback precision (escalated + manual)": "fb_precision_elevated",
        "feedback precision (auto-attack only)": "auto_fb_precision_elevated",
        "coverage, audit": "pv_audit_coverage_elevated",
        "cluster precision, audit": "pv_audit_precision_elevated",
        "pattern validity analogue, audit": "pv_audit_elevated",
        "coverage, oracle": "pv_oracle_coverage_elevated",
        "cluster precision, oracle": "pv_oracle_precision_elevated",
        "pattern validity analogue, oracle": "pv_oracle_elevated",
    }
    lines = ["Firing rate over windows 2..N (mean over seeds):"]
    for name, col in flags.items():
        if col in walk:
            lines.append(f"  {name:42s} {walk[col].astype(float).mean():.2f}")
    lines.append("")
    lines.append("Spearman correlation with DeepCASE's per-window e2e F1:")
    for col in (
        "unmatched_rate",
        "fb_precision",
        "pv_audit_cov_attack",
        "pv_audit_n_cluster_prec_drop",
        "pv_oracle_cov_attack",
    ):
        if col in walk and walk[col].notna().sum() > 2:
            rho = walk[[col, "e2e_f1"]].astype(float).corr(method="spearman").iloc[0, 1]
            lines.append(f"  {col:32s} {rho:+.2f}")
    lines.append("")
    lines.append("Combined decision per window (counts over seeds):")
    lines.append(walk["combined_action"].value_counts().to_string())
    return "\n".join(lines)
