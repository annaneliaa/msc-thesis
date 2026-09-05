"""
Non-plotting EDA logic (data loading, computation, per-scenario analysis,
orchestration) for the dataset-inspection notebooks:
    src/thesis/notebooks/inspect_ait_ads_dataset.ipynb
    src/thesis/notebooks/inspect_cscas_dataset.ipynb

Plotting lives in thesis.visualization.eda; this module computes the data
those plot functions need and writes the CSV/text-summary side outputs.
Formerly two standalone scripts (run_eda.py, run_eda_host.py); folded in here
so both notebooks can call the same functions instead of duplicating logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.alert_groups import build_labeled_window_alert_groups

from thesis.visualization.eda import (
    classify_signatures,
    plot_occurrence_burst_raster,
    plot_signature_activity_bins,
    plot_signature_vocabulary_churn,
    plot_signature_activity_heatmap,
    plot_attack_temporal_concentration,
)

from thesis.visualization.eda import (
    plot_alert_volume_concatenated,
    plot_attack_phase_zoom,
    plot_attack_type_heatmap,
    plot_signature_event_raster,
    plot_temporal_attack_overview,
)

# Gap threshold used by AlertBERT time-delta chaining (seconds); also the
# default stream/burst-splitting threshold for the per-host analysis.
ALERTBERT_DELTA = 2.0

_ZOOM_PARAMS: dict[str, dict] = {
    "ait-ads": dict(context_hours=0.5, phase_gap_hours=3.0, bin_hours=0.01),
    "cscas": dict(context_hours=6.0, phase_gap_hours=48.0, bin_hours=1.0),
}


def data_label(balanced: str | None = None, groups_balanced: str | None = None) -> str:
    if groups_balanced is not None:
        return f"data: raw alerts, groups balanced ({groups_balanced})"
    if balanced is None:
        return "data: raw"
    return f"data: balanced ({balanced})"


def groups_dir(
    alert_groups_base_dir: Path,
    balanced: str | None = None,
    groups_balanced: str | None = None,
) -> Path:
    if groups_balanced is not None:
        return alert_groups_base_dir / "balanced" / groups_balanced
    if balanced is None:
        return alert_groups_base_dir / "raw"
    return alert_groups_base_dir / "from_balanced_alerts" / balanced


def annotate_and_save(fig, out_path: Path, label: str) -> None:
    """Stamp the bottom-right `data:` provenance label on a figure and save
    it. Plot functions in visualization.eda don't do this themselves (the
    label is run-specific, not intrinsic to the plot), so the caller adds it
    before persisting."""
    fig.text(
        0.99,
        0.01,
        label,
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def compute_uniq_signature_counts(df: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    counts = df[sig_col].value_counts().reset_index()
    counts.columns = ["signature", "count"]
    return counts


# ---------------------------------------------------------------------------
# Phase 1: per-scenario analysis
# ---------------------------------------------------------------------------


def run_scenario_eda(
    df: pd.DataFrame,
    scenario: str,
    out_path: Path,
    summary_path: Path,
) -> None:
    """Basic per-scenario stats (row/column counts, missing values, time
    label distribution, timestamp range, unique-signature counts). Writes a
    text summary to summary_path/<scenario>_eda_summary.txt. Alert_group
    construction lives in build_scenario_alert_groups instead -- this
    function no longer builds or caches alert_groups (see git history for
    the pair-frequency/alert_group-size analysis that used to live here,
    removed as unused: nothing downstream read that part of the text
    summary back in)."""

    print(f"  Running EDA for scenario '{scenario}' (n={len(df):,} rows)...")
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)

    with open(summary_path / f"{scenario}_eda_summary.txt", "w") as f:
        f.write(f"Exploratory Data Analysis for {scenario} scenario\n")
        f.write("=" * 50 + "\n\n")

        f.write("Basic Information:\n")
        f.write(f"Number of rows: {len(df)}\n")
        f.write(f"Number of columns: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")

        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string() + "\n\n")

        if "label" in df.columns:
            f.write("Time label Distribution:\n")
            f.write(df["label"].value_counts().to_string() + "\n\n")

        if "time" in df.columns:
            f.write(f"Min timestamp: {df['time'].min()}\n")
            f.write(f"Max timestamp: {df['time'].max()}\n\n")

        sig_counts = compute_uniq_signature_counts(df, sig_col="signature")
        f.write(f"Unique signature counts in data: {len(sig_counts)}\n")
        f.write("Top 30 most common signatures:\n")
        f.write(sig_counts.head(30).to_string(index=False) + "\n\n")

    print(f"  {scenario}: done → {out_path}")


def build_scenario_alert_groups(
    df: pd.DataFrame,
    scenario: str,
    alert_groups_base_dir: Path,
    balanced: str | None = None,
    groups_balanced: str | None = None,
    window_size_s: int = 2,
    force: bool = False,
) -> None:
    """
    Build the fixed `window_size_s`-window alert_groups for one scenario and
    cache them to <alert_groups_base_dir>/.../<scenario>_alert_groups.csv
    (path from groups_dir) -- the cache Phase 2's alert_group volume plots
    read via visualization.eda.load_alert_groups. Skips rebuilding if a
    cached CSV already exists, unless force=True.
    """
    group_cache_dir = groups_dir(alert_groups_base_dir, balanced, groups_balanced)
    cache_path = group_cache_dir / f"{scenario}_alert_groups.csv"

    if cache_path.exists() and not force:
        print(f"  {scenario}: alert_groups cache exists → {cache_path} (skipping)")
        return

    print(f"  Building alert_groups for scenario '{scenario}' (n={len(df):,} rows)...")
    alert_groups = build_labeled_window_alert_groups(df, window_size_s=window_size_s)
    group_cache_dir.mkdir(parents=True, exist_ok=True)
    alert_groups.to_csv(cache_path, index=False)
    print(f"  {scenario}: alert_groups cached → {cache_path}")


# ---------------------------------------------------------------------------
# Signature-behaviour / temporal-stability analysis (CSCAS notebook parity)
# ---------------------------------------------------------------------------


def compute_signature_behaviour_summary(
    df: pd.DataFrame, sig_col: str = "signature"
) -> pd.DataFrame:
    """
    One table with per-class (All / Benign-only / Attack-only) signature
    stats: volume concentration (top-K share, signatures needed for 90% of
    volume), persistence (days active), and first-appearance timing.
    Mirrors the CSCAS notebook's signature-behaviour-summary table (there
    keyed on alert *groups*; here on raw alerts) so both write-ups can quote
    the same shape of table.
    """
    sig_cls = classify_signatures(df, sig_col)
    total_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    day = df["timestamp"].dt.normalize()
    per_sig = df.groupby(sig_col).agg(n_alerts=("is_attack", "size"))
    per_sig["days_active"] = day.groupby(df[sig_col]).nunique()
    per_sig["cls"] = sig_cls
    per_sig["alerts_per_day"] = per_sig["n_alerts"] / total_days
    per_sig["days_active_frac"] = per_sig["days_active"] / total_days

    fs = df.groupby(sig_col)["timestamp"].min()
    t0 = fs.min()
    t1 = fs.max()
    day1 = fs <= t0 + pd.Timedelta(days=1)
    last25 = fs > t0 + 0.75 * (t1 - t0)

    def _n_for_90(counts):
        c = counts.sort_values(ascending=False).cumsum() / counts.sum()
        return int((c < 0.9).sum() + 1)

    def _column(which):
        ids = per_sig.index if which == "all" else per_sig.index[per_sig.cls == which]
        sub = per_sig.loc[ids]
        v = df[df[sig_col].isin(ids)][sig_col].value_counts()
        return {
            "Distinct signatures": len(sub),
            "Signatures carrying 90% of their alerts": _n_for_90(v),
            "Median alerts per signature": round(sub.n_alerts.median(), 1),
            "Signatures with < 10 alerts": int((sub.n_alerts < 10).sum()),
            "Median alerts/day per signature": round(sub.alerts_per_day.median(), 2),
            "Median days-active fraction": round(sub.days_active_frac.median(), 2),
            "Signatures active every day": int((sub.days_active_frac == 1.0).sum()),
            "First appearance on day 1": int(day1.reindex(ids).sum()),
            "First appearance in final 25%": int(last25.reindex(ids).sum()),
        }

    vc = df[sig_col].value_counts()
    cum = vc.sort_values(ascending=False).cumsum() / vc.sum()

    def _top_share(k):
        return round(100 * cum.iloc[min(k - 1, len(cum) - 1)], 1)

    sig_summary = pd.DataFrame(
        {
            "All": _column("all"),
            "Benign-only": _column("benign"),
            "Attack-only": _column("attack"),
        }
    )
    overall = pd.DataFrame(
        {
            "All": {
                "Mixed-label signatures": int((sig_cls == "mixed").sum()),
                "Top 5 signatures, share of all alerts (%)": _top_share(5),
                "Top 10 signatures, share of all alerts (%)": _top_share(10),
                "Top 50 signatures, share of all alerts (%)": _top_share(50),
            }
        }
    ).reindex(columns=sig_summary.columns)
    return pd.concat([sig_summary.iloc[[0]], overall, sig_summary.iloc[1:]])


def compute_signature_overview_table(
    df: pd.DataFrame, sig_col: str = "signature"
) -> pd.DataFrame:
    """
    Per-signature stats table for the write-up's overview CSVs: volume,
    class, and activity span. Mirrors CSCAS's signature_overview.csv, minus
    the CVE-reference/category columns (parsed from Suricata's SignatureText
    convention, which has no AIT-ADS equivalent).
    """
    stats = (
        df.groupby(sig_col)
        .agg(count=("is_attack", "size"), attack_count=("is_attack", "sum"))
        .reset_index()
    )
    stats["benign_count"] = stats["count"] - stats["attack_count"]
    stats["attack_rate%"] = (stats["attack_count"] / stats["count"] * 100).round(3)
    stats["sig_class"] = stats[sig_col].map(classify_signatures(df, sig_col))

    span = (
        df.groupby(sig_col)
        .agg(
            n_days_active=("timestamp", lambda s: s.dt.normalize().nunique()),
            n_hours_active=("timestamp", lambda s: s.dt.floor("h").nunique()),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    return stats.merge(span, on=sig_col).sort_values("count", ascending=False)


def compute_signature_activity_by_bin(
    df: pd.DataFrame, sig_col: str = "signature", freq: str = "D"
) -> pd.DataFrame:
    """For each time bin: how many distinct signatures were active, split by
    class, and what fraction of that bin's active signatures each class
    represents. Mirrors CSCAS's signature_activity_per_day/hour.csv."""
    sig_cls_map = classify_signatures(df, sig_col)
    binned = df[["timestamp", sig_col]].copy()
    binned["_cls"] = binned[sig_col].map(sig_cls_map)
    binned["bin"] = binned["timestamp"].dt.floor(freq)
    active = binned.groupby(["bin", sig_col])["_cls"].first().reset_index()
    counts = (
        active.groupby(["bin", "_cls"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["benign", "attack", "mixed"], fill_value=0)
    )
    counts["n_active_signatures"] = counts.sum(axis=1)
    counts["frac_benign"] = counts["benign"] / counts["n_active_signatures"]
    counts["frac_attack"] = counts["attack"] / counts["n_active_signatures"]
    counts["frac_mixed"] = counts["mixed"] / counts["n_active_signatures"]
    return counts.reset_index().rename(columns={"bin": "timestamp"})


def compute_signature_activity_summary(
    activity_by_freq: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Min/max/mean rollup of compute_signature_activity_by_bin's output
    across granularities, e.g. {'day': daily_df, 'hour': hourly_df}. Mirrors
    CSCAS's signature_activity_summary.csv."""
    cols = [
        "benign",
        "attack",
        "mixed",
        "n_active_signatures",
        "frac_benign",
        "frac_attack",
        "frac_mixed",
    ]
    rows = [
        {
            "granularity": granularity,
            "metric": col,
            "min": counts_df[col].min(),
            "max": counts_df[col].max(),
            "mean": counts_df[col].mean(),
        }
        for granularity, counts_df in activity_by_freq.items()
        for col in cols
    ]
    out = pd.DataFrame(rows)
    out[["min", "max", "mean"]] = out[["min", "max", "mean"]].round(4)
    return out


def run_signature_behaviour_eda(
    df: pd.DataFrame,
    scenario: str,
    out_path: Path,
    summary_path: Path,
    overview_dir: Path,
    sig_col: str = "signature",
    bin_freq: str = "1h",
    balanced: str | None = None,
    groups_balanced: str | None = None,
) -> None:
    """
    Signature-behaviour + temporal-stability analysis for one scenario,
    mirroring the CSCAS notebook's signature-vocabulary-churn /
    signature-behaviour-summary / overview-CSV sections so both datasets'
    write-ups can quote comparable plots and tables. Writes:
      - <scenario>_signature_behaviour_summary.txt (table + LaTeX) to
        summary_path
      - <scenario>_signature_activity_bins.png, _vocabulary_churn.png,
        _signature_activity_heatmap.png, _attack_temporal_concentration.png
        to out_path
      - <scenario>_signature_overview.csv,
        _signature_activity_per_day.csv, _signature_activity_per_hour.csv,
        _signature_activity_summary.csv to overview_dir
    """
    print(f"  Running signature-behaviour EDA for scenario '{scenario}'...")
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    label = data_label(balanced, groups_balanced)

    sig_summary = compute_signature_behaviour_summary(df, sig_col=sig_col)
    with open(summary_path / f"{scenario}_signature_behaviour_summary.txt", "w") as f:
        f.write(f"Signature-behaviour summary — {scenario}\n")
        f.write("=" * 50 + "\n\n")
        f.write(sig_summary.to_string(na_rep="--") + "\n\n")
        f.write(
            sig_summary.to_latex(
                na_rep="--",
                caption=f"{scenario}: signature-behaviour summary.",
                label=f"tab:{scenario}-signature-behaviour",
            )
        )

    fig, _ = plot_signature_activity_bins(
        df, scenario, sig_col=sig_col, bin_freq=bin_freq
    )
    annotate_and_save(fig, out_path / f"{scenario}_signature_activity_bins.png", label)

    fig, _ = plot_signature_vocabulary_churn(
        df, scenario, sig_col=sig_col, bin_freq=bin_freq
    )
    annotate_and_save(fig, out_path / f"{scenario}_vocabulary_churn.png", label)

    fig, _ = plot_signature_activity_heatmap(
        df, scenario, group_cols=(sig_col, "short")
    )
    annotate_and_save(
        fig, out_path / f"{scenario}_signature_activity_heatmap.png", label
    )

    fig, _ = plot_attack_temporal_concentration(df, scenario, sig_col=sig_col)
    annotate_and_save(
        fig, out_path / f"{scenario}_attack_temporal_concentration.png", label
    )

    sig_overview = compute_signature_overview_table(df, sig_col=sig_col)
    sig_overview.to_csv(
        overview_dir / f"{scenario}_signature_overview.csv", index=False
    )

    daily = compute_signature_activity_by_bin(df, sig_col=sig_col, freq="D")
    hourly = compute_signature_activity_by_bin(df, sig_col=sig_col, freq="h")
    daily.to_csv(
        overview_dir / f"{scenario}_signature_activity_per_day.csv", index=False
    )
    hourly.to_csv(
        overview_dir / f"{scenario}_signature_activity_per_hour.csv", index=False
    )

    activity_summary = compute_signature_activity_summary(
        {"day": daily, "hour": hourly}
    )
    activity_summary.to_csv(
        overview_dir / f"{scenario}_signature_activity_summary.csv", index=False
    )

    print(f"  {scenario}: signature-behaviour EDA done → {out_path}")


# ---------------------------------------------------------------------------
# Cross-scenario tables
# ---------------------------------------------------------------------------


def compute_overview_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario stats: duration, alert counts, attack %, signature/type counts."""
    scenarios = sorted(all_df["scenario"].unique())
    rows = []
    for sc in scenarios:
        sc_df = all_df[all_df["scenario"] == sc]
        n_total = len(sc_df)
        n_benign = (~sc_df["is_attack"]).sum()
        n_attack = sc_df["is_attack"].sum()
        t_start = pd.to_datetime(sc_df["time"].min(), unit="s", utc=True)
        t_end = pd.to_datetime(sc_df["time"].max(), unit="s", utc=True)
        duration_days = (sc_df["time"].max() - sc_df["time"].min()) / 86400
        n_sig = sc_df["signature"].nunique()
        n_attack_types = sc_df.loc[sc_df["is_attack"], "label"].nunique()
        rows.append(
            {
                "scenario": sc,
                "start": t_start.strftime("%Y-%m-%d"),
                "end": t_end.strftime("%Y-%m-%d"),
                "duration_days": round(duration_days, 1),
                "total_alerts": n_total,
                "benign_alerts": int(n_benign),
                "attack_alerts": int(n_attack),
                "attack_pct": round(100 * n_attack / n_total, 1),
                "unique_signatures": n_sig,
                "attack_types": n_attack_types,
            }
        )
    return pd.DataFrame(rows)


def compute_ait_eda_summary_table(
    all_df: pd.DataFrame,
    scenarios: list[str],
    alert_groups_base_dir: Path,
    sig_col: str = "signature",
) -> pd.DataFrame:
    """
    Headline per-scenario overview table for the AIT-ADS write-up: duration,
    raw-alert counts + attack %, fixed-2s-window alert_group counts +
    attack % (ensures each scenario's alert_groups cache exists via
    build_scenario_alert_groups, building it if missing), signature counts
    (total + benign-only/attack-only/mixed split, see classify_signatures),
    and how many distinct attack labels ("attack stages") occur. One row
    per scenario, in the given `scenarios` order -- the numbers behind
    Table~\\ref{tab:ait-eda-summary} in the write-up.
    """
    overview = compute_overview_table(all_df).set_index("scenario")

    rows = []
    for sc in scenarios:
        sub = all_df[all_df["scenario"] == sc]
        row = overview.loc[sc]

        cls = classify_signatures(sub, sig_col)
        n_benign_sig = int((cls == "benign").sum())
        n_attack_sig = int((cls == "attack").sum())
        n_mixed_sig = int((cls == "mixed").sum())

        build_scenario_alert_groups(
            sub, sc, alert_groups_base_dir=alert_groups_base_dir
        )
        cache_path = groups_dir(alert_groups_base_dir) / f"{sc}_alert_groups.csv"
        group_labels = pd.read_csv(cache_path, usecols=["group_label"])["group_label"]
        n_groups = len(group_labels)
        n_groups_attack = int((group_labels == "attack").sum())

        rows.append(
            {
                "scenario": sc,
                "duration_days": int(round(row["duration_days"])),
                "total_alerts": int(row["total_alerts"]),
                "benign_alerts": int(row["benign_alerts"]),
                "attack_alerts": int(row["attack_alerts"]),
                "attack_pct": row["attack_pct"],
                "n_groups_2s": n_groups,
                "attack_pct_groups": round(100 * n_groups_attack / n_groups, 1)
                if n_groups
                else None,
                "n_signatures": int(row["unique_signatures"]),
                "signatures_benign_attack_mixed": f"{n_benign_sig} / {n_attack_sig} / {n_mixed_sig}",
                "n_attack_stages": int(row["attack_types"]),
            }
        )
    return pd.DataFrame(rows)


def compute_label_distribution_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy per-scenario/per-label breakdown: count, % of scenario's total rows,
    % of scenario's attack rows. Attack labels are listed before
    'false_positive' within each scenario -- feed straight to
    visualization.eda.plot_label_distribution_table for the table figure.
    """
    scenarios = sorted(all_df["scenario"].unique())
    rows = []
    for sc in scenarios:
        sc_df = all_df[all_df["scenario"] == sc]
        n_total = len(sc_df)
        label_counts = sc_df["label"].value_counts()
        n_attacks = int(label_counts[label_counts.index != "false_positive"].sum())
        fp_count = int(label_counts.get("false_positive", 0))

        for lbl, cnt in label_counts.items():
            if lbl == "false_positive":
                continue
            cnt = int(cnt)
            rows.append(
                {
                    "scenario": sc,
                    "label": lbl,
                    "count": cnt,
                    "pct_of_data": round(100 * cnt / n_total, 3),
                    "pct_of_attacks": round(100 * cnt / n_attacks, 3)
                    if n_attacks
                    else None,
                }
            )

        rows.append(
            {
                "scenario": sc,
                "label": "false_positive",
                "count": fp_count,
                "pct_of_data": round(100 * fp_count / n_total, 3),
                "pct_of_attacks": None,
            }
        )

    return pd.DataFrame(rows)


def compute_attack_temporal_concentration_summary(
    all_df: pd.DataFrame, scenarios: list[str] | None = None
) -> pd.DataFrame:
    """
    Per-scenario burstiness stats behind plot_attack_temporal_concentration /
    plot_attack_temporal_concentration_multi:
      - how far the cumulative-attack curve deviates from uniform arrival,
        and what share of all attack alerts falls in the busiest 1/5/10% of
        the timeline (1h-bin resolution) -- panels (a)/(b);
      - the tail of the inter-arrival-gap distribution behind panel (c)'s
        survival function: what fraction of gaps between consecutive attack
        alerts exceed 1s / 1 minute, and the 99th-percentile gap. Gaps are
        floored to 1s (matching the plot) before these are computed, so
        "pct_gaps_over_1s" reads as "longer than the timestamp's 1s
        resolution", not literally >1s.
    Meant to carry the numbers the multi-scenario plot omits as in-plot
    annotations (too many scenarios overlapping to label individually).
    """
    if scenarios is None:
        scenarios = sorted(all_df["scenario"].unique())

    edges = np.linspace(0, 1, 1001)
    rows = []
    for sc in scenarios:
        sub = all_df[all_df["scenario"] == sc]
        att = sub.loc[sub["is_attack"]].sort_values("time")
        if len(att) < 2:
            rows.append({"scenario": sc, "n_attack": len(att)})
            continue

        t0, t1 = att["time"].min(), sub["time"].max()
        frac_time = (att["time"].values.astype(float) - t0) / (t1 - t0)
        frac_att = np.arange(1, len(att) + 1) / len(att)
        max_dev = float(np.max(np.abs(frac_att - frac_time)))

        binned = np.histogram(frac_time, bins=edges)[0]
        busiest = np.sort(binned)[::-1].cumsum() / max(binned.sum(), 1)

        gap = np.diff(att["time"].values.astype(float))
        gs = np.clip(gap, 1, None) if len(gap) else np.array([])

        rows.append(
            {
                "scenario": sc,
                "n_attack": len(att),
                "max_deviation_from_uniform_pct": round(max_dev * 100, 1),
                "busiest_1pct_share_pct": round(float(busiest[9]) * 100, 1),
                "busiest_5pct_share_pct": round(float(busiest[49]) * 100, 1),
                "busiest_10pct_share_pct": round(float(busiest[99]) * 100, 1),
                "pct_gaps_over_1s": round(float((gs > 1).mean()) * 100, 2)
                if len(gs)
                else None,
                "pct_gaps_over_1min": round(float((gs > 60).mean()) * 100, 3)
                if len(gs)
                else None,
                "p99_gap_s": round(float(np.percentile(gs, 99)), 1)
                if len(gs)
                else None,
            }
        )
    return pd.DataFrame(rows)


def compute_signature_behaviour_summary_multi(
    all_df: pd.DataFrame,
    scenarios: list[str] | None = None,
    sig_col: str = "signature",
    host_col: str = "host",
) -> pd.DataFrame:
    """
    One row per scenario, the numbers behind the "signature behaviour" EDA
    question: (i) volume concentration, (ii) class purity (benign-only /
    attack-only / mixed), (iii) persistence across the collection window
    (by class), and (iv) host spread. Numbers-only counterpart to eyeballing
    per-scenario plots -- every claim in the signature-behaviour writeup
    should be traceable to a column here.
    """
    if scenarios is None:
        scenarios = sorted(all_df["scenario"].unique())

    rows = []
    for sc in scenarios:
        sub = all_df[all_df["scenario"] == sc]

        # (i) volume concentration
        vc = sub[sig_col].value_counts().sort_values(ascending=False)
        cum = vc.cumsum() / vc.sum() if len(vc) else pd.Series(dtype=float)
        top5 = (
            round(float(cum.iloc[min(4, len(cum) - 1)]) * 100, 1) if len(cum) else None
        )
        top10 = (
            round(float(cum.iloc[min(9, len(cum) - 1)]) * 100, 1) if len(cum) else None
        )
        n90 = int((cum < 0.9).sum() + 1) if len(cum) else None

        # (ii) class purity
        cls = classify_signatures(sub, sig_col)
        n_benign = int((cls == "benign").sum())
        n_attack = int((cls == "attack").sum())
        n_mixed = int((cls == "mixed").sum())
        n_total = len(cls)
        pct_mixed = round(100 * n_mixed / n_total, 1) if n_total else None

        # (iii) persistence: days active / scenario duration, by class
        total_days = (sub["timestamp"].max() - sub["timestamp"].min()).days + 1
        day = sub["timestamp"].dt.normalize()
        days_active = day.groupby(sub[sig_col]).nunique()
        frac = days_active / max(total_days, 1)

        def _class_persistence(c):
            ids = cls.index[cls == c]
            f = frac.reindex(ids).dropna()
            n_every_day = int((f >= 0.999).sum())
            median = round(float(f.median()), 2) if len(f) else None
            return n_every_day, median

        n_ed_benign, med_benign = _class_persistence("benign")
        n_ed_attack, med_attack = _class_persistence("attack")
        n_ed_mixed, med_mixed = _class_persistence("mixed")

        # (iv) host spread
        if host_col in sub.columns:
            n_hosts_per_sig = sub.groupby(sig_col)[host_col].nunique()
            pct_exclusive = (
                round(100 * float((n_hosts_per_sig == 1).mean()), 1)
                if len(n_hosts_per_sig)
                else None
            )
            median_hosts = (
                round(float(n_hosts_per_sig.median()), 1)
                if len(n_hosts_per_sig)
                else None
            )
            n_hosts_total = int(sub[host_col].nunique())
        else:
            pct_exclusive = median_hosts = n_hosts_total = None

        rows.append(
            {
                "scenario": sc,
                "n_signatures": n_total,
                "top5_share_pct": top5,
                "top10_share_pct": top10,
                "n_sigs_for_90pct_volume": n90,
                "n_benign_only": n_benign,
                "n_attack_only": n_attack,
                "n_mixed": n_mixed,
                "pct_mixed": pct_mixed,
                "median_days_active_frac_benign": med_benign,
                "median_days_active_frac_attack": med_attack,
                "median_days_active_frac_mixed": med_mixed,
                "n_active_every_day_benign": n_ed_benign,
                "n_active_every_day_attack": n_ed_attack,
                "n_active_every_day_mixed": n_ed_mixed,
                "n_hosts": n_hosts_total,
                "pct_sigs_exclusive_1_host": pct_exclusive,
                "median_hosts_per_sig": median_hosts,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2: overview plots
# ---------------------------------------------------------------------------


def build_overview_plots(
    scenarios: list[str],
    all_df: pd.DataFrame,
    dataset: str,
    bin_hours: float = 1.0,
) -> list[tuple[str, str, object]]:
    """Build every overview-phase figure. Returns [(label, name, fig), ...];
    doesn't save -- caller annotates + persists (see annotate_and_save) or
    just displays inline in a notebook."""

    zoom_params = _ZOOM_PARAMS.get(dataset, {})
    plots: list[tuple[str, str, object]] = []

    if dataset == "ait-ads":
        plots += [
            (
                "alert volume (concatenated timeline)",
                "volume_concatenated",
                lambda: plot_alert_volume_concatenated(all_df, bin_hours=bin_hours),
            ),
            (
                "alert volume (attack phase zoom)",
                "volume_attack_zoom",
                lambda: plot_attack_phase_zoom(all_df, **zoom_params),
            ),
        ]
    if dataset == "cscas":
        plots += [
            (
                "temporal attack overview",
                "temporal_attack_overview",
                lambda: plot_temporal_attack_overview(
                    all_df, bin_hours=zoom_params.get("bin_hours", bin_hours)
                ),
            ),
            (
                "signature event raster",
                "signature_event_raster",
                lambda: plot_signature_event_raster(all_df, time_unit="days"),
            ),
            (
                "signature_burst_raster",
                "signature_burst_raster",
                lambda: plot_occurrence_burst_raster(all_df, time_unit="days"),
            ),
        ]

    plots += [
        (
            "attack type heatmap",
            "attack_type_heatmap",
            lambda: plot_attack_type_heatmap(all_df),
        ),
    ]

    return [(label, name, fn()[0]) for label, name, fn in plots]


def save_overview_plots(
    plots: list[tuple[str, str, object]],
    plots_dir: Path,
    label: str,
    fmt: str = "png",
    close: bool = True,
) -> None:
    """Annotate + save every (label, name, fig) triple from build_overview_plots."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plots] Saving to {plots_dir}")
    for plot_label, name, fig in plots:
        print(f"  Saving {plot_label}...", end=" ", flush=True)
        annotate_and_save(fig, plots_dir / f"{name}.{fmt}", label)
        if close:
            plt.close(fig)
        print("done")


# ---------------------------------------------------------------------------
# Per-host analysis (AIT-ADS only -- 'host' has no cscas equivalent)
# ---------------------------------------------------------------------------


def load_scenario_alerts(
    processed_dir: Path, scenario: str, filtered: bool = False
) -> pd.DataFrame:
    import json

    filename = "alerts_filtered.json" if filtered else "alerts.json"
    path = processed_dir / scenario / filename
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["is_attack"] = df["label"].ne("false_positive") & df["label"].notna()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def compute_streams(times: np.ndarray, gap_threshold: float) -> pd.DataFrame:
    """Split a sorted timestamp array into continuous streams separated by a
    gap >= gap_threshold. Returns one row per stream."""
    if len(times) == 0:
        return pd.DataFrame(
            columns=["start_ts", "end_ts", "n_alerts", "duration_s", "rate_per_s"]
        )

    gaps = np.diff(times)
    boundaries = np.where(gaps >= gap_threshold)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(times)]])

    rows = []
    for s, e in zip(starts, ends):
        seg = times[s:e]
        duration = float(seg[-1] - seg[0])
        n = len(seg)
        rate = (n - 1) / duration if duration > 0 else float(n)
        rows.append(
            {
                "start_ts": int(seg[0]),
                "end_ts": int(seg[-1]),
                "n_alerts": n,
                "duration_s": round(duration, 3),
                "rate_per_s": round(rate, 3),
            }
        )
    return pd.DataFrame(rows)


def inter_arrival_times(times: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.array([])
    return np.diff(np.sort(times)).astype(float)


def summarise_host(host_df: pd.DataFrame, host: str, delta: float) -> dict:
    host_df = host_df.sort_values("time")
    times = host_df["time"].values

    short_counts = host_df["short"].value_counts()
    label_counts = host_df["label"].value_counts()
    n_attack = host_df["is_attack"].sum()
    n_benign = len(host_df) - n_attack

    streams = compute_streams(times, gap_threshold=delta)
    n_streams = len(streams)
    max_stream_n = int(streams["n_alerts"].max()) if n_streams else 0
    max_stream_rate = float(streams["rate_per_s"].max()) if n_streams else 0.0
    max_stream_dur = float(streams["duration_s"].max()) if n_streams else 0.0

    iat = inter_arrival_times(times)
    median_iat = float(np.median(iat)) if len(iat) else float("nan")
    pct_iat_sub_delta = float(np.mean(iat < delta)) if len(iat) else float("nan")

    return {
        "host": host,
        "n_alerts": len(host_df),
        "n_attack": int(n_attack),
        "n_benign": int(n_benign),
        "attack_pct": round(100 * n_attack / len(host_df), 1),
        "n_unique_short": host_df["short"].nunique(),
        "top_shorts": short_counts.head(5).to_dict(),
        "label_counts": label_counts.to_dict(),
        "n_streams": n_streams,
        "max_stream_alerts": max_stream_n,
        "max_stream_rate_per_s": round(max_stream_rate, 2),
        "max_stream_duration_s": round(max_stream_dur, 2),
        "median_iat_s": round(median_iat, 4),
        "pct_iat_sub_delta": round(pct_iat_sub_delta * 100, 1),
        "streams_df": streams,
    }


def write_host_summary(
    df: pd.DataFrame, host_summaries: list[dict], scenario: str, out_path: Path
) -> None:
    with open(out_path, "w") as f:
        f.write(f"Per-host alert EDA — {scenario}\n")
        f.write("=" * 60 + "\n\n")

        t_start = pd.to_datetime(df["time"].min(), unit="s", utc=True)
        t_end = pd.to_datetime(df["time"].max(), unit="s", utc=True)
        duration_h = (df["time"].max() - df["time"].min()) / 3600

        f.write(f"Total alerts:   {len(df):,}\n")
        f.write(f"Time range:     {t_start} → {t_end}\n")
        f.write(f"Duration:       {duration_h:.1f} hours\n")
        f.write(f"Hosts:          {df['host'].nunique()}\n")
        f.write(f"Alert types:    {df['short'].nunique()}\n")
        f.write(f"AlertBERT δ:    {ALERTBERT_DELTA}s (chaining threshold)\n\n")

        f.write("─" * 60 + "\n")
        f.write("PER-HOST SUMMARY\n")
        f.write("─" * 60 + "\n\n")

        for s in host_summaries:
            f.write(f"HOST: {s['host']}\n")
            f.write(f"  Total alerts:         {s['n_alerts']:,}\n")
            f.write(
                f"  Attack alerts:        {s['n_attack']:,} ({s['attack_pct']:.1f}%)\n"
            )
            f.write(f"  Unique alert types:   {s['n_unique_short']}\n")
            f.write(f"  Top alert types:      {s['top_shorts']}\n")
            f.write(f"  Labels seen:          {s['label_counts']}\n")
            f.write(f"  Streams (gap≥{ALERTBERT_DELTA}s):  {s['n_streams']}\n")
            f.write(f"  Largest stream:       {s['max_stream_alerts']:,} alerts\n")
            f.write(
                f"  Max stream rate:      {s['max_stream_rate_per_s']:.1f} alerts/s  "
                f"(IAT ≈ {1 / s['max_stream_rate_per_s'] * 1000:.1f} ms)\n"
                if s["max_stream_rate_per_s"] > 0
                else ""
            )
            f.write(f"  Max stream duration:  {s['max_stream_duration_s']:.1f}s\n")
            f.write(f"  Median IAT:           {s['median_iat_s']:.4f}s\n")
            f.write(f"  % IATs < δ:           {s['pct_iat_sub_delta']:.1f}%\n")
            f.write("\n")

        f.write("─" * 60 + "\n")
        f.write("ALERT TYPE EXCLUSIVITY\n")
        f.write("─" * 60 + "\n\n")

        host_per_type = df.groupby("short")["host"].nunique()
        exclusive = host_per_type[host_per_type == 1]
        shared = host_per_type[host_per_type > 1]
        f.write(
            f"  Exclusive types (1 host only):  {len(exclusive)} of {len(host_per_type)} "
            f"({100 * len(exclusive) / len(host_per_type):.0f}%)\n"
        )
        f.write(f"  Shared types (2+ hosts):        {len(shared)}\n\n")

        f.write("  Exclusive types by host:\n")
        for host in df["host"].unique():
            exc = exclusive[
                [
                    s
                    for s in exclusive.index
                    if df[df["short"] == s]["host"].iloc[0] == host
                ]
            ]
            if len(exc):
                f.write(f"    {host}: {list(exc.index)}\n")

        f.write("\n")
        f.write("─" * 60 + "\n")
        f.write("CROSS-HOST BURST OVERLAP (alertbert grouping concern)\n")
        f.write("─" * 60 + "\n\n")
        f.write("  If two hosts both have % IATs < δ > 90%, time-delta chaining\n")
        f.write("  will merge them into one giant pre-cluster regardless of host.\n\n")

        high_density = [
            (s["host"], s["pct_iat_sub_delta"])
            for s in host_summaries
            if s["pct_iat_sub_delta"] > 90
        ]
        if high_density:
            f.write("  High-density hosts (>90% IATs below δ):\n")
            for host, pct in high_density:
                f.write(f"    {host}: {pct:.1f}%\n")
        else:
            f.write("  No hosts with >90% IATs below δ.\n")
        f.write("\n")


def run_host_scenario_eda(
    scenario: str, run_dir: Path, filtered: bool = False, delta: float = ALERTBERT_DELTA
) -> None:
    """Per-host stream analysis + all six per-host plots + text summary for
    one AIT-ADS scenario. Mirrors the old run_eda_host.py run_scenario()."""
    from thesis.paths import ROOT
    from thesis.visualization.eda import (
        plot_host_alert_type_heatmap,
        plot_host_burst_profile,
        plot_host_exclusive_types,
        plot_host_interarrival_cdf,
        plot_host_timeline,
        plot_host_type_overlap,
    )

    processed_dir = ROOT / "artifacts" / "processed-data"
    label = "data: filtered" if filtered else "data: raw"

    print(f"\n[{scenario}] Loading alerts...")
    df = load_scenario_alerts(processed_dir, scenario, filtered=filtered)
    print(
        f"  {len(df):,} alerts, {df['host'].nunique()} hosts, "
        f"{df['short'].nunique()} alert types"
    )

    out_dir = run_dir / scenario
    per_host_dir = out_dir / "per_host"
    per_host_dir.mkdir(parents=True, exist_ok=True)

    host_summaries = []
    for host in df.groupby("host").size().sort_values(ascending=False).index:
        hdf = df[df["host"] == host].sort_values("time")
        s = summarise_host(hdf, host, delta=delta)
        host_summaries.append(s)
        s["streams_df"].to_csv(per_host_dir / f"{host}_streams.csv", index=False)

        type_stream_rows = []
        for short, group in hdf.groupby("short"):
            times = group["time"].values
            streams = compute_streams(times, gap_threshold=delta)
            for _, row in streams.iterrows():
                type_stream_rows.append({"short": short, **row.to_dict()})
        if type_stream_rows:
            pd.DataFrame(type_stream_rows).sort_values(
                "n_alerts", ascending=False
            ).to_csv(per_host_dir / f"{host}_type_streams.csv", index=False)

    print("  Plotting heatmap...")
    fig, _ = plot_host_alert_type_heatmap(df, scenario)
    annotate_and_save(fig, out_dir / "host_alert_type_heatmap.png", label)

    print("  Plotting timelines...")
    fig, _ = plot_host_timeline(df, scenario)
    annotate_and_save(fig, out_dir / "host_timeline.png", label)

    print("  Plotting inter-arrival CDFs...")
    fig, _ = plot_host_interarrival_cdf(df, scenario, delta=delta)
    annotate_and_save(fig, out_dir / "host_interarrival_cdf.png", label)

    print("  Plotting burst profile...")
    fig, _ = plot_host_burst_profile(host_summaries, scenario, delta=delta)
    annotate_and_save(fig, out_dir / "host_burst_profile.png", label)

    print("  Plotting type overlap...")
    fig, _ = plot_host_type_overlap(df, scenario)
    annotate_and_save(fig, out_dir / "host_type_overlap.png", label)

    print("  Plotting exclusivity...")
    fig, _ = plot_host_exclusive_types(df, scenario)
    annotate_and_save(fig, out_dir / "host_exclusive_types.png", label)

    write_host_summary(df, host_summaries, scenario, out_dir / "summary.txt")
    print(f"  → {out_dir}")
