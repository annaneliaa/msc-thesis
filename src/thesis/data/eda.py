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

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.alert_groups import build_labeled_window_alert_groups

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


# ---------------------------------------------------------------------------
# Item-pair statistics (used by run_scenario_eda)
# ---------------------------------------------------------------------------


def count_pair_frequency(df: pd.DataFrame, items_col: str = "items") -> pd.DataFrame:
    pair_counter: Counter = Counter()
    for items in df[items_col]:
        for pair in combinations(sorted(set(items)), 2):
            pair_counter[pair] += 1
    out = pd.DataFrame(
        [{"pair": pair, "pair_count": count} for pair, count in pair_counter.items()]
    ).sort_values("pair_count", ascending=False)
    return out.reset_index(drop=True)


def all_pair_metrics(
    groups: pd.DataFrame,
    items_col: str = "items",
    label_col: str = "group_label",
    min_total_count: int = 1,
) -> pd.DataFrame:
    attack_df = groups[groups[label_col] == "attack"]
    benign_df = groups[groups[label_col] == "benign"]

    attack_pairs = count_pair_frequency(attack_df, items_col).rename(
        columns={"pair_count": "attack_count"}
    )
    benign_pairs = count_pair_frequency(benign_df, items_col).rename(
        columns={"pair_count": "benign_count"}
    )

    pair_df = attack_pairs.merge(benign_pairs, on="pair", how="outer").fillna(0)
    pair_df["attack_count"] = pair_df["attack_count"].astype(int)
    pair_df["benign_count"] = pair_df["benign_count"].astype(int)
    pair_df["total_count"] = pair_df["attack_count"] + pair_df["benign_count"]

    n_attack = len(attack_df)
    n_benign = len(benign_df)

    pair_df["support_attack"] = pair_df["attack_count"] / n_attack if n_attack else 0.0
    pair_df["support_benign"] = pair_df["benign_count"] / n_benign if n_benign else 0.0

    denom = pair_df["support_attack"] + pair_df["support_benign"]
    pair_df["confidence_attack"] = pair_df["support_attack"] / denom.replace(0, pd.NA)
    pair_df["confidence_benign"] = pair_df["support_benign"] / denom.replace(0, pd.NA)
    pair_df["confidence_attack"] = pair_df["confidence_attack"].fillna(0.0)
    pair_df["confidence_benign"] = pair_df["confidence_benign"].fillna(0.0)

    pair_df = pair_df[pair_df["total_count"] >= min_total_count].copy()
    return pair_df.sort_values(
        ["attack_count", "support_attack", "total_count"], ascending=False
    ).reset_index(drop=True)


def compute_pair_tfidf_by_class(
    pair_df: pd.DataFrame,
    n_attack_windows: int,
    n_benign_windows: int,
) -> pd.DataFrame:
    df = pair_df.copy()
    N = n_attack_windows + n_benign_windows
    df["window_frequency"] = df["attack_count"] + df["benign_count"]
    df["tf_attack"] = df["attack_count"] / max(n_attack_windows, 1)
    df["tf_benign"] = df["benign_count"] / max(n_benign_windows, 1)
    df["idf_global"] = np.log((N + 1) / (1 + df["window_frequency"])).clip(lower=0)
    df["tfidf_attack"] = df["tf_attack"] * df["idf_global"]
    df["tfidf_benign"] = df["tf_benign"] * df["idf_global"]
    return df


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
    alert_groups_base_dir: Path,
    balanced: str | None = None,
    groups_balanced: str | None = None,
    skip_grouping: bool = False,
) -> None:
    """Per-scenario stats, pair-frequency tables, and the two per-scenario
    plots (alert_group size distribution, pair support scatter). Writes a
    text summary to summary_path/<scenario>_eda_summary.txt and plots to
    out_path."""
    from thesis.visualization.eda import (
        plot_alert_group_size_distribution,
        plot_pair_support_scatter,
        plot_signature_event_raster,
    )

    print(f"  Running EDA for scenario '{scenario}' (n={len(df):,} rows)...")
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    label = data_label(balanced, groups_balanced)

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

        if not skip_grouping:
            fig, _ = plot_signature_event_raster(df, time_unit="hours")
            annotate_and_save(
                fig, out_path / f"{scenario}_signature_event_raster.png", label
            )

            alert_groups = build_labeled_window_alert_groups(df, window_size_s=2)
            group_cache_dir = groups_dir(
                alert_groups_base_dir, balanced, groups_balanced
            )
            group_cache_dir.mkdir(parents=True, exist_ok=True)
            alert_groups.to_csv(
                group_cache_dir / f"{scenario}_alert_groups.csv", index=False
            )

            benign_groups = alert_groups[alert_groups["group_label"] == "benign"].copy()
            attack_groups = alert_groups[alert_groups["group_label"] == "attack"].copy()

            f.write("AlertGroup label distribution:\n")
            f.write(alert_groups["group_label"].value_counts().to_string() + "\n\n")

            groups = alert_groups.copy()
            groups["group_size"] = groups["items"].apply(len)

            size_summary = groups.groupby("group_label")["group_size"].describe()
            size_counts = (
                groups.groupby(["group_label", "group_size"])
                .size()
                .reset_index(name="count")
                .sort_values(["group_label", "group_size"])
            )

            f.write("AlertGroup Size Summary:\n")
            f.write(size_summary.to_string() + "\n\n")
            f.write("AlertGroup Size Counts:\n")
            f.write(size_counts.to_string(index=False) + "\n\n")

            fig, _ = plot_alert_group_size_distribution(alert_groups, scenario)
            annotate_and_save(
                fig, out_path / f"{scenario}_alert_group_size_distribution.png", label
            )

            pair_freq_all = count_pair_frequency(groups)
            f.write("Top 20 most common item pairs across all alert_groups:\n")
            f.write(pair_freq_all.head(20).to_string(index=False) + "\n\n")

            pair_freq_benign = count_pair_frequency(benign_groups)
            f.write("Top 20 most common item pairs in BENIGN alert_groups:\n")
            f.write(pair_freq_benign.head(20).to_string(index=False) + "\n\n")

            pair_freq_attack = count_pair_frequency(attack_groups)
            f.write("Top 20 most common item pairs in ATTACK alert_groups:\n")
            f.write(pair_freq_attack.head(20).to_string(index=False) + "\n\n")

            intersection = pd.merge(
                pair_freq_benign.rename(columns={"pair_count": "benign_count"}),
                pair_freq_attack.rename(columns={"pair_count": "attack_count"}),
                on="pair",
                how="inner",
            ).fillna(0)

            total_pairs = len(pair_freq_all)
            intersection_pairs = len(intersection)
            f.write(f"Total unique pairs: {total_pairs}\n")
            f.write(f"Pairs in both classes: {intersection_pairs}\n")
            f.write(
                f"Percentage of pairs in both classes: {intersection_pairs / total_pairs:.2%}\n\n"
            )
            f.write(
                "Top 20 most common item pairs in both BENIGN and ATTACK alert_groups:\n"
            )
            f.write(intersection.head(20).to_string(index=False) + "\n\n")

            pair_metrics_df = all_pair_metrics(groups)
            f.write("Top 20 item pairs by attack count + attack support:\n")
            f.write(pair_metrics_df.head(20).to_string(index=False) + "\n\n")

            fig, _ = plot_pair_support_scatter(pair_metrics_df, scenario)
            annotate_and_save(
                fig, out_path / f"{scenario}_pair_support_scatter.png", label
            )

            pair_tfidf = compute_pair_tfidf_by_class(
                pair_df=pair_metrics_df[
                    ["pair", "attack_count", "benign_count"]
                ].copy(),
                n_attack_windows=len(attack_groups),
                n_benign_windows=len(benign_groups),
            )
            f.write("Top 20 item pairs by attack TF-IDF score:\n")
            f.write(
                pair_tfidf.sort_values("tfidf_attack", ascending=False)
                .head(20)
                .to_string(index=False)
                + "\n\n"
            )
            f.write("Top 20 item pairs by benign TF-IDF score:\n")
            f.write(
                pair_tfidf.sort_values("tfidf_benign", ascending=False)
                .head(20)
                .to_string(index=False)
                + "\n\n"
            )

        sig_counts = compute_uniq_signature_counts(df, sig_col="signature")
        f.write(f"Unique signature counts in data: {len(sig_counts)}\n")
        f.write("Top 30 most common signatures:\n")
        f.write(sig_counts.head(30).to_string(index=False) + "\n\n")

    print(f"  {scenario}: done → {out_path}")


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


# ---------------------------------------------------------------------------
# Phase 2: overview plots
# ---------------------------------------------------------------------------


def build_overview_plots(
    scenarios: list[str],
    all_df: pd.DataFrame,
    dataset: str,
    groups_df: pd.DataFrame | None = None,
    bin_hours: float = 1.0,
    top_k: int = 20,
) -> list[tuple[str, str, object]]:
    """Build every overview-phase figure. Returns [(label, name, fig), ...];
    doesn't save -- caller annotates + persists (see annotate_and_save) or
    just displays inline in a notebook."""
    from thesis.visualization.eda import (
        plot_alert_group_volume_attack_zoom,
        plot_alert_group_volume_concatenated,
        plot_alert_volume_concatenated,
        plot_attack_phase_zoom,
        plot_attack_type_heatmap,
        plot_class_balance,
        plot_group_size_distribution,
        plot_inter_arrival_time_cdf,
        plot_scenario_overview,
        plot_signature_event_raster,
        plot_signature_purity_pie,
        plot_temporal_attack_overview,
        plot_top_alert_signatures,
    )

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
                "signature purity pie",
                "signature_purity_pie",
                lambda: plot_signature_purity_pie(all_df),
            ),
        ]

    plots += [
        ("class balance", "class_balance", lambda: plot_class_balance(all_df)),
        (
            "attack type heatmap",
            "attack_type_heatmap",
            lambda: plot_attack_type_heatmap(all_df),
        ),
        (
            "top alert signatures",
            "top_alert_signatures",
            lambda: plot_top_alert_signatures(all_df, top_k=top_k),
        ),
        (
            "inter-arrival time CDF",
            "inter_arrival_cdf",
            lambda: plot_inter_arrival_time_cdf(all_df),
        ),
        (
            "group size distribution",
            "group_size_dist",
            lambda: plot_group_size_distribution(all_df),
        ),
        (
            "scenario overview table",
            "scenario_overview",
            lambda: plot_scenario_overview(all_df, groups_df=groups_df),
        ),
    ]

    if groups_df is not None:
        groups_bin = zoom_params.get("bin_hours", bin_hours)
        plots += [
            (
                "alert_group volume (concatenated timeline)",
                "groups_volume_concatenated",
                lambda: plot_alert_group_volume_concatenated(
                    groups_df, bin_hours=groups_bin
                ),
            ),
            (
                "alert_group volume (attack phase zoom)",
                "groups_volume_attack_zoom",
                lambda: plot_alert_group_volume_attack_zoom(groups_df, **zoom_params),
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
