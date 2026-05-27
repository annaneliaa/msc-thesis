"""
Per-host alert EDA for a given scenario.

Answers:
  - What alert types (short codes) fire at each host?
  - What streams exist per host and at what rate?
  - Are alert type vocabularies host-specific or shared?
  - Do hosts exhibit independent burst patterns?
  - Are there cross-host grouping artefacts (same burst, multiple hosts)?

Usage:
    python src/thesis/scripts/run_eda_host.py fox
    python src/thesis/scripts/run_eda_host.py fox harrison
    python src/thesis/scripts/run_eda_host.py --all

Output:
    artifacts/experiments/run_eda_host/run_<ts>/
        <scenario>/
            summary.txt
            host_alert_type_heatmap.png
            host_timeline.png
            host_interarrival_cdf.png
            host_burst_profile.png
            host_type_overlap.png
            per_host/
                <host>_streams.csv
                <host>_burst_segments.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

matplotlib.use("Agg")

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())

PROCESSED_DIR = _REPO / "artifacts" / "processed-data"
EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_eda_host"

SCENARIOS = [
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
]

BENIGN_LABEL = "false_positive"

# Gap threshold used by AlertBERT time-delta chaining (seconds)
ALERTBERT_DELTA = 2.0
# Fixed window size for the new host-based grouping (seconds) — adjust as needed
FIXED_WINDOW_S = 60


# ── data loading ──────────────────────────────────────────────────────────────


def load_scenario(scenario: str) -> pd.DataFrame:
    path = PROCESSED_DIR / scenario / "alerts.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["is_attack"] = df["time_label"].ne(BENIGN_LABEL) & df["time_label"].notna()
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ── stream / burst helpers ────────────────────────────────────────────────────


def compute_streams(times: np.ndarray, gap_threshold: float) -> pd.DataFrame:
    """
    Split a sorted timestamp array into continuous streams separated by
    a gap >= gap_threshold.  Returns a DataFrame with one row per stream.
    """
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


# ── per-host summary ──────────────────────────────────────────────────────────


def summarise_host(host_df: pd.DataFrame, host: str, delta: float) -> dict:
    host_df = host_df.sort_values("time")
    times = host_df["time"].values

    short_counts = host_df["short"].value_counts()
    label_counts = host_df["time_label"].value_counts()
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


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_heatmap(df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    """Alert-type × host heatmap (count, log-scale)."""
    pivot = df.groupby(["host", "short"]).size().unstack(fill_value=0)
    # Sort hosts by total alert count descending; short codes by total descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(
        figsize=(max(12, len(pivot.columns) * 0.55), max(4, len(pivot) * 0.55 + 1.5))
    )
    data = pivot.values.astype(float)
    data_masked = np.ma.masked_where(data == 0, data)

    vmin = max(1, data_masked.min()) if data_masked.count() else 1
    im = ax.imshow(
        data_masked,
        aspect="auto",
        cmap="YlOrRd",
        norm=LogNorm(vmin=vmin, vmax=data.max() + 1),
    )
    plt.colorbar(im, ax=ax, label="Alert count (log scale)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f"{scenario} — Alert types per host", fontsize=12)
    ax.set_xlabel("Alert type (short code)")
    ax.set_ylabel("Host")

    # Annotate non-zero cells with count
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            if v > 0:
                text_color = "white" if v > data.max() / 4 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(v):,}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(
    df: pd.DataFrame, scenario: str, out_path: Path, bin_minutes: int = 5
) -> None:
    """Per-host alert rate over time (stacked subplots, benign vs attack)."""
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    n = len(hosts)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t_start = df["time"].min()
    t_end = df["time"].max()
    bin_s = bin_minutes * 60
    edges = np.arange(t_start, t_end + bin_s, bin_s)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    hours_elapsed = (bin_centers - t_start) / 3600

    for ax, host in zip(axes, hosts):
        hdf = df[df["host"] == host]
        benign = hdf[~hdf["is_attack"]]["time"].values
        attack = hdf[hdf["is_attack"]]["time"].values

        b_counts, _ = np.histogram(benign, bins=edges)
        a_counts, _ = np.histogram(attack, bins=edges)

        b_rate = np.where(b_counts > 0, b_counts / (bin_s / 60), np.nan)
        a_rate = np.where(a_counts > 0, a_counts / (bin_s / 60), np.nan)

        ax.fill_between(
            hours_elapsed,
            b_rate,
            alpha=0.6,
            color="#4477AA",
            label="benign",
            step="mid",
        )
        ax.fill_between(
            hours_elapsed,
            a_rate,
            alpha=0.8,
            color="#CC3311",
            label="attack",
            step="mid",
        )

        ax.set_ylabel(host, rotation=0, labelpad=6, ha="right", fontsize=8, va="center")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(axis="y", labelsize=7)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    axes[-1].set_xlabel("Elapsed time (h)")
    fig.suptitle(
        f"{scenario} — Alert rate per host ({bin_minutes}-min bins, alerts/min)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_interarrival_cdf(df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    """CDF of inter-arrival times per host (log x-axis)."""
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("tab10")
    for i, host in enumerate(hosts):
        hdf = df[df["host"] == host].sort_values("time")
        iat = inter_arrival_times(hdf["time"].values)
        if len(iat) == 0:
            continue
        iat_sorted = np.sort(iat)
        cdf = np.arange(1, len(iat_sorted) + 1) / len(iat_sorted)
        ax.plot(iat_sorted, cdf, label=host, color=cmap(i % 10), linewidth=1.5)

    ax.axvline(
        ALERTBERT_DELTA,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"AlertBERT δ={ALERTBERT_DELTA}s",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Inter-arrival time (seconds, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{scenario} — Inter-arrival time CDF per host")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_burst_profile(
    host_summaries: list[dict], scenario: str, out_path: Path
) -> None:
    """Bar chart: max stream size and % of IATs below delta, per host."""
    hosts = [s["host"] for s in host_summaries]
    max_stream = [s["max_stream_alerts"] for s in host_summaries]
    pct_sub_delta = [s["pct_iat_sub_delta"] for s in host_summaries]

    x = np.arange(len(hosts))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    bars = ax1.bar(x, max_stream, color="#CC3311", alpha=0.8)
    ax1.set_ylabel("Max stream size (alerts)")
    ax1.set_title(f"{scenario} — Per-host burst profile")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, v in zip(bars, max_stream):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2.bar(x, pct_sub_delta, color="#4477AA", alpha=0.8)
    ax2.axhline(50, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(f"% inter-arrival times < δ={ALERTBERT_DELTA}s")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hosts, rotation=25, ha="right", fontsize=8)
    ax2.set_xlabel("Host")
    for i, v in enumerate(pct_sub_delta):
        ax2.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_type_overlap(df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    """
    Host × host Jaccard similarity of alert-type sets.
    Low Jaccard = hosts see different alert types → per-host grouping separates semantically distinct streams.
    """
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    type_sets = {h: set(df[df["host"] == h]["short"].unique()) for h in hosts}

    n = len(hosts)
    matrix = np.zeros((n, n))
    for i, h1 in enumerate(hosts):
        for j, h2 in enumerate(hosts):
            s1, s2 = type_sets[h1], type_sets[h2]
            union = s1 | s2
            matrix[i, j] = len(s1 & s2) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.8)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Jaccard similarity of alert-type sets")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(hosts, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(hosts, fontsize=8)
    ax.set_title(
        f"{scenario} — Alert-type vocabulary overlap between hosts\n"
        "(low Jaccard → hosts speak different alert languages)"
    )
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > 0.6 else "black",
            )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exclusive_types(df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    """
    Stacked bar: for each alert type, how many hosts see it?
    Types seen at exactly 1 host are 'exclusive' — grouping by host keeps them pure.
    """
    host_per_type = df.groupby("short")["host"].nunique().sort_values(ascending=False)
    short_totals = df["short"].value_counts()

    # Bucket by host count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bucket_labels = ["1 host (exclusive)", "2–3 hosts", "4+ hosts"]
    buckets = [
        host_per_type[host_per_type == 1],
        host_per_type[(host_per_type >= 2) & (host_per_type <= 3)],
        host_per_type[host_per_type >= 4],
    ]
    colors = ["#228833", "#CCBB44", "#CC3311"]
    sizes = [len(b) for b in buckets]
    alert_volumes = [short_totals[b.index].sum() for b in buckets]

    ax1.bar(bucket_labels, sizes, color=colors, alpha=0.85)
    ax1.set_ylabel("Number of distinct alert types")
    ax1.set_title("Alert type host exclusivity\n(by # distinct types)")
    for i, (v, av) in enumerate(zip(sizes, alert_volumes)):
        ax1.text(
            i,
            v + 0.3,
            f"{v} types\n({av:,} alerts)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylim(0, max(sizes) * 1.25)

    # Show which types appear at only 1 host
    exclusive = host_per_type[host_per_type == 1]
    exc_df = pd.DataFrame(
        {
            "short": exclusive.index,
            "host": [df[df["short"] == s]["host"].iloc[0] for s in exclusive.index],
            "n_alerts": [short_totals.get(s, 0) for s in exclusive.index],
        }
    ).sort_values(["host", "n_alerts"], ascending=[True, False])

    host_colors = {
        h: plt.get_cmap("tab10")(i % 10) for i, h in enumerate(exc_df["host"].unique())
    }
    bar_colors = [host_colors[h] for h in exc_df["host"]]

    ax2.barh(range(len(exc_df)), exc_df["n_alerts"], color=bar_colors, alpha=0.85)
    ax2.set_yticks(range(len(exc_df)))
    ax2.set_yticklabels(exc_df["short"], fontsize=7)
    ax2.set_xlabel("Alert count (log scale)")
    ax2.set_xscale("log")
    ax2.set_title("Host-exclusive alert types\n(color = host)")
    # Legend for hosts
    patches = [plt.Rectangle((0, 0), 1, 1, color=host_colors[h]) for h in host_colors]
    ax2.legend(patches, list(host_colors.keys()), fontsize=7, loc="lower right")

    fig.suptitle(f"{scenario} — Alert type host exclusivity", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── text summary ──────────────────────────────────────────────────────────────


def write_summary(
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
                f"(IAT ≈ {1/s['max_stream_rate_per_s']*1000:.1f} ms)\n"
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
            f"({100*len(exclusive)/len(host_per_type):.0f}%)\n"
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


# ── main ──────────────────────────────────────────────────────────────────────


def run_scenario(scenario: str, run_dir: Path) -> None:
    print(f"\n[{scenario}] Loading alerts...")
    df = load_scenario(scenario)
    print(
        f"  {len(df):,} alerts, {df['host'].nunique()} hosts, "
        f"{df['short'].nunique()} alert types"
    )

    out_dir = run_dir / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    per_host_dir = out_dir / "per_host"
    per_host_dir.mkdir(parents=True, exist_ok=True)

    # Per-host stream analysis
    host_summaries = []
    for host in df.groupby("host").size().sort_values(ascending=False).index:
        hdf = df[df["host"] == host].sort_values("time")
        s = summarise_host(hdf, host, delta=ALERTBERT_DELTA)
        host_summaries.append(s)

        s["streams_df"].to_csv(per_host_dir / f"{host}_streams.csv", index=False)

        # Per-host, per short-type streams
        type_stream_rows = []
        for short, group in hdf.groupby("short"):
            times = group["time"].values
            streams = compute_streams(times, gap_threshold=ALERTBERT_DELTA)
            for _, row in streams.iterrows():
                type_stream_rows.append(
                    {
                        "short": short,
                        **row.to_dict(),
                    }
                )
        if type_stream_rows:
            pd.DataFrame(type_stream_rows).sort_values(
                "n_alerts", ascending=False
            ).to_csv(per_host_dir / f"{host}_type_streams.csv", index=False)

    # Plots
    print("  Plotting heatmap...")
    plot_heatmap(df, scenario, out_dir / "host_alert_type_heatmap.png")

    print("  Plotting timelines...")
    plot_timeline(df, scenario, out_dir / "host_timeline.png")

    print("  Plotting inter-arrival CDFs...")
    plot_interarrival_cdf(df, scenario, out_dir / "host_interarrival_cdf.png")

    print("  Plotting burst profile...")
    plot_burst_profile(host_summaries, scenario, out_dir / "host_burst_profile.png")

    print("  Plotting type overlap...")
    plot_type_overlap(df, scenario, out_dir / "host_type_overlap.png")

    print("  Plotting exclusivity...")
    plot_exclusive_types(df, scenario, out_dir / "host_exclusive_types.png")

    # Text summary
    write_summary(df, host_summaries, scenario, out_dir / "summary.txt")

    print(f"  → {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "scenarios",
        nargs="*",
        metavar="SCENARIO",
        help=f"Choices: {', '.join(SCENARIOS)}",
    )
    p.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help="Run for all scenarios.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.all_scenarios:
        scenarios = SCENARIOS
    elif args.scenarios:
        unknown = [s for s in args.scenarios if s not in SCENARIOS]
        if unknown:
            print(f"Unknown scenario(s): {', '.join(unknown)}")
            sys.exit(1)
        scenarios = args.scenarios
    else:
        print("Specify scenario names or --all.  Use -h for help.")
        sys.exit(1)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        run_scenario(scenario, run_dir)

    print(f"\nDone. Output: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
