"""
Exploratory Data Analysis for one or more alert scenarios.

Combines two phases in one script:
  Phase 1 (analysis): per-scenario alert_group stats, pair frequencies, CSVs,
                      plus cross-scenario overview and label-distribution tables.
  Phase 2 (plots):    overview plots — volume timeline, class balance, attack-type
                      heatmap, top signatures, inter-arrival CDF, group sizes, etc.

Usage:
    # full run (analysis + plots)
    python src/thesis/scripts/run_eda.py --all

    # specific scenarios
    python src/thesis/scripts/run_eda.py fox harrison

    # only regenerate plots from existing alert data (skip analysis)
    python src/thesis/scripts/run_eda.py --all --plots-only

    # analysis only, skip plot generation
    python src/thesis/scripts/run_eda.py --all --no-plots

    # balanced alert variants — loads artifacts/alerts/balanced/<METHOD>/<scenario>_alerts.csv
    python src/thesis/scripts/run_eda.py --all --balanced naive50
    python src/thesis/scripts/run_eda.py --all --balanced type_stratified

    # balanced alert_group variants — loads artifacts/alert_groups/balanced/<METHOD>/ (plots only)
    # (raw alerts for alert plots; alert_groups balanced in alert_group space)
    python src/thesis/scripts/run_eda.py --all --tx-balanced naive50 --plots-only

Output (all under artifacts/experiments/run_eda/run_<ts>/):
    <scenario>/                               -- per-scenario CSVs and plots (Phase 1)
    summary/<scenario>_eda_summary.txt        -- per-scenario text summary
    summary/signatures/                       -- unique signature count CSVs
    overview_table.csv / .png                 -- cross-scenario statistics table
    label_distribution_table.csv / .png       -- per-scenario label breakdown
    plots/                                    -- overview figures (Phase 2)
        volume_concatenated.png
        volume_attack_zoom.png
        class_balance.png
        attack_type_heatmap.png
        top_alert_names.png
        inter_arrival_cdf.png
        group_size_dist.png
        scenario_overview.png
        tx_volume_*.png                       -- only when alert_groups are available
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thesis.visualization.eda import (
    SCENARIOS,
    load_alerts,
    load_alert_groups,
    plot_alert_volume_concatenated,
    plot_attack_phase_zoom,
    plot_attack_type_heatmap,
    plot_class_balance,
    plot_group_size_distribution,
    plot_inter_arrival_time_cdf,
    plot_scenario_overview,
    plot_top_alert_names,
    plot_alert_group_volume_attack_zoom,
    plot_alert_group_volume_concatenated,
)
from thesis.preprocessing.alert_groups import build_labeled_window_alert_groups

import matplotlib

matplotlib.use("Agg")


_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_eda"
DATA_DIR = _REPO / "data" / "alerts_csv"
ALERT_GROUPS_BASE_DIR = _REPO / "artifacts" / "alert_groups"


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
    tx: pd.DataFrame,
    items_col: str = "items",
    label_col: str = "group_label",
    min_total_count: int = 1,
) -> pd.DataFrame:
    attack_df = tx[tx[label_col] == "attack"]
    benign_df = tx[tx[label_col] == "benign"]

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


def run_scenario(
    df: pd.DataFrame,
    scenario: str,
    out_path: Path,
    summary_path: Path,
    balanced: str | None = None,
    tx_balanced: str | None = None,
) -> None:
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    with open(summary_path / f"{scenario}_eda_summary.txt", "w") as f:
        f.write(f"Exploratory Data Analysis for {scenario} scenario\n")
        f.write("=" * 50 + "\n\n")

        f.write("Basic Information:\n")
        f.write(f"Number of rows: {len(df)}\n")
        f.write(f"Number of columns: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")

        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string() + "\n\n")

        if "time_label" in df.columns:
            f.write("Time label Distribution:\n")
            f.write(df["time_label"].value_counts().to_string() + "\n\n")

        if "time" in df.columns:
            f.write(f"Min timestamp: {df['time'].min()}\n")
            f.write(f"Max timestamp: {df['time'].max()}\n\n")

        alert_groups = build_labeled_window_alert_groups(df, window_size_s=2)
        tx_cache_dir = _tx_dir(balanced, tx_balanced)
        tx_cache_dir.mkdir(parents=True, exist_ok=True)
        alert_groups.to_csv(tx_cache_dir / f"{scenario}_alert_groups.csv", index=False)

        benign_tx = alert_groups[alert_groups["group_label"] == "benign"].copy()
        attack_tx = alert_groups[alert_groups["group_label"] == "attack"].copy()

        f.write("AlertGroup label distribution:\n")
        f.write(alert_groups["group_label"].value_counts().to_string() + "\n\n")

        tx = alert_groups.copy()
        tx["tx_size"] = tx["items"].apply(len)

        size_summary = tx.groupby("group_label")["tx_size"].describe()
        size_counts = (
            tx.groupby(["group_label", "tx_size"])
            .size()
            .reset_index(name="count")
            .sort_values(["group_label", "tx_size"])
        )

        f.write("AlertGroup Size Summary:\n")
        f.write(size_summary.to_string() + "\n\n")
        f.write("AlertGroup Size Counts:\n")
        f.write(size_counts.to_string(index=False) + "\n\n")

        plt.figure(figsize=(10, 6))
        for label, color in [("benign", "blue"), ("attack", "red")]:
            subset = tx.loc[tx["group_label"] == label, "tx_size"]
            plt.hist(subset, bins=20, alpha=0.7, color=color, label=label)
        plt.title(f"Distribution of alert_group size (scenario={scenario})")
        plt.xlabel("Number of items in alert_group")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.legend()
        plt.gcf().text(
            0.99,
            0.01,
            _data_label(balanced, tx_balanced),
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=plt.gcf().transFigure,
        )
        plt.savefig(out_path / f"{scenario}_alert_group_size_distribution.png")
        plt.close()

        pair_freq_all = count_pair_frequency(tx)
        f.write("Top 20 most common item pairs across all alert_groups:\n")
        f.write(pair_freq_all.head(20).to_string(index=False) + "\n\n")

        pair_freq_benign = count_pair_frequency(benign_tx)
        f.write("Top 20 most common item pairs in BENIGN alert_groups:\n")
        f.write(pair_freq_benign.head(20).to_string(index=False) + "\n\n")

        pair_freq_attack = count_pair_frequency(attack_tx)
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

        pair_metrics_df = all_pair_metrics(tx)
        f.write("Top 20 item pairs by attack count + attack support:\n")
        f.write(pair_metrics_df.head(20).to_string(index=False) + "\n\n")

        plt.figure()
        plt.scatter(
            pair_metrics_df["support_benign"],
            pair_metrics_df["support_attack"],
            alpha=0.5,
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Support (benign)")
        plt.ylabel("Support (attack)")
        plt.title(f"Pair support: attack vs benign (scenario={scenario})")
        plt.gcf().text(
            0.99,
            0.01,
            _data_label(balanced, tx_balanced),
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=plt.gcf().transFigure,
        )
        plt.savefig(out_path / f"{scenario}_pair_support_scatter.png")
        plt.close()

        pair_tfidf = compute_pair_tfidf_by_class(
            pair_df=pair_metrics_df[["pair", "attack_count", "benign_count"]].copy(),
            n_attack_windows=len(attack_tx),
            n_benign_windows=len(benign_tx),
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

        sig_counts = compute_uniq_signature_counts(df, sig_col="name")
        f.write(f"Unique signature counts in data: {len(sig_counts)}\n")
        f.write("Top 30 most common signatures:\n")
        f.write(sig_counts.head(30).to_string(index=False) + "\n\n")

    print(f"  {scenario}: done → {out_path}")


def save_overview_table(
    all_df: pd.DataFrame,
    run_dir: Path,
    balanced: str | None = None,
    tx_balanced: str | None = None,
) -> None:
    scenarios = [s for s in SCENARIOS if s in all_df["scenario"].unique()]

    rows = []
    for sc in scenarios:
        sc_df = all_df[all_df["scenario"] == sc]
        n_total = len(sc_df)
        n_benign = (~sc_df["is_attack"]).sum()
        n_attack = sc_df["is_attack"].sum()
        t_start = pd.to_datetime(sc_df["time"].min(), unit="s", utc=True)
        t_end = pd.to_datetime(sc_df["time"].max(), unit="s", utc=True)
        duration_days = (sc_df["time"].max() - sc_df["time"].min()) / 86400
        n_sig = sc_df["name"].nunique()
        n_attack_types = sc_df.loc[sc_df["is_attack"], "time_label"].nunique()
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

    overview_df = pd.DataFrame(rows)
    overview_df.to_csv(run_dir / "overview_table.csv", index=False)
    print(f"  overview_table.csv → {run_dir / 'overview_table.csv'}")

    col_labels = [
        "Scenario",
        "Start",
        "End",
        "Days",
        "Total",
        "Benign",
        "Attack",
        "Attack %",
        "Alert sigs",
        "Attack types",
    ]
    cell_text = [
        [
            r["scenario"],
            r["start"],
            r["end"],
            str(r["duration_days"]),
            f"{r['total_alerts']:,}",
            f"{r['benign_alerts']:,}",
            f"{r['attack_alerts']:,}",
            f"{r['attack_pct']:.1f}%",
            str(r["unique_signatures"]),
            str(r["attack_types"]),
        ]
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#DDDDDD")
        tbl[0, j].set_text_props(fontweight="bold")
    ax.set_title("Dataset overview — per-scenario statistics", fontsize=12, pad=12)
    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(balanced, tx_balanced),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    fig.savefig(run_dir / "overview_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overview_table.png → {run_dir / 'overview_table.png'}")


def _data_label(balanced: str | None, tx_balanced: str | None = None) -> str:
    if tx_balanced is not None:
        return f"data: raw alerts, tx balanced ({tx_balanced})"
    if balanced is None:
        return "data: raw"
    return f"data: balanced ({balanced})"


def _tx_dir(balanced: str | None, tx_balanced: str | None = None) -> Path:
    if tx_balanced is not None:
        return ALERT_GROUPS_BASE_DIR / "balanced" / tx_balanced
    if balanced is None:
        return ALERT_GROUPS_BASE_DIR / "raw"
    return ALERT_GROUPS_BASE_DIR / "from_balanced_alerts" / balanced


def save_label_distribution_table(
    all_df: pd.DataFrame,
    run_dir: Path,
    balanced: str | None = None,
    tx_balanced: str | None = None,
) -> None:
    """
    Plot a table showing per-scenario breakdown of each time_label:
    count, % of total data, and % of attacks (attack rows only).
    """
    scenarios = [s for s in SCENARIOS if s in all_df["scenario"].unique()]

    col_labels = ["Scenario", "Label", "Count", "% of data", "% of attacks"]
    cell_text = []
    cell_colors = []
    sc_bg = ["#EEF3FF", "#FFF8EE"]

    csv_rows = []

    for i, sc in enumerate(scenarios):
        sc_df = all_df[all_df["scenario"] == sc]
        n_total = len(sc_df)
        label_counts = sc_df["time_label"].value_counts()
        n_attacks = int(label_counts[label_counts.index != "false_positive"].sum())
        fp_count = int(label_counts.get("false_positive", 0))

        attack_labels = [
            (lbl, int(cnt))
            for lbl, cnt in label_counts.items()
            if lbl != "false_positive"
        ]

        bg = sc_bg[i % 2]
        first = True
        for lbl, cnt in attack_labels:
            cell_text.append(
                [
                    sc if first else "",
                    lbl,
                    f"{cnt:,}",
                    f"{100 * cnt / n_total:.1f}%",
                    f"{100 * cnt / n_attacks:.1f}%" if n_attacks > 0 else "—",
                ]
            )
            cell_colors.append([bg] * 5)
            csv_rows.append(
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
            first = False

        cell_text.append(
            [
                sc if first else "",
                "false_positive",
                f"{fp_count:,}",
                f"{100 * fp_count / n_total:.1f}%",
                "—",
            ]
        )
        cell_colors.append([bg] * 5)
        csv_rows.append(
            {
                "scenario": sc,
                "label": "false_positive",
                "count": fp_count,
                "pct_of_data": round(100 * fp_count / n_total, 3),
                "pct_of_attacks": None,
            }
        )

    pd.DataFrame(csv_rows).to_csv(run_dir / "label_distribution_table.csv", index=False)
    print(
        f"  label_distribution_table.csv → {run_dir / 'label_distribution_table.csv'}"
    )

    n_rows = len(cell_text)
    fig_height = max(5, 0.35 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2D2D2D")
        tbl[0, j].set_text_props(fontweight="bold", color="white")

    ax.set_title("Alert label distribution per scenario", fontsize=12, pad=10)
    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        _data_label(balanced, tx_balanced),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = run_dir / "label_distribution_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  label_distribution_table.png → {out}")


def _run_plots_phase(
    scenarios: list[str],
    all_df: pd.DataFrame,
    run_dir: Path,
    balanced: str | None,
    tx_balanced: str | None = None,
    bin_hours: float = 1.0,
    fmt: str = "png",
    top_k: int = 20,
) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tx_dir = _tx_dir(balanced, tx_balanced)
    tx_df = None
    if tx_dir.exists():
        try:
            tx_df = load_alert_groups(str(tx_dir), scenarios=scenarios)
            print(f"  {len(tx_df):,} alert_groups loaded from {tx_dir}.")
        except FileNotFoundError as e:
            print(f"  Warning: {e}. AlertGroup plots will be skipped.")
    else:
        print(
            f"  No alert_groups found at {tx_dir}. "
            "Run without --plots-only first to generate them."
        )

    data_label = _data_label(balanced, tx_balanced)

    def _out(name: str) -> str:
        return str(plots_dir / f"{name}.{fmt}")

    plots = [
        (
            "alert volume (concatenated timeline)",
            "volume_concatenated",
            lambda: plot_alert_volume_concatenated(all_df, bin_hours=bin_hours),
        ),
        (
            "alert volume (attack phase zoom)",
            "volume_attack_zoom",
            lambda: plot_attack_phase_zoom(all_df),
        ),
        ("class balance", "class_balance", lambda: plot_class_balance(all_df)),
        (
            "attack type heatmap",
            "attack_type_heatmap",
            lambda: plot_attack_type_heatmap(all_df),
        ),
        (
            "top alert names",
            "top_alert_names",
            lambda: plot_top_alert_names(all_df, top_k=top_k),
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
            lambda: plot_scenario_overview(all_df, tx_df=tx_df),
        ),
    ]

    if tx_df is not None:
        plots += [
            (
                "alert_group volume (concatenated timeline)",
                "tx_volume_concatenated",
                lambda: plot_alert_group_volume_concatenated(
                    tx_df, bin_hours=bin_hours
                ),
            ),
            (
                "alert_group volume (attack phase zoom)",
                "tx_volume_attack_zoom",
                lambda: plot_alert_group_volume_attack_zoom(tx_df),
            ),
        ]

    print(f"\n[plots] Saving to {plots_dir}")
    for label, name, fn in plots:
        print(f"  Plotting {label}...", end=" ", flush=True)
        fig, _ = fn()
        fig.text(
            0.99,
            0.01,
            data_label,
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )
        fig.savefig(_out(name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("done")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exploratory Data Analysis for alert scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scenarios",
        nargs="*",
        metavar="SCENARIO",
        help=f"Scenario names to analyse. Choices: {', '.join(SCENARIOS)}",
    )
    p.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help="Run EDA for all scenarios.",
    )
    p.add_argument(
        "--balanced",
        default=None,
        metavar="METHOD",
        help=(
            "Load balanced alerts instead of raw. "
            "For Phase 1, loads artifacts/alerts/balanced/<METHOD>/<scenario>_alerts.csv. "
            "For Phase 2, loads artifacts/alert_groups/from_balanced_alerts/<METHOD>/. "
            "Example: --balanced naive50"
        ),
    )
    p.add_argument(
        "--tx-balanced",
        default=None,
        metavar="METHOD",
        help=(
            "Load alert_groups balanced in alert_group space from "
            "artifacts/alert_groups/balanced/<METHOD>/. "
            "Alert plots use raw alert data. "
            "Example: --tx-balanced naive50 --plots-only"
        ),
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip Phase 1 (analysis); only generate overview plots (Phase 2).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Phase 2 (overview plots); run analysis only.",
    )
    p.add_argument(
        "--bin-hours",
        type=float,
        default=1.0,
        metavar="H",
        help="Bin width in hours for the volume-over-time plot (default: 1.0).",
    )
    p.add_argument(
        "--fmt",
        default="png",
        choices=["pdf", "png", "svg"],
        help="Output file format for plots (default: png).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        metavar="K",
        help="Number of alert signatures in the top-names plot (default: 20).",
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
            print(f"Valid choices: {', '.join(SCENARIOS)}")
            sys.exit(1)
        scenarios = [s for s in SCENARIOS if s in args.scenarios]
    else:
        print("Specify scenario names or --all.  Use -h for help.")
        sys.exit(1)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    balanced_method = args.balanced
    tx_balanced_method = args.tx_balanced
    if tx_balanced_method is not None:
        data_tag = f"balanced_tx_{tx_balanced_method}"
    elif balanced_method is not None:
        data_tag = f"balanced_alerts_{balanced_method}"
    else:
        data_tag = "raw"
    run_dir = EXPERIMENTS_DIR / f"run_{run_ts}_{data_tag}"
    summary_path = run_dir / "summary"

    print(f"Loading alerts for: {', '.join(scenarios)}")
    if balanced_method is not None and tx_balanced_method is None:
        print(
            f"  Source: artifacts/alerts/balanced/{balanced_method}/<scenario>_alerts.csv"
        )
        frames = []
        for sc in scenarios:
            path = (
                _REPO
                / "artifacts"
                / "alerts"
                / "balanced"
                / balanced_method
                / f"{sc}_alerts.csv"
            )
            df = pd.read_csv(path, dtype=str)
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["scenario"] = sc
            df["is_attack"] = (
                df["time_label"].ne("false_positive") & df["time_label"].notna()
            )
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
    else:
        if tx_balanced_method is not None:
            print(
                "  Source: raw alerts (alert_groups will be loaded from balanced tx dir)"
            )
        all_df = load_alerts(str(DATA_DIR), scenarios=scenarios)
    print(f"  {len(all_df):,} alerts loaded.\n")

    # Phase 1: per-scenario analysis
    if not args.plots_only:
        for scenario in scenarios:
            scenario_df = all_df[all_df["scenario"] == scenario].copy()
            run_scenario(
                scenario_df,
                scenario,
                run_dir / scenario,
                summary_path,
                balanced=balanced_method,
                tx_balanced=tx_balanced_method,
            )

        print("\nComputing dataset overview table...")
        save_overview_table(
            all_df, run_dir, balanced=balanced_method, tx_balanced=tx_balanced_method
        )

        print("\nComputing label distribution table...")
        save_label_distribution_table(
            all_df, run_dir, balanced=balanced_method, tx_balanced=tx_balanced_method
        )

    # Phase 2: overview plots
    if not args.no_plots:
        _run_plots_phase(
            scenarios,
            all_df,
            run_dir,
            balanced_method,
            tx_balanced=tx_balanced_method,
            bin_hours=args.bin_hours,
            fmt=args.fmt,
            top_k=args.top_k,
        )

    print(f"\nAll output saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
