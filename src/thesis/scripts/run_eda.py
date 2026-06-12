"""
Exploratory Data Analysis for one or more alert scenarios.

Usage:
    # all scenarios (raw alerts)
    python src/thesis/scripts/run_eda.py --all

    # one or more specific scenarios
    python src/thesis/scripts/run_eda.py fox harrison

    # filtered variants — loads artifacts/processed-data/<scenario>/alerts_filtered_<METHOD>.csv
    python src/thesis/scripts/run_eda.py --all --filtered naive50
    python src/thesis/scripts/run_eda.py --all --filtered type_stratified

--filtered options:
    naive50           Random 50/50 undersample of attack vs benign.
    type_stratified   Caps each attack type at the count of the 2nd most common
                      attack type, then keeps all benign samples.
    (bare --filtered) Loads alerts_filtered.csv (legacy detector-score filter).

Output (all under artifacts/experiments/run_eda/run_<ts>/):
    <scenario>/                               -- per-scenario CSVs and plots
    summary/<scenario>_eda_summary.txt        -- per-scenario text summary
    summary/signatures/                       -- unique signature count CSVs
    overview_table.csv                        -- cross-scenario statistics table
    overview_table.png                        -- same table as a figure
    label_distribution_table.csv             -- per-scenario label breakdown (count, % of data, % of attacks)
    label_distribution_table.png             -- same table as a figure
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
from thesis.visualization.eda import SCENARIOS, load_alerts
from thesis.preprocessing.tokenization import (
    extract_signature_tokens,
)

import matplotlib

matplotlib.use("Agg")


_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_eda"
DATA_DIR = _REPO / "data" / "alerts_csv"
TRANSACTIONS_DIR = _REPO / "artifacts" / "transactions"
TRANSACTIONS_FILTERED_DIR = _REPO / "artifacts" / "transactions_filtered"


def extract_short_tokens(short_value: str) -> set[str]:
    if pd.isna(short_value):
        return set()
    parts = [part.strip() for part in str(short_value).split("-")]
    return {part for part in parts if part}


def time_of_day_bucket(ts) -> str:
    if pd.isna(ts):
        return "unknown"
    h = ts.hour
    if 5 <= h < 10:
        return "morning"
    elif 10 <= h < 14:
        return "midday"
    elif 14 <= h < 18:
        return "afternoon"
    elif 18 <= h < 22:
        return "evening"
    else:
        return "night"


def build_labeled_window_transactions(
    df: pd.DataFrame,
    time_col: str = "time",
    detector_col: str = "short",
    host_col: str = "host",
    label_col: str = "time_label",
    signature_col: str = "name",
    benign_label: str = "false_positive",
    window_size_s: int = 2,
) -> pd.DataFrame:
    out = df.copy()

    needed = [time_col, detector_col, host_col, label_col]
    out = out.dropna(subset=needed).copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out[time_col] = out[time_col].astype("int64")

    out["time_norm"] = pd.to_datetime(
        out[time_col], unit="s", utc=True, errors="coerce"
    )
    out["time_of_day"] = out["time_norm"].apply(time_of_day_bucket)
    out["time_epoch"] = (out["time_norm"].astype("int64") // 10**9).astype("int64")
    out["window_start"] = (out[time_col] // window_size_s) * window_size_s
    out["window_end"] = out["window_start"] + window_size_s

    out["detector_item"] = out[detector_col].astype(str)
    out["host_item"] = out[host_col].astype(str)
    out["detector_subtokens"] = out[detector_col].apply(extract_short_tokens)

    if signature_col in out.columns:
        out["signature_tokens"] = out[signature_col].apply(extract_signature_tokens)
    else:
        out["signature_tokens"] = [set() for _ in range(len(out))]

    def _label_window(labels: pd.Series) -> str:
        labels = set(labels.astype(str))
        has_benign = benign_label in labels
        has_attack = any(lbl != benign_label for lbl in labels)
        if has_attack and has_benign:
            return "mixed"
        elif has_attack:
            return "attack"
        else:
            return "benign"

    def _build_items(g: pd.DataFrame) -> set[str]:
        items = set(g["detector_item"]).union(set(g["host_item"]))
        for toks in g["detector_subtokens"]:
            items.update(toks)
        for toks in g["signature_tokens"]:
            items.update(toks)
        return items

    tx = (
        out.groupby(["window_start", "window_end"], sort=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n_alerts": len(g),
                    "items": _build_items(g),
                    "alert_labels": set(g[label_col].astype(str)),
                    "tx_label": _label_window(g[label_col]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return tx


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
    label_col: str = "tx_label",
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
    filtered: str | None = None,
) -> None:
    _is_filtered = filtered is not None
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

        transactions = build_labeled_window_transactions(df, window_size_s=2)
        transactions.to_csv(out_path / f"{scenario}_transactions.csv", index=False)
        tx_cache_dir = TRANSACTIONS_FILTERED_DIR if _is_filtered else TRANSACTIONS_DIR
        tx_cache_dir.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(tx_cache_dir / f"{scenario}_transactions.csv", index=False)

        benign_tx = transactions[transactions["tx_label"] == "benign"].copy()
        attack_tx = transactions[transactions["tx_label"] == "attack"].copy()
        mixed_tx = transactions[transactions["tx_label"] == "mixed"].copy()

        f.write("Transaction label distribution:\n")
        f.write(transactions["tx_label"].value_counts().to_string() + "\n\n")

        benign_tx.to_csv(out_path / f"{scenario}_benign_transactions.csv", index=False)
        attack_tx.to_csv(out_path / f"{scenario}_attack_transactions.csv", index=False)
        mixed_tx.to_csv(out_path / f"{scenario}_mixed_transactions.csv", index=False)

        tx = transactions.copy()
        tx["tx_size"] = tx["items"].apply(len)

        size_summary = tx.groupby("tx_label")["tx_size"].describe()
        size_counts = (
            tx.groupby(["tx_label", "tx_size"])
            .size()
            .reset_index(name="count")
            .sort_values(["tx_label", "tx_size"])
        )

        f.write("Transaction Size Summary:\n")
        f.write(size_summary.to_string() + "\n\n")
        f.write("Transaction Size Counts:\n")
        f.write(size_counts.to_string(index=False) + "\n\n")

        plt.figure(figsize=(10, 6))
        for label, color in [("benign", "blue"), ("attack", "red")]:
            subset = tx.loc[tx["tx_label"] == label, "tx_size"]
            plt.hist(subset, bins=20, alpha=0.7, color=color, label=label)
        plt.title(f"Distribution of transaction size (scenario={scenario})")
        plt.xlabel("Number of items in transaction")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.legend()
        plt.gcf().text(
            0.99,
            0.01,
            _data_label(filtered),
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=plt.gcf().transFigure,
        )
        plt.savefig(out_path / f"{scenario}_transaction_size_distribution.png")
        plt.close()

        pair_freq_all = count_pair_frequency(tx)
        pair_freq_all.to_csv(out_path / f"{scenario}_pair_frequencies.csv", index=False)
        f.write("Top 20 most common item pairs across all transactions:\n")
        f.write(pair_freq_all.head(20).to_string(index=False) + "\n\n")

        pair_freq_benign = count_pair_frequency(benign_tx)
        pair_freq_benign.to_csv(
            out_path / f"{scenario}_benign_pair_frequencies.csv", index=False
        )
        f.write("Top 20 most common item pairs in BENIGN transactions:\n")
        f.write(pair_freq_benign.head(20).to_string(index=False) + "\n\n")

        pair_freq_attack = count_pair_frequency(attack_tx)
        pair_freq_attack.to_csv(
            out_path / f"{scenario}_attack_pair_frequencies.csv", index=False
        )
        f.write("Top 20 most common item pairs in ATTACK transactions:\n")
        f.write(pair_freq_attack.head(20).to_string(index=False) + "\n\n")

        intersection = pd.merge(
            pair_freq_benign.rename(columns={"pair_count": "benign_count"}),
            pair_freq_attack.rename(columns={"pair_count": "attack_count"}),
            on="pair",
            how="inner",
        ).fillna(0)
        intersection.to_csv(
            out_path / f"{scenario}_pair_frequency_intersection.csv", index=False
        )

        total_pairs = len(pair_freq_all)
        intersection_pairs = len(intersection)
        f.write(f"Total unique pairs: {total_pairs}\n")
        f.write(f"Pairs in both classes: {intersection_pairs}\n")
        f.write(
            f"Percentage of pairs in both classes: {intersection_pairs / total_pairs:.2%}\n\n"
        )
        f.write(
            "Top 20 most common item pairs in both BENIGN and ATTACK transactions:\n"
        )
        f.write(intersection.head(20).to_string(index=False) + "\n\n")

        pair_metrics_df = all_pair_metrics(tx)
        pair_metrics_df.to_csv(out_path / f"{scenario}_pair_metrics.csv", index=False)
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
            _data_label(filtered),
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
        pair_tfidf.to_csv(out_path / f"{scenario}_pair_tfidf.csv", index=False)
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

        sig_dir = summary_path / "signatures"
        os.makedirs(sig_dir, exist_ok=True)
        sig_counts.to_csv(
            sig_dir / f"{scenario}_unique_signature_counts.csv", index=False
        )

    print(f"  {scenario}: done → {out_path}")


def save_overview_table(
    all_df: pd.DataFrame, run_dir: Path, filtered: str | None = None
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
        _data_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    fig.savefig(run_dir / "overview_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overview_table.png → {run_dir / 'overview_table.png'}")


def _data_label(filtered: str | None) -> str:
    if filtered is None:
        return "data: raw"
    return f"data: filtered ({filtered})" if filtered else "data: filtered"


def save_label_distribution_table(
    all_df: pd.DataFrame, run_dir: Path, filtered: str | None = None
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
        _data_label(filtered),
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
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help=(
            "Load a pre-processed variant instead of raw alerts. "
            "Pass a method name to load alerts_filtered_<METHOD>.csv "
            "(e.g. --filtered naive50), or bare --filtered for alerts_filtered.csv."
        ),
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
    run_dir = EXPERIMENTS_DIR / f"run_{run_ts}"
    summary_path = run_dir / "summary"

    filtered_method = (
        args.filtered
    )  # None = raw, "" = alerts_filtered.csv, "naive50" = alerts_filtered_naive50.csv
    suffix = f"_{filtered_method}" if filtered_method else ""
    filtered_filename = f"alerts_filtered{suffix}.csv"

    print(f"Loading alerts for: {', '.join(scenarios)}")
    if filtered_method is not None:
        print(f"  Source: artifacts/processed-data/<scenario>/{filtered_filename}")
        frames = []
        for sc in scenarios:
            path = _REPO / "artifacts" / "processed-data" / sc / filtered_filename
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
        all_df = load_alerts(str(DATA_DIR), scenarios=scenarios)
    print(f"  {len(all_df):,} alerts loaded.\n")

    for scenario in scenarios:
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        run_scenario(
            scenario_df,
            scenario,
            run_dir / scenario,
            summary_path,
            filtered=filtered_method,
        )

    print("\nComputing dataset overview table...")
    save_overview_table(all_df, run_dir, filtered=filtered_method)

    print("\nComputing label distribution table...")
    save_label_distribution_table(all_df, run_dir, filtered=filtered_method)

    print(f"\nAll output saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
