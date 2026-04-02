import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import numpy as np
import os
import argparse


def time_of_day_bucket(ts):
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
    benign_label: str = "false_positive",
    window_size_s: int = 2,
):
    out = df.copy()

    # Clean
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
    # Window assignment
    out["window_start"] = (out[time_col] // window_size_s) * window_size_s
    out["window_end"] = out["window_start"] + window_size_s

    # Flat items
    out["detector_item"] = out[detector_col].astype(str)
    out["host_item"] = out[host_col].astype(str)

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

    tx = (
        out.groupby(["window_start", "window_end"], sort=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n_alerts": len(g),
                    "items": set(g["detector_item"]).union(set(g["host_item"])),
                    "alert_labels": set(g[label_col].astype(str)),
                    "tx_label": _label_window(g[label_col]),
                    "time_of_day": (
                        g["time_of_day"].mode().iloc[0]
                        if not g["time_of_day"].mode().empty
                        else "unknown"
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return tx


def count_pair_frequency(df: pd.DataFrame, items_col: str = "items") -> pd.DataFrame:
    pair_counter = Counter()

    for items in df[items_col]:
        items = sorted(set(items))
        for pair in combinations(items, 2):
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

    # Count pairs in each class
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
    """
    Compute class-aware TF-IDF-like scores for pairs.

    Expected columns:
      - pair
      - attack_count
      - benign_count

    Interpretation:
      tf_attack  = fraction of attack windows containing the pair
      idf_global = penalizes pairs common across both classes/windows
      tfidf_attack = attack-specific importance

    Returns added columns:
      - tf_attack
      - tf_benign
      - window_frequency
      - idf_global
      - tfidf_attack
      - tfidf_benign
    """
    df = pair_df.copy()

    N = n_attack_windows + n_benign_windows
    df["window_frequency"] = df["attack_count"] + df["benign_count"]

    df["tf_attack"] = df["attack_count"] / max(n_attack_windows, 1)
    df["tf_benign"] = df["benign_count"] / max(n_benign_windows, 1)

    df["idf_global"] = np.log((N + 1) / (1 + df["window_frequency"]))
    df["idf_global"] = df["idf_global"].clip(lower=0)

    df["tfidf_attack"] = df["tf_attack"] * df["idf_global"]
    df["tfidf_benign"] = df["tf_benign"] * df["idf_global"]

    return df


def main(dataset_name, scenario):
    root = Path.cwd().parent.parent
    sys.path.insert(0, str(root / "thesis"))

    data_path = root.parent / "data" / dataset_name
    out_path = root.parent / "out" / "eda" / dataset_name / scenario

    os.makedirs(out_path, exist_ok=True)

    # create text file to store EDA summary
    with open(out_path / "eda_summary.txt", "w") as f:
        f.write(f"Exploratory Data Analysis for {scenario} scenario\n")
        f.write("=" * 50 + "\n\n")

        # Load the dataset
        file_path = os.path.join(data_path, f"{scenario}_alerts.txt")

        # Load the file into a DataFrame
        df = pd.read_csv(file_path)

        # Basic information about the dataset
        f.write("Basic Information:\n")
        f.write(f"Number of rows: {len(df)}\n")
        f.write(f"Number of columns: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")

        # Check for missing values
        f.write("Missing Values:\n")
        missing_values = df.isnull().sum()
        f.write(missing_values.to_string() + "\n\n")

        # Distribution of labels
        if "time_label" in df.columns:
            f.write("Time label Distribution:\n")
            alert_type_counts = df["time_label"].value_counts()
            f.write(alert_type_counts.to_string() + "\n\n")

        # Min and max timestamps
        if "time" in df.columns:
            min_time = df["time"].min()
            max_time = df["time"].max()
            f.write(f"Min timestamp: {min_time}\n")
            f.write(f"Max timestamp: {max_time}\n\n")

        transactions = build_labeled_window_transactions(
            df,
            window_size_s=2,
        )

        transactions.to_csv(
            os.path.join(out_path, f"{scenario}_transactions.csv"), index=False
        )

        benign_transactions = transactions[transactions["tx_label"] == "benign"].copy()
        attack_transactions = transactions[transactions["tx_label"] == "attack"].copy()
        mixed_transactions = transactions[transactions["tx_label"] == "mixed"].copy()

        f.write("Transaction label distribution:\n")
        f.write(transactions["tx_label"].value_counts().to_string() + "\n\n")

        benign_transactions.to_csv(
            os.path.join(out_path, f"{scenario}_benign_transactions.csv"), index=False
        )
        attack_transactions.to_csv(
            os.path.join(out_path, f"{scenario}_attack_transactions.csv"), index=False
        )
        mixed_transactions.to_csv(
            os.path.join(out_path, f"{scenario}_mixed_transactions.csv"), index=False
        )

        # Transaction size
        tx = transactions.copy()

        tx["tx_size"] = tx["items"].apply(len)

        # Summary stats per class
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

        colors = {"benign": "blue", "attack": "red"}

        for label in ["benign", "attack"]:
            subset = tx.loc[tx["tx_label"] == label, "tx_size"]
            plt.hist(subset, bins=20, alpha=0.7, color=colors[label], label=label)

        plt.title(f"Distribution of transaction size (scenario={scenario})")
        plt.xlabel("Number of items in transaction")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.legend()

        plt.savefig(
            os.path.join(out_path, f"{scenario}_transaction_size_distribution.png")
        )

        # Pair frequency
        pair_freq_all = count_pair_frequency(tx)
        pair_freq_all.to_csv(
            os.path.join(out_path, f"{scenario}_pair_frequencies.csv"), index=False
        )

        f.write("Top 20 most common item pairs across all transactions:\n")
        f.write(pair_freq_all.head(20).to_string(index=False) + "\n\n")

        pair_freq_benign = count_pair_frequency(benign_transactions)
        pair_freq_benign.to_csv(
            os.path.join(out_path, f"{scenario}_benign_pair_frequencies.csv"),
            index=False,
        )

        f.write("Top 20 most common item pairs in BENIGN transactions:\n")
        f.write(pair_freq_benign.head(20).to_string(index=False) + "\n\n")

        pair_freq_attack = count_pair_frequency(attack_transactions)
        pair_freq_attack.to_csv(
            os.path.join(out_path, f"{scenario}_attack_pair_frequencies.csv"),
            index=False,
        )

        f.write("Top 20 most common item pairs in ATTACK transactions:\n")
        f.write(pair_freq_attack.head(20).to_string(index=False) + "\n\n")

        # Intersection
        intersection = pd.merge(
            pair_freq_benign.rename(columns={"pair_count": "benign_count"}),
            pair_freq_attack.rename(columns={"pair_count": "attack_count"}),
            on="pair",
            how="inner",
        ).fillna(0)

        intersection.to_csv(
            os.path.join(out_path, f"{scenario}_pair_frequency_intersection.csv"),
            index=False,
        )

        # compare the total number of pairs existing with the number of pairs that exist in both classes
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

        # Support + confidence
        pair_metrics_df = all_pair_metrics(tx)
        pair_metrics_df.to_csv(
            os.path.join(out_path, f"{scenario}_pair_metrics.csv"), index=False
        )

        f.write("Top 20 item pairs by attack count + attack support:\n")
        f.write(pair_metrics_df.head(20).to_string(index=False) + "\n\n")

        df = pair_metrics_df.copy()

        plt.figure()

        plt.scatter(df["support_benign"], df["support_attack"], alpha=0.5)

        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Support (benign)")
        plt.ylabel("Support (attack)")
        plt.title(f"Pair support: attack vs benign (scenario={scenario})")
        plt.savefig(os.path.join(out_path, f"{scenario}_pair_support_scatter.png"))

        # TF-IDF (class specific)
        n_attack_windows = len(attack_transactions)
        n_benign_windows = len(benign_transactions)

        pair_tfidf = compute_pair_tfidf_by_class(
            pair_df=pair_metrics_df[["pair", "attack_count", "benign_count"]].copy(),
            n_attack_windows=n_attack_windows,
            n_benign_windows=n_benign_windows,
        )

        pair_tfidf.to_csv(
            os.path.join(out_path, f"{scenario}_pair_tfidf.csv"), index=False
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA script")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the folder containing the dataset (e.g., 'alerts_csv')",
    )
    parser.add_argument(
        "--scenario", required=True, help="Scenario name for output files"
    )

    args = parser.parse_args()
    main(args.dataset_name, args.output_dir)
