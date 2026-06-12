"""
Generate EDA plots for the alert dataset.

Loads the raw alert CSVs and produces a set of exploratory figures useful for
the dataset introduction chapter of the thesis.  Optionally loads pre-computed
transaction data (produced by run_eda.py) to add transaction volume plots and
extend the scenario overview table with per-transaction statistics.

Usage:
    # all scenarios (alert plots only, raw data)
    python src/thesis/scripts/run_eda_plots.py --all

    # with transaction volume plots (run run_eda.py first)
    python src/thesis/scripts/run_eda_plots.py --all --transactions-dir artifacts/transactions

    # one or more specific scenarios
    python src/thesis/scripts/run_eda_plots.py fox harrison

    # filtered variants — loads artifacts/processed-data/<scenario>/alerts_filtered_<METHOD>.csv
    python src/thesis/scripts/run_eda_plots.py --all --filtered naive50
    python src/thesis/scripts/run_eda_plots.py --all --filtered type_stratified

--filtered options:
    naive50           Random 50/50 undersample of attack vs benign.
    type_stratified   Caps each attack type at the count of the 2nd most common
                      attack type, then keeps all benign samples.
    (bare --filtered) Loads alerts_filtered.csv (legacy detector-score filter).

Output (all saved to <out-dir>/):
    volume_concatenated.png       -- alert volume over concatenated timelines
    volume_attack_zoom.png        -- alert volume zoomed into each attack phase
    class_balance.png             -- benign/attack counts + percentages per scenario
    attack_type_heatmap.png       -- attack type × scenario count heatmap
    top_alert_names.png           -- top-20 most frequent IDS signatures
    inter_arrival_cdf.png         -- inter-arrival time CDF per scenario
    group_size_dist.png           -- alert group size distribution (2s window)
    scenario_overview.png         -- per-scenario summary table (+ tx stats if available)
    tx_volume_concatenated.png    -- transaction volume over concatenated timelines [optional]
    tx_volume_attack_zoom.png     -- transaction volume zoomed into attack phases  [optional]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

from thesis.visualization.eda import (
    SCENARIOS,
    load_alerts,
    load_transactions,
    plot_alert_volume_concatenated,
    plot_attack_phase_zoom,
    plot_attack_type_heatmap,
    plot_class_balance,
    plot_group_size_distribution,
    plot_inter_arrival_time_cdf,
    plot_scenario_overview,
    plot_top_alert_names,
    plot_transaction_volume_attack_zoom,
    plot_transaction_volume_concatenated,
)

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate EDA plots for the alert dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scenarios",
        nargs="*",
        metavar="SCENARIO",
        help=f"Scenario names to include. Choices: {', '.join(SCENARIOS)}",
    )
    p.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help="Include all 8 scenarios (overrides positional args).",
    )
    p.add_argument(
        "--data-dir",
        default="data/alerts_csv",
        metavar="PATH",
        help="Directory containing <scenario>_alerts.txt files (default: data/alerts_csv).",
    )
    p.add_argument(
        "--transactions-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory containing <scenario>_transactions.csv files produced by run_eda.py "
            "(default: artifacts/transactions if it exists, otherwise skipped)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Output directory for figures (default: artifacts/experiments/run_eda_plots/eda_<ts>/plots/).",
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
        help="Output file format (default: png).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        metavar="K",
        help="Number of alert signatures shown in the top-names plot (default: 20).",
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

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = (
            ROOT
            / "artifacts"
            / "experiments"
            / "run_eda_plots"
            / f"eda_{run_ts}"
            / "plots"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

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
            path = ROOT / "artifacts" / "processed-data" / sc / filtered_filename
            frame = pd.read_csv(path, dtype=str)
            frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame["scenario"] = sc
            frame["is_attack"] = (
                frame["time_label"].ne("false_positive") & frame["time_label"].notna()
            )
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = load_alerts(args.data_dir, scenarios=scenarios)
    print(f"  {len(df):,} alerts loaded.")

    # Resolve transactions directory: explicit arg > default location > skip
    if args.transactions_dir is not None:
        tx_dir = (
            Path(args.transactions_dir)
            if Path(args.transactions_dir).is_absolute()
            else ROOT / args.transactions_dir
        )
    else:
        tx_dir = (
            ROOT
            / "artifacts"
            / (
                "transactions_filtered"
                if filtered_method is not None
                else "transactions"
            )
        )

    tx_df = None
    if tx_dir.exists():
        try:
            tx_df = load_transactions(str(tx_dir), scenarios=scenarios)
            print(f"  {len(tx_df):,} transactions loaded from {tx_dir}.")
        except FileNotFoundError as e:
            print(f"  Warning: {e}. Transaction plots will be skipped.")
    else:
        print(
            f"  No transactions dir found at {tx_dir}. "
            "Run run_eda.py first to enable transaction plots."
        )
    print()

    fmt = args.fmt

    def out(name: str) -> str:
        return str(out_dir / f"{name}.{fmt}")

    plots = [
        (
            "alert volume (concatenated timeline)",
            "volume_concatenated",
            lambda: plot_alert_volume_concatenated(df, bin_hours=args.bin_hours),
        ),
        (
            "alert volume (attack phase zoom)",
            "volume_attack_zoom",
            lambda: plot_attack_phase_zoom(df),
        ),
        (
            "class balance",
            "class_balance",
            lambda: plot_class_balance(df),
        ),
        (
            "attack type heatmap",
            "attack_type_heatmap",
            lambda: plot_attack_type_heatmap(df),
        ),
        (
            "top alert names",
            "top_alert_names",
            lambda: plot_top_alert_names(df, top_k=args.top_k),
        ),
        (
            "inter-arrival time CDF",
            "inter_arrival_cdf",
            lambda: plot_inter_arrival_time_cdf(df),
        ),
        (
            "group size distribution",
            "group_size_dist",
            lambda: plot_group_size_distribution(df),
        ),
        (
            "scenario overview table",
            "scenario_overview",
            lambda: plot_scenario_overview(df, tx_df=tx_df),
        ),
    ]

    if tx_df is not None:
        plots += [
            (
                "transaction volume (concatenated timeline)",
                "tx_volume_concatenated",
                lambda: plot_transaction_volume_concatenated(
                    tx_df, bin_hours=args.bin_hours
                ),
            ),
            (
                "transaction volume (attack phase zoom)",
                "tx_volume_attack_zoom",
                lambda: plot_transaction_volume_attack_zoom(tx_df),
            ),
        ]

    if filtered_method is None:
        data_label = "data: raw"
    elif filtered_method:
        data_label = f"data: filtered ({filtered_method})"
    else:
        data_label = "data: filtered"

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
        fig.savefig(out(name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("done")

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
