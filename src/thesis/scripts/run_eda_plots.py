"""
Generate EDA plots for the alert dataset.

Loads the raw alert CSVs and produces a set of exploratory figures useful for
the dataset introduction chapter of the thesis.

Usage:
    # all scenarios
    python src/thesis/scripts/run_eda_plots.py --all

    # one or more specific scenarios
    python src/thesis/scripts/run_eda_plots.py fox harrison

    # custom output directory and bin width
    python src/thesis/scripts/run_eda_plots.py --all --out-dir plots/eda --bin-hours 2

Output (all saved to <out-dir>/):
    volume_concatenated.pdf   -- alert volume over concatenated timelines
    class_balance.pdf         -- benign/attack counts + percentages per scenario
    attack_type_heatmap.pdf   -- attack type × scenario count heatmap
    top_alert_names.pdf       -- top-20 most frequent IDS signatures
    inter_arrival_cdf.pdf     -- inter-arrival time CDF per scenario
    group_size_dist.pdf       -- alert group size distribution (2s window)
    scenario_overview.pdf     -- per-scenario summary table
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

from src.thesis.visualization.eda import (
    SCENARIOS,
    load_alerts,
    plot_alert_volume_concatenated,
    plot_attack_type_heatmap,
    plot_class_balance,
    plot_group_size_distribution,
    plot_inter_arrival_time_cdf,
    plot_scenario_overview,
    plot_top_alert_names,
)


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


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
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output file format (default: pdf).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        metavar="K",
        help="Number of alert signatures shown in the top-names plot (default: 20).",
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

    print(f"Loading alerts for: {', '.join(scenarios)}")
    df = load_alerts(args.data_dir, scenarios=scenarios)
    print(f"  {len(df):,} alerts loaded.\n")

    fmt = args.fmt

    def out(name: str) -> str:
        return str(out_dir / f"{name}.{fmt}")

    plots = [
        (
            "alert volume (concatenated timeline)",
            lambda: plot_alert_volume_concatenated(
                df, bin_hours=args.bin_hours, out_path=out("volume_concatenated")
            ),
        ),
        (
            "class balance",
            lambda: plot_class_balance(df, out_path=out("class_balance")),
        ),
        (
            "attack type heatmap",
            lambda: plot_attack_type_heatmap(df, out_path=out("attack_type_heatmap")),
        ),
        (
            "top alert names",
            lambda: plot_top_alert_names(
                df, top_k=args.top_k, out_path=out("top_alert_names")
            ),
        ),
        (
            "inter-arrival time CDF",
            lambda: plot_inter_arrival_time_cdf(df, out_path=out("inter_arrival_cdf")),
        ),
        (
            "group size distribution",
            lambda: plot_group_size_distribution(df, out_path=out("group_size_dist")),
        ),
        (
            "scenario overview table",
            lambda: plot_scenario_overview(df, out_path=out("scenario_overview")),
        ),
    ]

    import matplotlib.pyplot as plt

    for label, fn in plots:
        print(f"  Plotting {label}...", end=" ", flush=True)
        fig, _ = fn()
        plt.close(fig)
        print("done")

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
