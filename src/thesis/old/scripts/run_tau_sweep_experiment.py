"""
Sweep detector filtering thresholds (tau) to measure effect of noise level on FP rate.

For each tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
  1. Load pre-computed detector scores from detector_filter.py
  2. Filter alerts at that threshold per scenario
  3. Save alerts_filtered_tau_X.json under artifacts/processed-data/<scenario>/
  4. Run run_compare_scenarios.py with the tau-filtered data (--filtered flag)
  5. Results go to artifacts/experiments/run_compare/compare_tau_X_<timestamp>/

After all tau values complete, generate a summary plot showing FP rate vs tau for each scenario.

Outputs:
  artifacts/processed-data/<scenario>/alerts_filtered_tau_X.json
  artifacts/experiments/run_compare/compare_tau_X_TIMESTAMP/
  artifacts/experiments/tau_sweep_summary/
    - fp_rate_by_tau.png         -- FP rate per scenario, lines for each tau
    - benign_ratio_by_tau.png    -- benign ratio per scenario, lines for each tau
    - tau_sweep_summary.csv       -- aggregated results

Usage:
    python src/thesis/scripts/run_tau_sweep_experiment.py
    python src/thesis/scripts/run_tau_sweep_experiment.py --taus 0.5,0.6,0.7
    python src/thesis/scripts/run_tau_sweep_experiment.py --force
    python src/thesis/scripts/run_tau_sweep_experiment.py --no-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib

matplotlib.use("Agg")

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_PROCESSED_DATA = _REPO / "artifacts" / "processed-data"
_RUN_COMPARE = _REPO / "artifacts" / "experiments" / "run_compare"
_OUT_DIR = _REPO / "artifacts" / "experiments" / "tau_sweep_summary"

# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------


def load_detector_scores() -> pd.DataFrame:
    """Load pre-computed detector scores from detector_filter.py output."""
    scores_csv = _PROCESSED_DATA / "detector_scores.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(
            f"Detector scores not found at {scores_csv}. "
            f"Run detector_filter.py first."
        )
    return pd.read_csv(scores_csv)


def load_raw_alerts() -> pd.DataFrame:
    """Load raw alerts.json files from all scenarios."""
    from thesis.old.data.detector_filter import load_alerts

    alerts_dir = _REPO / "data" / "alerts_csv"
    return load_alerts(alerts_dir)


def filter_and_save_alerts(
    alerts_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    threshold: float,
    tau_id: str,
) -> dict[str, int]:
    """
    Filter alerts at the given threshold and save to artifacts/processed-data/<scenario>/
    as alerts_filtered_tau_X.json (alongside the existing alerts_filtered.json).

    Returns dict: {scenario: num_alerts_after_filter}
    """
    keep_detectors = set(scores_df[scores_df["s_det"] >= threshold]["detector"])
    filtered = alerts_df[alerts_df["detector"].isin(keep_detectors)].copy()

    results = {}
    for scenario in alerts_df["scenario"].unique():
        scen_alerts = filtered[filtered["scenario"] == scenario]
        n_after = len(scen_alerts)
        results[scenario] = n_after

        out_dir = _PROCESSED_DATA / scenario
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON (format matching alerts.json)
        records = (
            scen_alerts.rename(columns={"timestamp": "time", "detector": "short"})[
                ["time", "name", "ip", "host", "short", "time_label", "event_label"]
            ]
            .assign(time=lambda d: d["time"].astype(int))
            .to_dict(orient="records")
        )
        out_json = out_dir / f"alerts_filtered_{tau_id}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Running compare_scenarios for a tau
# ---------------------------------------------------------------------------


def run_compare_for_tau(
    tau: float,
    tau_id: str,
    force: bool = False,
) -> str:
    """
    Call run_compare_scenarios.py with the tau-filtered alerts.

    Since run_compare_scenarios.py --filtered looks for alerts_filtered.json,
    we temporarily rename alerts_filtered_tau_X.json to alerts_filtered.json,
    run the compare, then restore.

    Returns the compare run directory name (compare_*_TIMESTAMP).
    """
    from thesis.visualization.eda import SCENARIOS

    # Step 1: Temporarily rename tau-filtered files to alerts_filtered.json
    backup_files = {}
    for scenario in SCENARIOS:
        scenario_dir = _PROCESSED_DATA / scenario
        tau_file = scenario_dir / f"alerts_filtered_{tau_id}.json"
        standard_file = scenario_dir / "alerts_filtered.json"

        if tau_file.exists():
            if standard_file.exists():
                backup_files[scenario] = standard_file.read_text()
            tau_file.rename(standard_file)

    try:
        # Step 2: Run compare_scenarios with --filtered
        cmd = [
            "python",
            str(_REPO / "src" / "thesis" / "scripts" / "run_compare_scenarios.py"),
            "--all",
            "--filtered",
        ]
        if force:
            cmd.append("--force")

        print(f"\n[tau={tau:.1f}] Running compare_scenarios...")
        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
        if result.returncode != 0:
            print("  ERROR: compare_scenarios failed!")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"compare_scenarios failed for tau={tau}")

        # Step 3: Extract run directory name from output
        output_lines = result.stdout.strip().split("\n")
        run_dir_name = None
        for line in reversed(output_lines):
            if "compare_" in line and "artifacts" in line:
                parts = line.split("/")
                for p in parts:
                    if p.startswith("compare_"):
                        run_dir_name = p
                        break
                if run_dir_name:
                    break

        if not run_dir_name:
            raise RuntimeError(
                f"Could not extract run dir from compare_scenarios output for tau={tau}"
            )

        return run_dir_name

    finally:
        # Step 4: Restore original alerts_filtered.json files
        for scenario in SCENARIOS:
            scenario_dir = _PROCESSED_DATA / scenario
            standard_file = scenario_dir / "alerts_filtered.json"
            tau_file = scenario_dir / f"alerts_filtered_{tau_id}.json"

            if standard_file.exists() and not tau_file.exists():
                standard_file.rename(tau_file)

            if scenario in backup_files:
                standard_file.write_text(backup_files[scenario])


# ---------------------------------------------------------------------------
# Loading results and aggregating
# ---------------------------------------------------------------------------


def load_compare_results_for_tau(tau: float, run_dir_pattern: str) -> list[dict]:
    """Load all scenario results from a tau run directory."""

    run_dir = _RUN_COMPARE / run_dir_pattern
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    records = []
    scenario_root = run_dir / "scenario"

    for scenario_dir in sorted(scenario_root.iterdir()):
        compare_files = sorted(scenario_dir.glob("compare_*.json"))
        if not compare_files:
            continue

        data = json.loads(compare_files[-1].read_text())
        scenario = data["scenario"]

        for model_type in ("baseline", "symbolic"):
            m = data[model_type]["metrics"]
            tp, fp, tn, fn = m.get("tp"), m.get("fp"), m.get("tn"), m.get("fn")
            if any(v is None for v in (tp, fp, tn, fn)):
                continue

            benign = fp + tn
            fp_rate = fp / benign if benign > 0 else float("nan")
            benign_ratio = benign / (tp + fp + tn + fn)

            records.append(
                {
                    "tau": tau,
                    "scenario": scenario,
                    "model_type": model_type,
                    "fp_rate": fp_rate,
                    "benign_ratio": benign_ratio,
                    "fp": fp,
                    "tn": tn,
                    "n_features": data[model_type]["n_features"],
                    "n_alert_groups": data[model_type]["n_alert_groups"],
                }
            )

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_tau_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot FP rate and benign ratio vs tau for each scenario."""
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    colors = {"baseline": "#1f77b4", "symbolic": "#ff7f0e"}
    markers = {"baseline": "o", "symbolic": "s"}

    # FP rate vs tau
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in scenarios:
        for model_type in ("baseline", "symbolic"):
            subset = df[
                (df["scenario"] == scenario) & (df["model_type"] == model_type)
            ].sort_values("tau")
            if subset.empty:
                continue
            ax.plot(
                subset["tau"],
                subset["fp_rate"],
                marker=markers[model_type],
                color=colors[model_type],
                label=f"{scenario}·{model_type}",
                alpha=0.7,
                linewidth=1.5,
            )

    ax.set_xlabel("Detection score threshold (tau)")
    ax.set_ylabel("FP rate  [FP / (FP + TN)]")
    ax.set_title("FP rate vs detector filtering threshold (tau sweep)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    out = out_dir / "fp_rate_by_tau.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

    # Benign ratio vs tau
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in scenarios:
        for model_type in ("baseline", "symbolic"):
            subset = df[
                (df["scenario"] == scenario) & (df["model_type"] == model_type)
            ].sort_values("tau")
            if subset.empty:
                continue
            ax.plot(
                subset["tau"],
                subset["benign_ratio"],
                marker=markers[model_type],
                color=colors[model_type],
                label=f"{scenario}·{model_type}",
                alpha=0.7,
                linewidth=1.5,
            )

    ax.set_xlabel("Detection score threshold (tau)")
    ax.set_ylabel("Benign ratio (test set)")
    ax.set_title("Benign traffic ratio vs detector filtering threshold (tau sweep)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    out = out_dir / "benign_ratio_by_tau.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep detector filtering thresholds (tau) and measure FP rate effect."
    )
    parser.add_argument(
        "--taus",
        default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated list of thresholds to sweep (default: 0.3,0.4,0.5,0.6,0.7,0.8,0.9).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run compare_scenarios even if results exist.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running; only load existing results and (re-)plot.",
    )
    args = parser.parse_args()

    taus = [float(t.strip()) for t in args.taus.split(",")]
    taus = sorted(taus)

    print("=" * 72)
    print("  TAU SWEEP EXPERIMENT")
    print(f"  Thresholds: {taus}")
    print(f"  Force re-run: {args.force}")
    print(f"  No-run mode: {args.no_run}")
    print("=" * 72)

    print("\nLoading detector scores...")
    scores_df = load_detector_scores()
    print(f"  {len(scores_df)} detectors loaded")

    print("\nLoading raw alerts...")
    alerts_df = load_raw_alerts()
    print(
        f"  {len(alerts_df):,} alerts across {alerts_df['scenario'].nunique()} scenarios"
    )

    all_results = []

    for tau in taus:
        tau_id = f"tau_{tau:.1f}".replace(".", "_")
        print(f"\n{'─'*72}")
        print(f"  Processing tau = {tau:.1f}")
        print(f"{'─'*72}")

        if not args.no_run:
            # Filter and save
            print(f"  Filtering and saving alerts_filtered_{tau_id}.json...")
            filter_counts = filter_and_save_alerts(alerts_df, scores_df, tau, tau_id)
            for scenario, n_after in filter_counts.items():
                print(f"    {scenario}: {n_after:,} alerts")

            # Run compare_scenarios
            run_dir_name = run_compare_for_tau(tau, tau_id, force=args.force)
            print(f"  Results saved to: compare_{run_dir_name}")
        else:
            # Try to find existing run directory for this tau
            pattern = f"compare_*{tau_id}*"
            matches = sorted(_RUN_COMPARE.glob(pattern), reverse=True)
            if not matches:
                print(
                    f"  [--no-run] No existing results found for {pattern}. Skipping."
                )
                continue
            run_dir_name = matches[0].name

        # Load results
        print(f"  Loading results from {run_dir_name}...")
        tau_results = load_compare_results_for_tau(tau, run_dir_name)
        all_results.extend(tau_results)
        print(f"    {len(tau_results)} records loaded")

    if not all_results:
        print("\nNo results to process. Exiting.")
        return

    # Aggregate and plot
    df = pd.DataFrame(all_results)
    print(f"\n{'─'*72}")
    print("  AGGREGATED RESULTS")
    print(f"{'─'*72}")
    print(f"  Total records: {len(df)}")
    print(f"  Scenarios: {df['scenario'].nunique()}")
    print(f"  Tau values: {sorted(df['tau'].unique())}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[plots] Saving to {_OUT_DIR}")
    plot_tau_sweep(df, _OUT_DIR)

    csv_out = _OUT_DIR / "tau_sweep_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"  Saved → {csv_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
