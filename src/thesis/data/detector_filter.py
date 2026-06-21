"""
Detector prioritization scoring for AIT-ADS (AIT Alert Data Set).

Implements the robustness score s_rob and detection score s_det from:
  Landauer et al. (2024): "Introducing a New Alert Data Set for Multi-Step Attack Analysis"
  Appendix A, Equations 1 & 2.

Inputs:
  - data/alerts_csv/<scenario>_alerts.txt : per-scenario alert files (CSV-formatted)
  - data/ait_ads/labels.csv               : attack phase start/end times per scenario

Output:
  - artifacts/processed-data/detector_scores.csv
  - artifacts/processed-data/<scenario>/alerts_filtered.csv  (one file per scenario)

Usage:
  python src/thesis/detector_filter.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from thesis.paths import ROOT
from thesis.preprocessing.transactions import build_labeled_window_transactions


class _Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Duration of the normal-operation baseline window in seconds (5 hours).
# The paper specifies "a test phase of 5 hours roughly at the same time of
# day as the multi-step attack but one day earlier."
BASELINE_WINDOW_SECONDS = 5 * 3600  # 18000 s


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load labels.csv from the Zenodo AIT-ADS download.
    Returns a DataFrame with columns: scenario, attack, start, end
    where start/end are Unix timestamps (float).
    """
    df = pd.read_csv(labels_path)
    # Ensure numeric timestamps
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    return df


def load_alerts(alerts_dir: Path) -> pd.DataFrame:
    """
    Load all per-scenario alert files from data/alerts_csv/.
    Expects files named <scenario>_alerts.txt (CSV-formatted).

    Columns: time, name, ip, host, short, time_label, event_label
    Renamed: time → timestamp, short → detector
    """
    dfs = []

    for scenario in SCENARIOS:
        path = alerts_dir / f"{scenario}_alerts.txt"
        if not path.exists():
            print(
                f"  Warning: no alerts file found for scenario '{scenario}' at {path}"
            )
            continue

        df = pd.read_csv(path)
        df["scenario"] = scenario
        df = df.rename(columns={"time": "timestamp", "short": "detector"})
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No alert files found in {alerts_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce")
    return combined


# ---------------------------------------------------------------------------
# Step 2: Build baseline windows
# ---------------------------------------------------------------------------


def build_baseline_windows(labels_df: pd.DataFrame) -> dict:
    """
    For each scenario, derive the 5-hour baseline (normal operation) window.

    Paper specification:
      "a test phase of 5 hours roughly at the same time of day as the
       multi-step attack but one day earlier"

    Strategy: take the start of the multi-step attack (network_scans) and
    subtract exactly 24 hours to get the baseline window start, then add
    BASELINE_WINDOW_SECONDS for the end.

    Returns dict: { scenario: (baseline_start, baseline_end) }
    """
    baselines = {}
    ONE_DAY = 86400  # seconds

    for scenario in SCENARIOS:
        scenario_labels = labels_df[labels_df["scenario"] == scenario]
        if scenario_labels.empty:
            continue

        # Anchor on the first attack phase chronologically (network_scans)
        # Some scenarios have service_stop / dnsteal starting earlier —
        # we want the multi-step chain start specifically.
        multistep_phases = [
            "network_scans",
            "service_scans",
            "wpscan",
            "dirb",
            "webshell",
            "cracking",
            "reverse_shell",
            "privilege_escalation",
        ]
        multistep = scenario_labels[scenario_labels["attack"].isin(multistep_phases)]

        if multistep.empty:
            # Fall back to overall earliest start
            attack_start = scenario_labels["start"].min()
        else:
            attack_start = multistep["start"].min()

        baseline_start = attack_start - ONE_DAY
        baseline_end = baseline_start + BASELINE_WINDOW_SECONDS
        baselines[scenario] = (baseline_start, baseline_end)

    return baselines


# ---------------------------------------------------------------------------
# Step 3: Compute robustness score per (attack_phase, detector)
# ---------------------------------------------------------------------------


def compute_robustness_scores(
    alerts_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    baselines: dict,
) -> pd.DataFrame:
    """
    Compute s_rob(A, D) for every (attack_phase, detector) pair.

    s_rob(A, D) = (1 / |S|) * sum_S [
        1 - min(1, (alerts_in_baseline / alerts_in_attack) * (delta_A / delta_T))
    ]

    Where:
      - delta_A = duration of attack phase A in scenario S
      - delta_T = duration of baseline window (BASELINE_WINDOW_SECONDS, constant)
      - If a detector fires 0 alerts during the attack phase in a scenario,
        that (scenario, phase) contributes 0 to the sum (detector missed the attack).

    Returns a DataFrame with columns:
      attack, detector, s_rob, n_scenarios_with_attack
    """
    attack_phases = labels_df["attack"].unique()
    detectors = alerts_df["detector"].unique()
    records = []

    for attack in attack_phases:
        phase_labels = labels_df[labels_df["attack"] == attack]
        # Scenarios where this attack phase exists
        attack_scenarios = phase_labels["scenario"].tolist()

        for detector in detectors:
            det_alerts = alerts_df[alerts_df["detector"] == detector]
            robustness_values = []

            for scenario in attack_scenarios:
                if scenario not in baselines:
                    continue

                row = phase_labels[phase_labels["scenario"] == scenario]
                if row.empty:
                    continue

                delta_A = float(row["end"].values[0]) - float(row["start"].values[0])
                delta_T = BASELINE_WINDOW_SECONDS

                if delta_A <= 0:
                    continue

                scen_alerts = det_alerts[det_alerts["scenario"] == scenario]

                # Count alerts inside the attack phase window
                n_attack = (
                    (scen_alerts["timestamp"] >= row["start"].values[0])
                    & (scen_alerts["timestamp"] <= row["end"].values[0])
                ).sum()

                if n_attack == 0:
                    # Detector did not fire during attack — contributes 0
                    robustness_values.append(0.0)
                    continue

                # Count alerts inside the baseline window
                b_start, b_end = baselines[scenario]
                n_baseline = (
                    (scen_alerts["timestamp"] >= b_start)
                    & (scen_alerts["timestamp"] <= b_end)
                ).sum()

                ratio = (n_baseline / n_attack) * (delta_A / delta_T)
                s_rob_contribution = 1.0 - min(1.0, ratio)
                robustness_values.append(s_rob_contribution)

            if robustness_values:
                s_rob = float(np.mean(robustness_values))
            else:
                s_rob = 0.0

            records.append(
                {
                    "attack": attack,
                    "detector": detector,
                    "s_rob": s_rob,
                    "n_scenarios_with_attack": len(attack_scenarios),
                    "n_scenarios_detector_fired": sum(
                        1 for v in robustness_values if v > 0
                    ),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 4: Compute detection score per detector
# ---------------------------------------------------------------------------


def compute_detection_scores(robustness_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute s_det(D) for every detector.

    s_det(D) = max_A [
        s_rob(A, D) * (scenarios_where_detector_fired / scenarios_where_attack_exists)
    ]

    Returns a DataFrame with columns:
      detector, s_det, best_attack, s_rob_best
    sorted descending by s_det.
    """
    records = []

    for detector, group in robustness_df.groupby("detector"):
        best_score = 0.0
        best_attack = None
        best_s_rob = 0.0

        for _, row in group.iterrows():
            n_total = row["n_scenarios_with_attack"]
            n_fired = row["n_scenarios_detector_fired"]

            if n_total == 0:
                continue

            coverage = n_fired / n_total
            s_det_candidate = row["s_rob"] * coverage

            if s_det_candidate > best_score:
                best_score = s_det_candidate
                best_attack = row["attack"]
                best_s_rob = row["s_rob"]

        records.append(
            {
                "detector": detector,
                "s_det": round(best_score, 4),
                "best_attack": best_attack,
                "s_rob_best": round(best_s_rob, 4),
            }
        )

    result = pd.DataFrame(records).sort_values("s_det", ascending=False)
    result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Step 5: Filter alerts by threshold (keeping full timeline)
# ---------------------------------------------------------------------------


def filter_by_score(
    alerts_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Keep only alerts from detectors with s_det >= threshold.
    The full timeline is preserved — benign/false-positive alerts are kept.
    Only low-signal detectors are removed.

    Parameters
    ----------
    alerts_df  : full alert DataFrame loaded by load_alerts()
    scores_df  : output of compute_detection_scores()
    threshold  : s_det cutoff; paper uses 0.7

    Returns
    -------
    Filtered DataFrame with same columns as alerts_df.
    """
    keep = scores_df[scores_df["s_det"] >= threshold]["detector"].tolist()
    filtered = alerts_df[alerts_df["detector"].isin(keep)].copy()

    n_before = len(alerts_df)
    n_after = len(filtered)
    n_det_before = alerts_df["detector"].nunique()
    n_det_after = filtered["detector"].nunique()

    print(f"Threshold: {threshold}")
    print(f"  Detectors : {n_det_before} -> {n_det_after} kept")
    print(
        f"  Alerts    : {n_before:,} -> {n_after:,} kept "
        f"({100 * (1 - n_after / n_before):.1f}% removed)"
    )
    return filtered


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_detector_scores(
    alerts_dir: Path,
    labels_path: Path,
) -> pd.DataFrame:
    """
    Full pipeline: load data, build baselines, compute and return scores.

    Parameters
    ----------
    alerts_dir  : path to data/alerts_csv/ (contains per-scenario _alerts.txt files)
    labels_path : path to data/ait_ads/labels.csv

    Returns
    -------
    scores_df : DataFrame with columns [detector, s_det, best_attack, s_rob_best]
                sorted descending by s_det
    """
    print("Loading labels...")
    labels_df = load_labels(labels_path)

    print("Loading alerts...")
    alerts_df = load_alerts(alerts_dir)

    print("Building baseline windows...")
    baselines = build_baseline_windows(labels_df)
    for s, (start, end) in baselines.items():
        print(
            f"  {s}: baseline [{pd.Timestamp(start, unit='s')} -> "
            f"{pd.Timestamp(end, unit='s')}]"
        )

    print("Computing robustness scores...")
    robustness_df = compute_robustness_scores(alerts_df, labels_df, baselines)

    print("Computing detection scores...")
    scores_df = compute_detection_scores(robustness_df)

    return scores_df, alerts_df


# ---------------------------------------------------------------------------
# Threshold sweep logging
# ---------------------------------------------------------------------------


def _log_threshold_sweep(
    scores_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    taus: list[float],
) -> None:
    """
    For each threshold in taus, print:
      - Full per-detector score table annotated with KEEP / drop status
      - Alert impact broken down by time_label (benign vs attack phases)
    """
    all_detectors = scores_df.sort_values("s_det", ascending=False)
    n_alerts_total = len(alerts_df)
    label_totals = alerts_df["time_label"].value_counts().sort_index()

    sep = "=" * 72

    print("\n" + sep)
    print("  THRESHOLD SWEEP — full per-detector scoring")
    print(sep)

    for tau in taus:
        kept_set = set(scores_df[scores_df["s_det"] >= tau]["detector"])

        print(f"\n{'─'*72}")
        print(
            f"  tau = {tau:.1f}   ({len(kept_set)} detectors kept / {len(all_detectors)} total)"
        )
        print(f"{'─'*72}")

        # Per-detector table
        print(f"  {'detector':<30} {'s_det':>6}  {'best_attack':<25} {'status':>6}")
        print(f"  {'-'*30} {'-'*6}  {'-'*25} {'-'*6}")
        for _, row in all_detectors.iterrows():
            status = "KEEP" if row["detector"] in kept_set else "drop"
            print(
                f"  {row['detector']:<30} {row['s_det']:>6.4f}  "
                f"{str(row['best_attack']):<25} {status:>6}"
            )

        # Overall alert counts
        kept_alerts = alerts_df[alerts_df["detector"].isin(kept_set)]
        n_kept = len(kept_alerts)
        pct_removed = 100 * (1 - n_kept / n_alerts_total) if n_alerts_total > 0 else 0.0
        print(f"\n  Alerts total : {n_alerts_total:,}")
        print(
            f"  Alerts kept  : {n_kept:,}  ({100 - pct_removed:.1f}% retained, {pct_removed:.1f}% removed)"
        )

        # Per time_label breakdown: how much of each label is removed
        print(
            f"\n  {'time_label':<20} {'total':>8} {'kept':>8} {'removed':>8} {'% removed':>10}"
        )
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        kept_label_counts = kept_alerts["time_label"].value_counts().sort_index()
        for label, total in label_totals.items():
            kept = kept_label_counts.get(label, 0)
            removed = total - kept
            pct = 100 * removed / total if total > 0 else 0.0
            print(
                f"  {str(label):<20} {total:>8,} {kept:>8,} {removed:>8,} {pct:>9.1f}%"
            )

    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ALERTS_DIR = ROOT / "data" / "alerts_csv"
    LABELS_PATH = ROOT / "data" / "ait_ads" / "labels.csv"
    PROCESSED_DATA_DIR = ROOT / "artifacts" / "processed-data"
    TRANSACTIONS_DIR = ROOT / "artifacts" / "transactions" / "detector_filtered"
    THRESHOLD = 0.7

    log_path = PROCESSED_DATA_DIR / "detector_filter.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    try:
        scores_df, alerts_df = compute_detector_scores(ALERTS_DIR, LABELS_PATH)

        print("\n--- Detector scores ---")
        print(scores_df.to_string(index=False))

        print(f"\n--- Filtering at threshold {THRESHOLD} ---")
        filtered_df = filter_by_score(alerts_df, scores_df, threshold=THRESHOLD)

        # Per-scenario reduction rates
        print("\n--- Reduction rate per scenario ---")
        per_scenario_rates = []
        for scenario in SCENARIOS:
            scen_before = alerts_df[alerts_df["scenario"] == scenario]
            scen_after = filtered_df[filtered_df["scenario"] == scenario]
            n_before = len(scen_before)
            n_after = len(scen_after)
            if n_before > 0:
                rate = 1 - (n_after / n_before)
                per_scenario_rates.append(rate)
                print(f"  {scenario}: {rate*100:.2f}% ({n_before - n_after:,} removed)")
                removed = scen_before[~scen_before.index.isin(scen_after.index)]
                label_counts = removed["time_label"].value_counts()
                label_totals = scen_before["time_label"].value_counts()
                for label, count in label_counts.items():
                    total = label_totals.get(label, 0)
                    print(f"    {label}: {count:,} / {total:,}")

        avg_reduction = np.mean(per_scenario_rates)
        print(f"\nAverage reduction rate: {avg_reduction*100:.2f}%")

        # Save detector scores at the top of processed-data/
        scores_out = PROCESSED_DATA_DIR / "detector_scores.csv"
        scores_df.to_csv(scores_out, index=False)
        print(f"\nSaved: {scores_out}")

        # Save filtered alerts per scenario (CSV + JSON matching original alerts.json format)
        import json as _json

        for scenario, group in filtered_df.groupby("scenario"):
            original = group.rename(columns={"timestamp": "time", "detector": "short"})[
                ["time", "name", "ip", "host", "short", "time_label", "event_label"]
            ].assign(time=lambda d: d["time"].astype(int))
            out_csv = PROCESSED_DATA_DIR / scenario / "alerts_filtered.csv"
            original.to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")

            records = original.to_dict(orient="records")
            out_json = PROCESSED_DATA_DIR / scenario / "alerts_filtered.json"
            with out_json.open("w", encoding="utf-8") as f:
                _json.dump(records, f, indent=2)
            print(f"Saved: {out_json}")

        # Build and save window transactions per scenario
        TRANSACTIONS_DIR.mkdir(parents=True, exist_ok=True)
        for scenario, group in filtered_df.groupby("scenario"):
            tx_alerts = group.rename(
                columns={"timestamp": "time", "detector": "short"}
            )[
                ["time", "name", "ip", "host", "short", "time_label", "event_label"]
            ].assign(time=lambda d: d["time"].astype(int))
            transactions = build_labeled_window_transactions(tx_alerts)
            out_tx = TRANSACTIONS_DIR / f"{scenario}_transactions.csv"
            transactions.to_csv(out_tx, index=False)
            print(f"Saved: {out_tx}")

        # Threshold sweep — full per-detector breakdown at each tau
        _log_threshold_sweep(
            scores_df, alerts_df, taus=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        print(f"\nLog saved to: {log_path}")
    finally:
        sys.stdout = sys.__stdout__
        _log_file.close()
