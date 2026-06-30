"""
Balance the AIT-ADS dataset via undersampling, at either the alert or alert_group level.

Levels
------
alert
    Reads raw alert files from data/alerts_csv/<scenario>_alerts.txt.
    Writes to artifacts/alerts/balanced/<method>/<scenario>_alerts.{csv,json}.

alert_groups
    Reads raw alert files from data/alerts_csv/<scenario>_alerts.txt, builds alert_groups
    using a configurable window size, then balances them.
    Writes to artifacts/alert_groups/balanced/<method>/<scenario>_alert_groups.{csv,json}.

Methods (alert level)
---------------------
naive50
    Randomly undersample the majority class down to a 50/50 split.

type_stratified
    Cap each attack type at the count of the 2nd most common attack type,
    keeping all benign samples intact.

Methods (alert_group level)
---------------------------
naive50
    Randomly undersample the majority class down to a 50/50 split.

Usage
-----
  python src/thesis/scripts/run_balance_dataset.py --level alert
  python src/thesis/scripts/run_balance_dataset.py --level alert --method type_stratified
  python src/thesis/scripts/run_balance_dataset.py --level alert_groups
  python src/thesis/scripts/run_balance_dataset.py --level alert_groups --window-size 5
  python src/thesis/scripts/run_balance_dataset.py --level alert --scenarios fox harrison
  python src/thesis/scripts/run_balance_dataset.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from thesis.data.balance_alerts import METHODS as ALERT_METHODS
from thesis.data.balance_alert_groups import METHODS as TX_METHODS
from thesis.preprocessing.alert_groups import build_labeled_window_alert_groups

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_ALERTS_DIR = _REPO / "data" / "alerts_csv"
_ALERTS_BALANCED_DIR = _REPO / "artifacts" / "alerts" / "balanced"
_TX_BALANCED_DIR = _REPO / "artifacts" / "alert_groups" / "balanced"
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_balance_dataset"


class _Tee:
    """Write to both a file and the original stdout simultaneously."""

    def __init__(self, log_path: Path) -> None:
        self._file = log_path.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)
        return self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


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


def _save(df: pd.DataFrame, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}.csv"
    out_json = out_dir / f"{stem}.json"
    df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            df.to_dict(orient="records"),
            f,
            indent=2,
            default=lambda o: sorted(o) if isinstance(o, set) else str(o),
        )
    return out_csv.name


def run_alert(scenario: str, method: str, seed: int) -> None:
    src = _ALERTS_DIR / f"{scenario}_alerts.txt"
    if not src.exists():
        print(f"  [{scenario}] SKIP — {src} not found")
        return

    df = pd.read_csv(src)
    fn = ALERT_METHODS[method]
    try:
        balanced, info = fn(df, seed=seed)
    except ValueError as e:
        print(f"  [{scenario}] SKIP — {e}")
        return

    out_dir = _ALERTS_BALANCED_DIR / method
    fname = _save(balanced, out_dir, f"{scenario}_alerts")

    if method == "naive50":
        majority = info["undersampled"]
        print(
            f"  [{scenario}]  attack={info['n_attack']:>8,}  fp={info['n_benign']:>6,}  "
            f"→  {info['minority_n']:,} each  (undersampled: {majority})  saved → {fname}"
        )
    elif method == "type_stratified":
        print(
            f"  [{scenario}]  attack_in={info['n_attack_in']:>8,}  fp={info['n_benign']:>6,}  "
            f"top='{info['dominant_type']}'  {info['dominant_before']:,}→{info['dominant_after']:,}  "
            f"target={info['target']:,}  attack_out={info['n_attack_out']:,}  saved → {fname}"
        )
    else:
        print(f"  [{scenario}] saved → {fname}")


def run_alert_groups(scenario: str, method: str, seed: int, window_size: int) -> None:
    src = _ALERTS_DIR / f"{scenario}_alerts.txt"
    if not src.exists():
        print(f"  [{scenario}] SKIP — {src} not found")
        return

    alerts = pd.read_csv(src)
    df = build_labeled_window_alert_groups(alerts, window_size_s=window_size)
    fn = TX_METHODS[method]
    try:
        balanced, info = fn(df, seed=seed)
    except ValueError as e:
        print(f"  [{scenario}] SKIP — {e}")
        return

    out_dir = _TX_BALANCED_DIR / method
    fname = _save(balanced, out_dir, f"{scenario}_alert_groups")

    if method == "naive50":
        majority = info["undersampled"]
        print(
            f"  [{scenario}]  attack={info['n_attack']:>6,}  benign={info['n_benign']:>6,}  "
            f"→  {info['minority_n']:,} each  (undersampled: {majority})  saved → {fname}"
        )
    else:
        print(f"  [{scenario}] saved → {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance AIT-ADS dataset via undersampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level",
        choices=["alert", "alert_groups"],
        required=True,
        help="Whether to balance at the alert or alert_group level.",
    )
    parser.add_argument(
        "--method",
        default="naive50",
        metavar="METHOD",
        help="Balancing method (default: naive50).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=SCENARIOS,
        metavar="SCENARIO",
        help="Scenarios to process (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        metavar="SECONDS",
        help="AlertGroup window size in seconds, used when --level alert_groups (default: 2).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _EXPERIMENTS_DIR / f"{timestamp}_{args.level}_{args.method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(run_dir / "run.log")
    sys.stdout = tee

    available = ALERT_METHODS if args.level == "alert" else TX_METHODS
    if args.method not in available:
        parser.error(
            f"Method '{args.method}' is not available for level '{args.level}'. "
            f"Available: {', '.join(available)}"
        )

    print(f"Level            : {args.level}")
    print(f"Balancing method : {args.method}")
    print(f"Random seed      : {args.seed}")
    if args.level == "alert_groups":
        print(f"Window size      : {args.window_size}s")
    print(f"Scenarios        : {args.scenarios}\n")

    for scenario in args.scenarios:
        if args.level == "alert":
            run_alert(scenario, args.method, args.seed)
        else:
            run_alert_groups(scenario, args.method, args.seed, args.window_size)

    print("\nDone.")
    print(f"\nLog saved → {run_dir / 'run.log'}")
    tee.close()


if __name__ == "__main__":
    main()
