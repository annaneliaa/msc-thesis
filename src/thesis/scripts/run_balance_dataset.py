"""
Dataset balancing methods for the AIT-ADS alert dataset.

Methods
-------
naive50
    Randomly undersample the majority class (attack or benign) down to a 50/50
    split.  Output: alerts_filtered_naive50.{csv,json}

type_stratified
    Within the attack class, cap any attack type whose count exceeds the 2nd
    most common attack type's count, so the single dominant type is brought
    in line with the rest.  Benign samples are kept as-is.
    Output: alerts_filtered_type_stratified.{csv,json}

Reads raw alert files:
  data/alerts_csv/<scenario>_alerts.txt

Writes:
  artifacts/processed-data/<scenario>/alerts_filtered_<method>.{csv,json}

Class definition:
  positive (attack) : time_label != 'false_positive'
  negative (benign) : time_label == 'false_positive'

Usage:
  python src/thesis/scripts/run_balance_dataset.py
  python src/thesis/scripts/run_balance_dataset.py --method type_stratified
  python src/thesis/scripts/run_balance_dataset.py --scenarios fox harrison
  python src/thesis/scripts/run_balance_dataset.py --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_ALERTS_DIR = _REPO / "data" / "alerts_csv"
_PROCESSED_DIR = _REPO / "artifacts" / "processed-data"

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


def _save(df: pd.DataFrame, out_dir: Path, method: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"alerts_filtered_{method}.csv"
    out_json = out_dir / f"alerts_filtered_{method}.json"
    df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    return out_csv.name


def balance_naive50(scenario: str, seed: int) -> None:
    src = _ALERTS_DIR / f"{scenario}_alerts.txt"
    if not src.exists():
        print(f"  [{scenario}] SKIP — {src} not found")
        return

    df = pd.read_csv(src)
    attack = df[df["time_label"] != "false_positive"]
    benign = df[df["time_label"] == "false_positive"]

    n_attack, n_benign = len(attack), len(benign)
    minority_n = min(n_attack, n_benign)

    if minority_n == 0:
        print(
            f"  [{scenario}] SKIP — one class is empty (attack={n_attack}, fp={n_benign})"
        )
        return

    balanced = (
        pd.concat(
            [
                attack.sample(n=minority_n, random_state=seed),
                benign.sample(n=minority_n, random_state=seed),
            ]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    fname = _save(balanced, _PROCESSED_DIR / scenario, "naive50")
    majority = "attack" if n_attack >= n_benign else "fp"
    print(
        f"  [{scenario}]  attack={n_attack:>8,}  fp={n_benign:>6,}  "
        f"→  {minority_n:,} each  (undersampled: {majority})  saved → {fname}"
    )


def balance_type_stratified(scenario: str, seed: int) -> None:
    """Cap each attack type at the count of the 2nd most common attack type.

    This prevents the single dominant attack type (typically dirb) from
    overwhelming all others while keeping the full benign set intact.
    """
    src = _ALERTS_DIR / f"{scenario}_alerts.txt"
    if not src.exists():
        print(f"  [{scenario}] SKIP — {src} not found")
        return

    df = pd.read_csv(src)
    attack = df[df["time_label"] != "false_positive"]
    benign = df[df["time_label"] == "false_positive"]

    type_counts = attack["time_label"].value_counts()  # sorted descending

    if len(type_counts) < 2:
        attack_sampled = attack
        target = len(attack)
    else:
        target = int(type_counts.iloc[1])  # count of 2nd most common type
        parts = []
        for _, group in attack.groupby("time_label"):
            if len(group) > target:
                parts.append(group.sample(n=target, random_state=seed))
            else:
                parts.append(group)
        attack_sampled = pd.concat(parts)

    balanced = (
        pd.concat([attack_sampled, benign]).sort_values("time").reset_index(drop=True)
    )

    fname = _save(balanced, _PROCESSED_DIR / scenario, "type_stratified")
    dominant_type = type_counts.index[0]
    dominant_before = int(type_counts.iloc[0])
    dominant_after = min(dominant_before, target)

    print(
        f"  [{scenario}]  attack_in={len(attack):>8,}  fp={len(benign):>6,}  "
        f"top='{dominant_type}'  {dominant_before:,}→{dominant_after:,}  "
        f"target={target:,}  attack_out={len(attack_sampled):,}  saved → {fname}"
    )


METHODS: dict[str, callable] = {
    "naive50": balance_naive50,
    "type_stratified": balance_type_stratified,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance alert dataset via undersampling."
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="naive50",
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
    args = parser.parse_args()

    fn = METHODS[args.method]
    print(f"Balancing method : {args.method}")
    print(f"Random seed      : {args.seed}")
    print(f"Scenarios        : {args.scenarios}\n")

    for scenario in args.scenarios:
        fn(scenario, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
