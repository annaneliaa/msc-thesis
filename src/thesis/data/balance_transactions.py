"""Transaction-level undersampling methods for the AIT-ADS dataset."""

from __future__ import annotations

import pandas as pd


def naive50(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    attack = df[df["tx_label"] == "attack"]
    benign = df[df["tx_label"] == "benign"]
    n_attack, n_benign = len(attack), len(benign)
    minority_n = min(n_attack, n_benign)

    if minority_n == 0:
        raise ValueError(f"one class is empty (attack={n_attack}, benign={n_benign})")

    balanced = (
        pd.concat(
            [
                attack.sample(n=minority_n, random_state=seed),
                benign.sample(n=minority_n, random_state=seed),
            ]
        )
        .sort_values("window_start")
        .reset_index(drop=True)
    )
    majority = "attack" if n_attack >= n_benign else "benign"
    return balanced, {
        "n_attack": n_attack,
        "n_benign": n_benign,
        "minority_n": minority_n,
        "undersampled": majority,
    }


METHODS: dict[str, callable] = {
    "naive50": naive50,
}
