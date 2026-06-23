"""Alert-level undersampling methods for the AIT-ADS dataset."""

from __future__ import annotations

import pandas as pd


def naive50(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    attack = df[df["time_label"] != "false_positive"]
    benign = df[df["time_label"] == "false_positive"]
    n_attack, n_benign = len(attack), len(benign)
    minority_n = min(n_attack, n_benign)

    if minority_n == 0:
        raise ValueError(f"one class is empty (attack={n_attack}, fp={n_benign})")

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
    majority = "attack" if n_attack >= n_benign else "fp"
    return balanced, {
        "n_attack": n_attack,
        "n_benign": n_benign,
        "minority_n": minority_n,
        "undersampled": majority,
    }


def type_stratified(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Cap each attack type at the count of the 2nd most common attack type.

    Prevents a single dominant attack type from overwhelming the rest while
    keeping the full benign set intact.
    """
    attack = df[df["time_label"] != "false_positive"]
    benign = df[df["time_label"] == "false_positive"]
    type_counts = attack["time_label"].value_counts()

    if len(type_counts) < 2:
        attack_sampled = attack
        target = len(attack)
    else:
        target = int(type_counts.iloc[1])
        parts = []
        for _, group in attack.groupby("time_label"):
            parts.append(
                group.sample(n=target, random_state=seed)
                if len(group) > target
                else group
            )
        attack_sampled = pd.concat(parts)

    balanced = (
        pd.concat([attack_sampled, benign]).sort_values("time").reset_index(drop=True)
    )
    dominant_type = type_counts.index[0]
    dominant_before = int(type_counts.iloc[0])
    return balanced, {
        "n_attack_in": len(attack),
        "n_benign": len(benign),
        "dominant_type": dominant_type,
        "dominant_before": dominant_before,
        "dominant_after": min(dominant_before, target),
        "target": target,
        "n_attack_out": len(attack_sampled),
    }


METHODS: dict[str, callable] = {
    "naive50": naive50,
    "type_stratified": type_stratified,
}
