import pandas as pd
from pathlib import Path


def support_in_group(
    transactions: list[frozenset[str]],
    itemset: tuple[str, ...],
) -> tuple[int, float]:
    """
    Count how many transactions contain the itemset.
    """
    if not transactions:
        return 0, 0.0

    itemset_set = set(itemset)
    count = sum(1 for basket in transactions if itemset_set.issubset(basket))
    support = count / len(transactions)
    return count, support


def add_cross_label_supports(
    mined_df: pd.DataFrame,
    target_transactions: list[frozenset[str]],
    other_transactions: list[frozenset[str]],
    target_label: str,
    other_label: str,
) -> pd.DataFrame:
    """
    For each mined itemset, compute support in both target and other label groups.
    """
    print("Calculating cross-label supports...")
    if mined_df.empty:
        return mined_df.copy()

    out = mined_df.copy()

    target_counts = []
    target_supports = []
    other_counts = []
    other_supports = []

    for itemset in out["itemset"]:
        c_t, s_t = support_in_group(target_transactions, itemset)
        c_o, s_o = support_in_group(other_transactions, itemset)

        target_counts.append(c_t)
        target_supports.append(s_t)
        other_counts.append(c_o)
        other_supports.append(s_o)

    out[f"count_{target_label}"] = target_counts
    out[f"support_{target_label}"] = target_supports
    out[f"count_{other_label}"] = other_counts
    out[f"support_{other_label}"] = other_supports
    out["support_diff"] = out[f"support_{target_label}"] - out[f"support_{other_label}"]

    return out


def add_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating confidence scores...")
    out = df.copy()

    denom = out["support_attack"] + out["support_benign"]

    out["confidence_attack"] = (out["support_attack"] / denom.replace(0, pd.NA)).fillna(
        0.0
    )

    out["confidence_benign"] = (out["support_benign"] / denom.replace(0, pd.NA)).fillna(
        0.0
    )

    return out


def sort_itemsets_for_class(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    sorted_df = df.copy()
    conf_string = f"confidence_{target_class}"
    support_string = f"support_{target_class}"

    return sorted_df.sort_values(
        [conf_string, support_string, "k"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def filter_itemsets(
    df: pd.DataFrame,
    *,
    min_k: int | None = None,
    max_k: int | None = None,
    min_total_count: int = 10,
    min_support_benign: float = 0.0,
    min_support_attack: float = 0.0,
    min_confidence_benign: float = 0.0,
    min_confidence_attack: float = 0.0,
    min_abs_support_diff: float = 0.0,
    keep_direction: str = "both",
) -> pd.DataFrame:
    """
    Filter mined itemsets using already-computed class counts/supports.

    keep_direction:
        - "both": keep benign-leaning and attack-leaning itemsets
        - "benign": only keep itemsets where support_benign > support_attack
        - "attack": only keep itemsets where support_attack > support_benign
    """
    out = df.copy()

    mask = pd.Series(True, index=out.index)

    if min_k is not None:
        mask &= out["k"] >= min_k
    if max_k is not None:
        mask &= out["k"] <= max_k

    mask &= out["support_count"] >= min_total_count
    mask &= out["support_benign"] >= min_support_benign
    mask &= out["support_attack"] >= min_support_attack
    mask &= out["confidence_benign"] >= min_confidence_benign
    mask &= out["confidence_attack"] >= min_confidence_attack
    mask &= out["support_diff"].abs() >= min_abs_support_diff

    if keep_direction == "benign":
        mask &= out["support_benign"] > out["support_attack"]
    elif keep_direction == "attack":
        mask &= out["support_attack"] > out["support_benign"]
    elif keep_direction != "both":
        raise ValueError(
            f"Invalid keep_direction={keep_direction!r}. "
            "Expected one of: 'both', 'benign', 'attack'."
        )

    return out.loc[mask].copy().reset_index(drop=True)


def select_top_itemsets_per_class(
    df: pd.DataFrame,
    *,
    top_n_benign: int = 100,
    top_n_attack: int = 100,
    min_total_count: int = 10,
    min_abs_support_diff: float = 0.01,
    min_confidence: float = 0.6,
    min_k: int | None = None,
    max_k: int | None = None,
) -> pd.DataFrame:
    """
    Select top discriminative itemsets for both classes.

    This is intended for building a feature vocabulary for a downstream model.
    """
    scored = df.copy()

    benign = filter_itemsets(
        scored,
        min_k=min_k,
        max_k=max_k,
        min_total_count=min_total_count,
        min_confidence_benign=min_confidence,
        min_abs_support_diff=min_abs_support_diff,
        keep_direction="benign",
    )
    benign = sort_itemsets_for_class(benign, target_class="benign").head(top_n_benign)

    attack = filter_itemsets(
        scored,
        min_k=min_k,
        max_k=max_k,
        min_total_count=min_total_count,
        min_confidence_attack=min_confidence,
        min_abs_support_diff=min_abs_support_diff,
        keep_direction="attack",
    )
    attack = sort_itemsets_for_class(attack, target_class="attack").head(top_n_attack)

    out = pd.concat([benign, attack], axis=0, ignore_index=True)

    # De-duplicate identical itemsets if one appears in both selections
    subset_cols = ["itemset"] if "itemset" in out.columns else None
    if subset_cols is not None:
        out = out.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    return out


def save_filtered_views(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save a few useful filtered views for inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = df.copy()

    benign = select_top_itemsets_per_class(
        scored,
        top_n_benign=200,
        top_n_attack=0,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    benign.to_csv(output_dir / "top_benign_itemsets.csv", index=False)

    attack = select_top_itemsets_per_class(
        scored,
        top_n_benign=0,
        top_n_attack=200,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    attack.to_csv(output_dir / "top_attack_itemsets.csv", index=False)

    features = select_top_itemsets_per_class(
        scored,
        top_n_benign=100,
        top_n_attack=100,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    features.to_csv(output_dir / "feature_itemsets.csv", index=False)
