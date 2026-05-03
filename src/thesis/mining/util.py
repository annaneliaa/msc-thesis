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
    if df.empty:
        return df.copy()

    out = df.copy()

    support_cols = [
        c
        for c in out.columns
        if c.startswith("support_")
        and c not in ("support_diff", "support_count", "support")
    ]
    if len(support_cols) < 2:
        return out

    denom = sum(out[c] for c in support_cols)
    for col in support_cols:
        label = col[len("support_") :]
        out[f"confidence_{label}"] = (out[col] / denom.replace(0, pd.NA)).fillna(0.0)

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


# ---------------------------------------------------------------------
# Sequence mining utilities
# ---------------------------------------------------------------------


def sequence_contains_pattern(
    sequence: list[set[str]],
    pattern: tuple[str, ...],
) -> bool:
    """
    True if pattern occurs as an ordered subsequence across the alert itemsets.

    Each pattern item must appear in some alert itemset that comes after the
    alert itemset matched by the previous pattern item.
    """
    if not pattern:
        return True

    j = 0
    for itemset in sequence:
        if pattern[j] in itemset:
            j += 1
            if j == len(pattern):
                return True

    return False


def sequence_support_in_group(
    sequences: list[list[set[str]]],
    pattern: tuple[str, ...],
) -> tuple[int, float]:
    """
    Count how many sequences contain the given sequential pattern.
    """
    if not sequences:
        return 0, 0.0

    count = sum(1 for seq in sequences if sequence_contains_pattern(seq, pattern))
    support = count / len(sequences)

    return count, support


def add_cross_label_sequence_supports(
    mined_df: pd.DataFrame,
    target_sequences: list[list[set[str]]],
    other_sequences: list[list[set[str]]],
    target_label: str,
    other_label: str,
) -> pd.DataFrame:
    """
    For each mined sequence, compute support in both target and other label groups.
    """
    print("Calculating cross-label sequence supports...")

    if mined_df.empty:
        return mined_df.copy()

    out = mined_df.copy()

    target_counts = []
    target_supports = []
    other_counts = []
    other_supports = []

    for sequence in out["sequence"]:
        c_t, s_t = sequence_support_in_group(target_sequences, sequence)
        c_o, s_o = sequence_support_in_group(other_sequences, sequence)

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


def sort_sequences_for_class(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    sorted_df = df.copy()
    conf_string = f"confidence_{target_class}"
    support_string = f"support_{target_class}"

    return sorted_df.sort_values(
        [conf_string, support_string, "k"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def filter_sequences(
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
    Filter mined sequences using already-computed class counts/supports.

    keep_direction:
        - "both": keep benign-leaning and attack-leaning sequences
        - "benign": only keep sequences where support_benign > support_attack
        - "attack": only keep sequences where support_attack > support_benign
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


def select_top_sequences_per_class(
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
    Select top discriminative sequences for both classes.

    This is intended for building a sequence-pattern feature vocabulary.
    """
    scored = df.copy()

    benign = filter_sequences(
        scored,
        min_k=min_k,
        max_k=max_k,
        min_total_count=min_total_count,
        min_confidence_benign=min_confidence,
        min_abs_support_diff=min_abs_support_diff,
        keep_direction="benign",
    )
    benign = sort_sequences_for_class(benign, target_class="benign").head(top_n_benign)

    attack = filter_sequences(
        scored,
        min_k=min_k,
        max_k=max_k,
        min_total_count=min_total_count,
        min_confidence_attack=min_confidence,
        min_abs_support_diff=min_abs_support_diff,
        keep_direction="attack",
    )
    attack = sort_sequences_for_class(attack, target_class="attack").head(top_n_attack)

    out = pd.concat([benign, attack], axis=0, ignore_index=True)

    if "sequence" in out.columns:
        out = out.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    return out


# -----------------------------------------------------------------
# Itemset sequence mining utilities
# -----------------------------------------------------------------


def itemset_sequence_contains_pattern(
    sequence: list[set[str]],
    pattern: tuple[frozenset[str], ...],
) -> bool:
    """
    True if each step of pattern (a frozenset) is a subset of some alert itemset
    in sequence, with steps matching in strictly increasing alert order.
    """
    if not pattern:
        return True

    j = 0
    for itemset in sequence:
        if pattern[j].issubset(itemset):
            j += 1
            if j == len(pattern):
                return True

    return False


def itemset_sequence_support_in_group(
    sequences: list[list[set[str]]],
    pattern: tuple[frozenset[str], ...],
) -> tuple[int, float]:
    """
    Count how many sequences contain the given itemset sequential pattern.
    """
    if not sequences:
        return 0, 0.0

    count = sum(
        1 for seq in sequences if itemset_sequence_contains_pattern(seq, pattern)
    )
    return count, count / len(sequences)


def add_cross_label_itemset_sequence_supports(
    mined_df: pd.DataFrame,
    target_sequences: list[list[set[str]]],
    other_sequences: list[list[set[str]]],
    target_label: str,
    other_label: str,
) -> pd.DataFrame:
    """
    For each mined itemset sequence, compute support in both label groups.
    """
    print("Calculating cross-label itemset sequence supports...")

    if mined_df.empty:
        return mined_df.copy()

    out = mined_df.copy()

    target_counts, target_supports, other_counts, other_supports = [], [], [], []

    for pattern in out["sequence"]:
        c_t, s_t = itemset_sequence_support_in_group(target_sequences, pattern)
        c_o, s_o = itemset_sequence_support_in_group(other_sequences, pattern)
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


def save_filtered_itemset_sequence_views(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save useful filtered views of mined itemset sequences for inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = df.copy()

    benign = select_top_sequences_per_class(
        scored,
        top_n_benign=200,
        top_n_attack=0,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    benign.to_csv(output_dir / "top_benign_itemset_sequences.csv", index=False)

    attack = select_top_sequences_per_class(
        scored,
        top_n_benign=0,
        top_n_attack=200,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    attack.to_csv(output_dir / "top_attack_itemset_sequences.csv", index=False)

    features = select_top_sequences_per_class(
        scored,
        top_n_benign=100,
        top_n_attack=100,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    features.to_csv(output_dir / "feature_itemset_sequences.csv", index=False)


def save_filtered_sequence_views(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save useful filtered sequence views for inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = df.copy()

    benign = select_top_sequences_per_class(
        scored,
        top_n_benign=200,
        top_n_attack=0,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    benign.to_csv(output_dir / "top_benign_sequences.csv", index=False)

    attack = select_top_sequences_per_class(
        scored,
        top_n_benign=0,
        top_n_attack=200,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    attack.to_csv(output_dir / "top_attack_sequences.csv", index=False)

    features = select_top_sequences_per_class(
        scored,
        top_n_benign=100,
        top_n_attack=100,
        min_total_count=20,
        min_abs_support_diff=0.01,
        min_confidence=0.7,
    )
    features.to_csv(output_dir / "feature_sequences.csv", index=False)
