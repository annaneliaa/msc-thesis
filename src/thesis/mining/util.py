import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# Worker-process state; populated once per child process by _init_parallel_workers
_worker_seqs: list | None = None


def _init_parallel_workers(seqs):
    global _worker_seqs
    _worker_seqs = seqs


def _itemset_pattern_support_worker(pattern):
    return itemset_sequence_support_in_group(_worker_seqs, pattern)


def _sequence_pattern_support_worker(pattern):
    return sequence_support_in_group(_worker_seqs, pattern)


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


def remove_subset_subsumed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove itemsets that are a proper subset of a larger itemset in df.

    If {A, B} and {A, B, C} both appear, {A, B} is dropped because {A, B, C}
    captures strictly more information. {A, B} is only kept when no superset
    of it survives the preceding quality filters.
    """
    if df.empty:
        return df.copy()

    itemsets = [frozenset(s) for s in df["itemset"].tolist()]
    subsumed: set[int] = set()
    for i, s in enumerate(itemsets):
        for t in itemsets:
            if s < t:
                subsumed.add(i)
                break

    return df[~df.index.isin(subsumed)].reset_index(drop=True)


def filter_mined_itemsets(
    df: pd.DataFrame,
    *,
    min_k: int = 1,
    max_k: int | None = None,
    min_support_count: int = 10,
    min_abs_support_diff: float = 0.0,
    min_confidence_attack: float = 0.0,
    max_confidence_attack: float | None = None,
    min_confidence_benign: float = 0.0,
    max_overlap: float | None = None,
    remove_subsumed: bool = True,
) -> pd.DataFrame:
    """
    Apply quality filters to a mined itemsets DataFrame.

    Applied in order:
      1. Size bounds    — min_k <= k <= max_k
      2. Occurrence     — support_count >= min_support_count
      3. Discriminative — |support_diff| >= min_abs_support_diff,
                          confidence_attack >= min_confidence_attack,
                          confidence_attack <= max_confidence_attack,
                          confidence_benign >= min_confidence_benign,
                          overlap <= max_overlap
      4. Non-redundancy — drop itemsets that are a proper subset of a larger
                          itemset that already passed filters 1-3

    Parameters
    ----------
    df : enriched mined itemsets DataFrame (after add_cross_label_supports
         and add_confidence_scores).
    min_k : minimum itemset size.
    max_k : maximum itemset size (None = no limit).
    min_support_count : minimum absolute occurrence count.
    min_abs_support_diff : minimum |support_target - support_other|.
    min_confidence_attack : minimum confidence_attack.
    max_confidence_attack : maximum confidence_attack; drop patterns that appear
        too frequently in attacks (useful when mining benign-only features).
        None disables this filter.
    min_confidence_benign : minimum confidence_benign.
    max_overlap : maximum allowed class overlap, defined as
        min(support_benign, support_attack) / max(support_benign, support_attack).
        0.0 keeps only class-pure itemsets; 1.0 applies no overlap filtering.
        None disables this filter.
    remove_subsumed : if True, remove proper-subset duplicates after filtering.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    mask = pd.Series(True, index=out.index)

    # 1. Size bounds
    mask &= out["k"] >= min_k
    if max_k is not None:
        mask &= out["k"] <= max_k

    # 2. Occurrence
    mask &= out["support_count"] >= min_support_count

    # 3. Discriminativeness
    if "support_diff" in out.columns:
        mask &= out["support_diff"].abs() >= min_abs_support_diff
    if min_confidence_attack > 0.0 and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] >= min_confidence_attack
    if max_confidence_attack is not None and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] <= max_confidence_attack
    if min_confidence_benign > 0.0 and "confidence_benign" in out.columns:
        mask &= out["confidence_benign"] >= min_confidence_benign
    if (
        max_overlap is not None
        and "support_benign" in out.columns
        and "support_attack" in out.columns
    ):
        lo = out[["support_benign", "support_attack"]].min(axis=1)
        hi = out[["support_benign", "support_attack"]].max(axis=1)
        overlap = (lo / hi.replace(0, pd.NA)).fillna(0.0)
        mask &= overlap <= max_overlap

    out = out.loc[mask].reset_index(drop=True)

    # 4. Non-redundancy — applied last so smaller itemsets survive when their
    #    supersets failed the quality filters above.
    if remove_subsumed:
        out = remove_subset_subsumed(out)

    return out


def filter_or_patterns(
    df: pd.DataFrame,
    *,
    min_abs_support_diff: float = 0.0,
    min_confidence_attack: float = 0.0,
    max_confidence_attack: float | None = None,
    min_confidence_benign: float = 0.0,
    max_n_clauses: int | None = None,
) -> pd.DataFrame:
    """
    Apply quality filters to a mined OR-pattern DataFrame.

    OR patterns bypass the itemset/sequence filter step and need their own
    post-mining gate.  The DataFrame must have columns produced by
    mine_or_disjunctions (support_diff, confidence_attack, confidence_benign,
    n_clauses) which are present on eclat_result.or_df before column trimming.

    Parameters
    ----------
    min_abs_support_diff : minimum |support_benign - support_attack|.
    min_confidence_attack : minimum confidence_attack.
    max_confidence_attack : maximum confidence_attack (None = no limit).
    min_confidence_benign : minimum confidence_benign.
    max_n_clauses : drop OR patterns with more than this many clauses (None = no limit).
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    mask = pd.Series(True, index=out.index)

    if min_abs_support_diff > 0.0 and "support_diff" in out.columns:
        mask &= out["support_diff"].abs() >= min_abs_support_diff
    if min_confidence_attack > 0.0 and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] >= min_confidence_attack
    if max_confidence_attack is not None and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] <= max_confidence_attack
    if min_confidence_benign > 0.0 and "confidence_benign" in out.columns:
        mask &= out["confidence_benign"] >= min_confidence_benign
    if max_n_clauses is not None and "n_clauses" in out.columns:
        mask &= out["n_clauses"] <= max_n_clauses

    return out.loc[mask].reset_index(drop=True)


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
    patterns = out["sequence"].tolist()
    n_target = len(target_sequences)

    # support_count from PrefixSpan is already the count in target_sequences
    target_counts = out["support_count"].tolist()
    target_supports = [c / n_target for c in target_counts]

    n_workers = min(os.cpu_count() or 1, max(1, len(patterns)))
    chunksize = max(1, len(patterns) // (n_workers * 4))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_parallel_workers,
        initargs=(other_sequences,),
    ) as executor:
        other_results = list(
            executor.map(
                _sequence_pattern_support_worker, patterns, chunksize=chunksize
            )
        )

    other_counts = [r[0] for r in other_results]
    other_supports = [r[1] for r in other_results]

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
    patterns = out["sequence"].tolist()
    n_target = len(target_sequences)

    # support_count from PrefixSpan is already the count in target_sequences
    target_counts = out["support_count"].tolist()
    target_supports = [c / n_target for c in target_counts]

    n_workers = min(os.cpu_count() or 1, max(1, len(patterns)))
    chunksize = max(1, len(patterns) // (n_workers * 4))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_parallel_workers,
        initargs=(other_sequences,),
    ) as executor:
        other_results = list(
            executor.map(_itemset_pattern_support_worker, patterns, chunksize=chunksize)
        )

    other_counts = [r[0] for r in other_results]
    other_supports = [r[1] for r in other_results]

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


# -----------------------------------------------------------------
# Advanced sequence quality filtering
# -----------------------------------------------------------------


def _compute_item_supports(df: pd.DataFrame, support_col: str) -> dict[str, float]:
    """Build {item: support} from k=1 singleton rows in df."""
    k1 = df[df["k"] == 1]
    result: dict[str, float] = {}
    for _, row in k1.iterrows():
        seq = row["sequence"]
        step = seq[0]
        if isinstance(step, frozenset):
            if len(step) == 1:
                result[next(iter(step))] = row[support_col]
        elif isinstance(step, str):
            result[step] = row[support_col]
    return result


def add_lift_scores(
    df: pd.DataFrame,
    support_col: str = "support",
) -> pd.DataFrame:
    """
    Add a lift column: observed_support / product(k=1 item supports).

    lift > 1 means the sequence co-occurs more than expected by independence.
    Rows where any item has no k=1 entry (e.g., filtered out as infrequent)
    receive NaN — they will fail a min_lift filter if one is applied.
    Works for both item sequences (tuple[str]) and itemset sequences
    (tuple[frozenset[str]]): items in each step are flattened for the product.
    """
    if df.empty:
        return df.copy()

    item_supports = _compute_item_supports(df, support_col)

    def _lift(row) -> float:
        seq = row["sequence"]
        if not seq:
            return float("nan")
        items = (
            [item for step in seq for item in step]
            if isinstance(seq[0], frozenset)
            else list(seq)
        )
        expected = 1.0
        for item in items:
            p = item_supports.get(item)
            if p is None or p <= 0.0:
                return float("nan")
            expected *= p
        return row[support_col] / expected

    out = df.copy()
    out["lift"] = out.apply(_lift, axis=1)
    return out


def remove_prefix_subsumed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sequences that are a proper prefix of a longer sequence in df.

    If A->B and A->B->C both appear, A->B is dropped because A->B->C
    captures strictly more information. A->B is only kept when no extension
    of it survives the preceding quality filters.
    """
    if df.empty:
        return df.copy()

    all_seqs: set = set(df["sequence"].tolist())
    subsumed: set = set()
    for seq in all_seqs:
        for length in range(1, len(seq)):
            prefix = seq[:length]
            if prefix in all_seqs:
                subsumed.add(prefix)

    return df[~df["sequence"].isin(subsumed)].reset_index(drop=True)


def filter_mined_sequences(
    df: pd.DataFrame,
    *,
    min_k: int = 3,
    min_support_count: int = 10,
    min_abs_support_diff: float = 0.0,
    min_confidence_attack: float = 0.0,
    max_confidence_attack: float | None = None,
    min_confidence_benign: float = 0.0,
    min_lift: float | None = None,
    max_overlap: float | None = None,
    remove_subsumed: bool = True,
    support_col: str = "support",
) -> pd.DataFrame:
    """
    Apply quality filters to a mined sequences DataFrame.

    Applied in order:
      1. Length         — k >= min_k  (k=1 rarely informative, k=2 weak signal)
      2. Occurrence     — support_count >= min_support_count
      3. Discriminative — |support_diff| >= min_abs_support_diff,
                          confidence_attack >= min_confidence_attack,
                          confidence_attack <= max_confidence_attack,
                          confidence_benign >= min_confidence_benign,
                          overlap <= max_overlap
      4. Novelty        — lift >= min_lift  (skipped when min_lift is None)
                          lift = support / prod(k=1 item supports); items
                          below min_support have no k=1 entry and yield NaN
      5. Non-redundancy — drop sequences that are a proper prefix of another
                          sequence that already passed filters 1-4

    Parameters
    ----------
    df : enriched mined sequences DataFrame (after add_cross_label_*
         and add_confidence_scores).
    min_k : minimum number of steps.
    min_support_count : minimum absolute occurrence count in the target group.
    min_abs_support_diff : minimum |support_target - support_other|.
    min_confidence_attack : minimum confidence_attack.
    max_confidence_attack : maximum confidence_attack; drop patterns that appear
        too frequently in attacks (useful when mining benign-only features).
        None disables this filter.
    min_confidence_benign : minimum confidence_benign.
    min_lift : if given, require lift >= min_lift.
    max_overlap : maximum allowed class overlap, defined as
        min(support_benign, support_attack) / max(support_benign, support_attack).
        0.0 keeps only class-pure sequences; 1.0 applies no overlap filtering.
        None disables this filter.
    remove_subsumed : if True, remove proper-prefix duplicates after filtering.
    support_col : support column used for lift computation.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Lift must be computed before applying the k filter so k=1 rows are
    # still present for the per-item support lookup.
    if min_lift is not None:
        out = add_lift_scores(out, support_col=support_col)

    mask = pd.Series(True, index=out.index)

    # 1. Length
    mask &= out["k"] >= min_k

    # 2. Occurrence
    mask &= out["support_count"] >= min_support_count

    # 3. Discriminativeness
    if "support_diff" in out.columns:
        mask &= out["support_diff"].abs() >= min_abs_support_diff
    if min_confidence_attack > 0.0 and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] >= min_confidence_attack
    if max_confidence_attack is not None and "confidence_attack" in out.columns:
        mask &= out["confidence_attack"] <= max_confidence_attack
    if min_confidence_benign > 0.0 and "confidence_benign" in out.columns:
        mask &= out["confidence_benign"] >= min_confidence_benign
    if (
        max_overlap is not None
        and "support_benign" in out.columns
        and "support_attack" in out.columns
    ):
        lo = out[["support_benign", "support_attack"]].min(axis=1)
        hi = out[["support_benign", "support_attack"]].max(axis=1)
        overlap = (lo / hi.replace(0, pd.NA)).fillna(0.0)
        mask &= overlap <= max_overlap

    # 4. Novelty
    if min_lift is not None and "lift" in out.columns:
        mask &= out["lift"].fillna(0.0) >= min_lift

    out = out.loc[mask].reset_index(drop=True)

    # 5. Non-redundancy — applied last so short sequences survive when their
    #    extensions failed the quality filters above.
    if remove_subsumed:
        out = remove_prefix_subsumed(out)

    return out
