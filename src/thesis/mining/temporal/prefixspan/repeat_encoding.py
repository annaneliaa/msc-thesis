"""
Repeat encoding for sequence mining.

Collapses consecutive token repetitions into bucketed categories to reduce
redundancy while preserving the signal that repetition occurred.
"""


def encode_runs(seq: list[str]) -> list[str]:
    """
    Collapse consecutive token repetitions into bucketed categories.

    Instead of tracking exact repetition counts (e.g., host:webserver x5),
    bucket into categories (e.g., host:webserver__repeat_5_plus).

    Buckets:
    - 1: no repeat marker (keep as-is)
    - 2: __repeat_2
    - 3-4: __repeat_3_4
    - 5+: __repeat_5_plus
    """
    if not seq:
        return []

    result = []
    i = 0

    while i < len(seq):
        token = seq[i]
        count = 1

        # Count consecutive occurrences
        while i + count < len(seq) and seq[i + count] == token:
            count += 1

        # Encode based on count
        if count == 1:
            result.append(token)
        elif count == 2:
            result.append(f"{token}__repeat_2")
        elif count <= 4:
            result.append(f"{token}__repeat_3_4")
        else:
            result.append(f"{token}__repeat_5_plus")

        i += count

    return result


def encode_sequence_of_itemsets(
    sequence: list[set[str]],
) -> list[set[str]]:
    """
    Apply repeat encoding to sequences of itemsets.

    Each itemset is processed independently, collapsing consecutive
    identical itemsets into a single encoded itemset.
    """
    if not sequence:
        return []

    result = []
    i = 0

    while i < len(sequence):
        itemset = sequence[i]
        count = 1

        # Count consecutive identical itemsets
        while i + count < len(sequence) and sequence[i + count] == itemset:
            count += 1

        # Encode the itemset based on repetition count
        encoded_itemset = _encode_itemset(itemset, count)
        result.append(encoded_itemset)

        i += count

    return result


def _encode_itemset(itemset: set[str], count: int) -> set[str]:
    """
    Encode a single itemset with a repeat marker based on its repetition count.
    """
    if count == 1:
        return itemset.copy()

    # Add repeat marker to each item in the itemset
    repeat_marker = _get_repeat_marker(count)
    return {f"{item}_{repeat_marker}" for item in itemset}


def _get_repeat_marker(count: int) -> str:
    """Get the appropriate repeat marker for a given count."""
    if count == 2:
        return "repeat_2"
    elif count <= 4:
        return "repeat_3_4"
    else:
        return "repeat_5_plus"


def decode_sequence_string(seq_str: str) -> str:
    """
    Simplify sequence string by removing repeat markers for display.

    Useful for grouping near-duplicates: sequences that differ only
    in their repeat markers will have identical decoded strings.
    """
    # Remove all repeat markers
    for marker in [
        "__repeat_2",
        "__repeat_3_4",
        "__repeat_5_plus",
        "_repeat_2",
        "_repeat_3_4",
        "_repeat_5_plus",
    ]:
        seq_str = seq_str.replace(marker, "")
    return seq_str


def filter_redundant_sequences(df, keep="highest_support"):
    """
    Filter out near-duplicate sequences that differ only in repeat markers.

    When sequences are identical after removing repeat markers, keep only the
    one with the highest support (or other criteria).

    Args:
        df: DataFrame with 'sequence_str' column
        keep: 'highest_support' to keep sequence with max support_count,
              or 'first' to keep the first occurrence

    Returns:
        Filtered DataFrame with one row per unique decoded sequence
    """
    if df.empty or "sequence_str" not in df.columns:
        return df

    df = df.copy()
    df["decoded_sequence"] = df["sequence_str"].apply(decode_sequence_string)

    if keep == "highest_support":
        # Keep the sequence with highest support for each decoded group
        return df.loc[df.groupby("decoded_sequence")["support_count"].idxmax()]
    elif keep == "first":
        # Keep the first occurrence of each decoded sequence
        return df.drop_duplicates(subset=["decoded_sequence"], keep="first")
    else:
        raise ValueError(f"Unknown keep strategy: {keep}")
