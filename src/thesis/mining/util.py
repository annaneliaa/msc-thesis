import pandas as pd


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

    return out.sort_values(
        by=["support_diff", f"support_{target_label}", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
