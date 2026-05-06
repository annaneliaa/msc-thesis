from __future__ import annotations

import ast
import itertools

import pandas as pd


def _jaccard_similarity(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _parse_itemset(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(x) for x in value)
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)):
            return tuple(str(x) for x in parsed)
    raise ValueError(f"Unsupported itemset value: {value!r}")


def _remove_near_duplicate_patterns(
    df: pd.DataFrame,
    target_tidsets: dict[tuple[str, ...], frozenset[int]],
    other_tidsets: dict[tuple[str, ...], frozenset[int]],
    jaccard_threshold: float = 0.98,
) -> pd.DataFrame:
    """Remove OR patterns that are near-duplicates based on Jaccard similarity of TID-sets."""
    kept_indices = []
    kept_keys: list[tuple[frozenset[int], frozenset[int]]] = []

    for idx, row in df.iterrows():
        clauses = row["clauses"]
        t_union = frozenset().union(*(target_tidsets[c] for c in clauses))
        o_union = frozenset().union(*(other_tidsets[c] for c in clauses))

        is_near_duplicate = False
        for kt, ko in kept_keys:
            t_sim = _jaccard_similarity(t_union, kt)
            o_sim = _jaccard_similarity(o_union, ko)
            if t_sim >= jaccard_threshold and o_sim >= jaccard_threshold:
                is_near_duplicate = True
                break

        if not is_near_duplicate:
            kept_indices.append(idx)
            kept_keys.append((t_union, o_union))

    return df.loc[kept_indices].reset_index(drop=True)


def _build_class_tidsets(
    transactions: list[frozenset[str]],
    itemsets: list[tuple[str, ...]],
) -> dict[tuple[str, ...], frozenset[int]]:
    """TID-set for each AND-itemset: indices of transactions where all items appear."""
    result: dict[tuple[str, ...], frozenset[int]] = {}
    for itemset in itemsets:
        s = frozenset(itemset)
        result[itemset] = frozenset(
            i for i, tx in enumerate(transactions) if s.issubset(tx)
        )
    return result


def mine_or_disjunctions(
    base_df: pd.DataFrame,
    target_transactions: list[frozenset[str]],
    other_transactions: list[frozenset[str]],
    target_label: str,
    other_label: str,
    *,
    max_or_arity: int = 2,
    min_coverage_gain: float = 0.02,
    min_abs_support_diff: float = 0.05,
    min_confidence: float = 0.6,
    direction: str = "both",
    remove_dominated: bool = True,
    jaccard_threshold: float = 0.98,
) -> pd.DataFrame:
    """
    Synthesize OR-of-AND patterns from already-filtered frequent itemsets.

    Each candidate is a disjunction of 2..max_or_arity AND-clauses.  Support
    is computed via TID-set union — the natural dual of ECLAT's intersection.

    Parameters
    ----------
    base_df:
        Filtered frequent itemsets (output of select_top_itemsets_per_class or
        filter_mined_itemsets).  Must have an 'itemset' column.
    target_transactions / other_transactions:
        Class-partitioned transaction lists (frozensets of items).
    min_coverage_gain:
        The OR must cover at least this much more of the target class than its
        best single component.  Set to 0.0 to disable.
    min_abs_support_diff:
        Minimum |support_target - support_other| for the OR pattern.
    min_confidence:
        At least one class confidence must exceed this threshold.
    direction:
        'benign' — only keep patterns where support_target > support_other;
        'attack' — only keep patterns where support_other > support_target;
        'both'   — keep either direction.
    remove_dominated:
        Drop OR patterns whose |support_diff| is no better than the best
        single clause.  These patterns add coverage but not discrimination.
    jaccard_threshold:
        Remove OR patterns that are near-duplicates: if both target and other
        TID-sets have Jaccard similarity >= threshold with a previously kept
        pattern, drop this one.  Range [0, 1]; 1.0 = only exact duplicates,
        0.98 removes patterns with >98% overlap.

    Returns
    -------
    DataFrame with columns mirroring frequent_itemsets plus 'clauses' and
    'n_clauses'.  'clauses' is a tuple of AND-clause tuples.
    """
    if base_df.empty:
        return pd.DataFrame()

    itemsets = [_parse_itemset(v) for v in base_df["itemset"]]
    n_target = len(target_transactions)
    n_other = len(other_transactions)

    if n_target == 0 or n_other == 0:
        return pd.DataFrame()

    target_tidsets = _build_class_tidsets(target_transactions, itemsets)
    other_tidsets = _build_class_tidsets(other_transactions, itemsets)

    # Pre-compute per-clause support_diff for dominance check
    clause_support_diff: dict[tuple[str, ...], float] = {}
    for itemset in itemsets:
        st = len(target_tidsets[itemset]) / n_target
        so = len(other_tidsets[itemset]) / n_other
        clause_support_diff[itemset] = st - so

    seen: set[frozenset[tuple[str, ...]]] = set()
    results: list[dict] = []

    for arity in range(2, max_or_arity + 1):
        for combo in itertools.combinations(range(len(itemsets)), arity):
            clauses = tuple(itemsets[i] for i in combo)

            # Deduplicate (order of clauses should not matter)
            key = frozenset(clauses)
            if key in seen:
                continue
            seen.add(key)

            # OR support = TID union
            target_union = frozenset().union(*(target_tidsets[c] for c in clauses))
            other_union = frozenset().union(*(other_tidsets[c] for c in clauses))

            count_target = len(target_union)
            count_other = len(other_union)
            support_target = count_target / n_target
            support_other = count_other / n_other
            support_diff = support_target - support_other

            # Require the OR to improve target coverage over the best single clause
            if min_coverage_gain > 0.0:
                best_single = max(len(target_tidsets[c]) for c in clauses) / n_target
                if support_target - best_single < min_coverage_gain:
                    continue

            # Discriminability filter
            if abs(support_diff) < min_abs_support_diff:
                continue

            if direction == "benign" and support_diff <= 0.0:
                continue
            if direction == "attack" and support_diff >= 0.0:
                continue

            denom = support_target + support_other
            conf_target = support_target / denom if denom else 0.0
            conf_other = support_other / denom if denom else 0.0

            if conf_target < min_confidence and conf_other < min_confidence:
                continue

            # Dominance: skip if no clause alone is strictly worse than the OR
            # (i.e., OR doesn't improve discrimination over any single component)
            if remove_dominated:
                best_single_diff = max(abs(clause_support_diff[c]) for c in clauses)
                if abs(support_diff) <= best_single_diff:
                    continue

            clauses_str = " OR ".join("(" + " & ".join(c) + ")" for c in clauses)

            results.append(
                {
                    "clauses": clauses,
                    "n_clauses": arity,
                    "itemset_str": clauses_str,
                    f"count_{target_label}": count_target,
                    f"support_{target_label}": support_target,
                    f"count_{other_label}": count_other,
                    f"support_{other_label}": support_other,
                    "support_diff": support_diff,
                    f"confidence_{target_label}": conf_target,
                    f"confidence_{other_label}": conf_other,
                }
            )

    if not results:
        return pd.DataFrame()

    out = (
        pd.DataFrame(results)
        .sort_values("support_diff", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    # Drop OR patterns whose effective TID-union equals that of any single clause.
    # This happens when all other clauses' TIDs are subsets of the dominant clause
    # (e.g. TID(sig:update) ⊆ TID(sig:authentication)), making the extra clause
    # semantically redundant at inference time.
    effective_keys: set[frozenset[int]] = set()
    dedup_idx: list[int] = []
    for idx, row in out.iterrows():
        clauses = row["clauses"]
        t_union = frozenset().union(*(target_tidsets[c] for c in clauses))
        o_union = frozenset().union(*(other_tidsets[c] for c in clauses))
        key = (t_union, o_union)
        if key not in effective_keys:
            effective_keys.add(key)
            dedup_idx.append(idx)

    out_dedup = out.loc[dedup_idx].reset_index(drop=True)

    if not out_dedup.empty and jaccard_threshold < 1.0:
        out_dedup = _remove_near_duplicate_patterns(
            out_dedup, target_tidsets, other_tidsets, jaccard_threshold
        )

    return out_dedup
