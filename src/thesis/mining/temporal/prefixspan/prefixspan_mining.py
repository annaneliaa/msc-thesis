from collections import Counter
from pathlib import Path

import pandas as pd


def contains_subsequence(sequence: list[set[str]], pattern: tuple[str, ...]) -> bool:
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


def _project_sequence(
    sequence: list[set[str]], prefix: tuple[str, ...]
) -> list[set[str]]:
    """
    Return the suffix of alert itemsets after the first occurrence of prefix.
    """
    j = 0
    for idx, itemset in enumerate(sequence):
        if prefix[j] in itemset:
            j += 1
            if j == len(prefix):
                return sequence[idx + 1 :]

    return []


def _prefixspan_recursive(
    prefix: tuple[str, ...],
    projected_db: list[list[set[str]]],
    min_count: int,
    results: list[dict],
    max_len: int | None,
) -> None:
    counts: Counter[str] = Counter()

    for sequence in projected_db:
        # count each unique item across all alert itemsets in this sequence, once per sequence
        seen: set[str] = set()
        for itemset in sequence:
            seen.update(itemset)
        counts.update(seen)

    frequent_items = sorted(
        [(item, count) for item, count in counts.items() if count >= min_count],
        key=lambda x: (-x[1], x[0]),
    )

    for item, support_count in frequent_items:
        new_pattern = prefix + (item,)

        results.append(
            {
                "sequence": new_pattern,
                "k": len(new_pattern),
                "support_count": support_count,
            }
        )

        if max_len is not None and len(new_pattern) >= max_len:
            continue

        new_projected_db: list[list[set[str]]] = [
            _project_sequence(sequence, (item,)) for sequence in projected_db
        ]
        new_projected_db = [seq for seq in new_projected_db if seq]

        if new_projected_db:
            _prefixspan_recursive(
                prefix=new_pattern,
                projected_db=new_projected_db,
                min_count=min_count,
                results=results,
                max_len=max_len,
            )


def run_prefixspan(
    sequences: list[list[set[str]]],
    min_support: float = 0.05,
    max_len: int | None = 3,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Mine frequent sequential patterns.

    Returns one row per frequent sequence.
    """
    n_tx = len(sequences)

    if n_tx == 0:
        return pd.DataFrame(
            columns=[
                "sequence",
                "sequence_str",
                "k",
                "support_count",
                "support",
            ]
        )

    min_count = max(1, int(min_support * n_tx))

    results: list[dict] = []

    _prefixspan_recursive(
        prefix=(),
        projected_db=sequences,
        min_count=min_count,
        results=results,
        max_len=max_len,
    )

    if not results:
        return pd.DataFrame(
            columns=[
                "sequence",
                "sequence_str",
                "k",
                "support_count",
                "support",
            ]
        )

    out = pd.DataFrame(results).drop_duplicates(subset=["sequence"]).copy()
    out["sequence_str"] = out["sequence"].apply(lambda x: " -> ".join(x))
    out["support"] = out["support_count"] / n_tx
    out["n_alert_groups"] = n_tx
    out["min_support"] = min_support
    out["min_count"] = min_count

    return out.sort_values(
        by=["k", "support_count", "sequence_str"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


# -----------------------------------------------------------------
# Itemset PrefixSpan
# Each pattern step is a frozenset; steps must match in different alerts.
# Within one step, items may be drawn from the same alert (i-extension).
# Across steps, a later alert is required (s-extension).
# -----------------------------------------------------------------


def _project_itemset_entry(
    partial: frozenset[str],
    suffix: list[frozenset[str]],
    new_item: str,
    i_extension: bool,
) -> tuple[frozenset[str], list[frozenset[str]]] | None:
    """
    Project one sequence entry for the next extension.

    i_extension=True  — new_item must be in partial (same alert as last match,
                        lexicographically after the previous pivot item).
    i_extension=False — new_item must be in some alert in suffix (new alert).
    """
    if i_extension:
        if new_item in partial:
            return frozenset(x for x in partial if x > new_item), suffix
        return None
    else:
        for idx, alert in enumerate(suffix):
            if new_item in alert:
                return frozenset(x for x in alert if x > new_item), suffix[idx + 1 :]
        return None


def _prefixspan_itemset_recursive(
    prefix: tuple[frozenset[str], ...],
    projected_db: list[tuple[frozenset[str], list[frozenset[str]]]],
    min_count: int,
    results: list[dict],
    max_len: int | None,
) -> None:
    i_counts: Counter[str] = Counter()
    s_counts: Counter[str] = Counter()

    for partial, suffix in projected_db:
        i_counts.update(partial)

        s_seen: set[str] = set()
        for alert in suffix:
            s_seen.update(alert)
        s_counts.update(s_seen)

    # i-extensions: add an item to the last step (no new alert required)
    if prefix:
        for item, count in sorted(i_counts.items()):
            if count < min_count:
                continue
            new_prefix = prefix[:-1] + (frozenset(prefix[-1] | {item}),)
            results.append(
                {
                    "sequence": new_prefix,
                    "k": len(new_prefix),
                    "n_items": sum(len(s) for s in new_prefix),
                    "support_count": count,
                }
            )
            new_db: list[tuple[frozenset[str], list[frozenset[str]]]] = []
            for partial, suffix in projected_db:
                proj = _project_itemset_entry(partial, suffix, item, i_extension=True)
                if proj is not None:
                    new_db.append(proj)
            if new_db:
                _prefixspan_itemset_recursive(
                    new_prefix, new_db, min_count, results, max_len
                )

    # s-extensions: append a new singleton step (new alert required)
    if max_len is None or len(prefix) < max_len:
        for item, count in sorted(s_counts.items()):
            if count < min_count:
                continue
            new_prefix = prefix + (frozenset({item}),)
            results.append(
                {
                    "sequence": new_prefix,
                    "k": len(new_prefix),
                    "n_items": sum(len(s) for s in new_prefix),
                    "support_count": count,
                }
            )
            new_db = []
            for partial, suffix in projected_db:
                proj = _project_itemset_entry(partial, suffix, item, i_extension=False)
                if proj is not None:
                    new_db.append(proj)
            if new_db:
                _prefixspan_itemset_recursive(
                    new_prefix, new_db, min_count, results, max_len
                )


def run_itemset_prefixspan(
    sequences: list[list[set[str]]],
    min_support: float = 0.05,
    max_len: int | None = 3,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Mine frequent sequential patterns where each step is an itemset.

    max_len limits the number of steps (i.e. the number of distinct alerts
    spanned by a pattern); i-extensions that grow a step do not count against it.

    Returns one row per frequent pattern.
    """
    _EMPTY_COLS = [
        "sequence",
        "sequence_str",
        "k",
        "n_items",
        "support_count",
        "support",
    ]

    n_tx = len(sequences)
    if n_tx == 0:
        return pd.DataFrame(columns=_EMPTY_COLS)

    min_count = max(1, int(min_support * n_tx))

    projected_db: list[tuple[frozenset[str], list[frozenset[str]]]] = [
        (frozenset(), [frozenset(alert) for alert in seq]) for seq in sequences
    ]

    results: list[dict] = []
    _prefixspan_itemset_recursive(
        prefix=(),
        projected_db=projected_db,
        min_count=min_count,
        results=results,
        max_len=max_len,
    )

    if not results:
        return pd.DataFrame(columns=_EMPTY_COLS)

    out = pd.DataFrame(results).drop_duplicates(subset=["sequence"]).copy()
    out["sequence_str"] = out["sequence"].apply(
        lambda pattern: " -> ".join(
            "{" + ", ".join(sorted(step)) + "}" for step in pattern
        )
    )
    out["support"] = out["support_count"] / n_tx
    out["n_alert_groups"] = n_tx
    out["min_support"] = min_support
    out["min_count"] = min_count

    return out.sort_values(
        by=["k", "n_items", "support_count", "sequence_str"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
