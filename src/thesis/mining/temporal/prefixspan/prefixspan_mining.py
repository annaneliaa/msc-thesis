from collections import Counter
from pathlib import Path

import pandas as pd

from thesis.mining.common.util import min_support_count


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

    min_count = min_support_count(min_support, n_tx)

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
#
# Projection: for each sequence we keep `ends`, every alert index at which
# the pattern's last step can be matched, given the earlier steps matched as
# early as possible. Keeping all of them (not just the first) is what makes
# i-extensions correct: in <{a},{a,b}> the first alert holding {a} can't be
# grown to {a,b}, but a later one can. The earlier version kept only the
# first match and missed or undercounted such patterns.
# -----------------------------------------------------------------


def _prefixspan_itemset_recursive(
    prefix: tuple[frozenset[str], ...],
    projected_db: list[tuple[list[frozenset[str]], list[int]]],
    min_count: int,
    results: list[dict],
    max_len: int | None,
) -> None:
    last_step = prefix[-1] if prefix else frozenset()
    last_max = max(last_step) if last_step else None

    i_counts: Counter[str] = Counter()
    s_counts: Counter[str] = Counter()
    for seq, ends in projected_db:
        if prefix:
            # i-extension candidates: items after the step's largest item, in
            # any alert where the current last step can end.
            i_seen: set[str] = set()
            for e in ends:
                i_seen.update(x for x in seq[e] if x > last_max)
            i_counts.update(i_seen)
        # s-extension candidates: items in any alert after the earliest end.
        first = ends[0] if prefix else -1
        s_seen: set[str] = set()
        for alert in seq[first + 1 :]:
            s_seen.update(alert)
        s_counts.update(s_seen)

    def _emit(new_prefix, count, new_db):
        results.append(
            {
                "sequence": new_prefix,
                "k": len(new_prefix),
                "n_items": sum(len(s) for s in new_prefix),
                "support_count": count,
            }
        )
        if new_db:
            _prefixspan_itemset_recursive(
                new_prefix, new_db, min_count, results, max_len
            )

    # i-extensions: add an item to the last step (no new alert required)
    if prefix:
        for item, count in sorted(i_counts.items()):
            if count < min_count:
                continue
            new_db = []
            for seq, ends in projected_db:
                new_ends = [e for e in ends if item in seq[e]]
                if new_ends:
                    new_db.append((seq, new_ends))
            _emit(prefix[:-1] + (last_step | {item},), count, new_db)

    # s-extensions: append a new singleton step (new alert required)
    if max_len is None or len(prefix) < max_len:
        for item, count in sorted(s_counts.items()):
            if count < min_count:
                continue
            new_db = []
            for seq, ends in projected_db:
                first = ends[0] if prefix else -1
                new_ends = [j for j in range(first + 1, len(seq)) if item in seq[j]]
                if new_ends:
                    new_db.append((seq, new_ends))
            _emit(prefix + (frozenset({item}),), count, new_db)


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

    min_count = min_support_count(min_support, n_tx)

    projected_db: list[tuple[list[frozenset[str]], list[int]]] = [
        ([frozenset(alert) for alert in seq], []) for seq in sequences
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
