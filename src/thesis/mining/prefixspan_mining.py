from collections import Counter
from pathlib import Path

import pandas as pd


def contains_subsequence(sequence: list[str], pattern: tuple[str, ...]) -> bool:
    """
    True if pattern occurs as an ordered subsequence in sequence.
    Not necessarily contiguous.
    """
    if not pattern:
        return True

    j = 0
    for item in sequence:
        if item == pattern[j]:
            j += 1
            if j == len(pattern):
                return True

    return False


def _project_sequence(sequence: list[str], prefix: tuple[str, ...]) -> list[str]:
    """
    Return suffix after first occurrence of prefix as subsequence.
    """
    j = 0
    for idx, item in enumerate(sequence):
        if item == prefix[j]:
            j += 1
            if j == len(prefix):
                return sequence[idx + 1 :]

    return []


def _prefixspan_recursive(
    prefix: tuple[str, ...],
    projected_db: list[list[str]],
    min_count: int,
    results: list[dict],
    max_len: int | None,
) -> None:
    counts: Counter[str] = Counter()

    for sequence in projected_db:
        # count once per transaction/sequence
        counts.update(set(sequence))

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

        new_projected_db = [
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
    sequences: list[list[str]],
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
    out["n_transactions"] = n_tx
    out["min_support"] = min_support
    out["min_count"] = min_count

    return out.sort_values(
        by=["k", "support_count", "sequence_str"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
