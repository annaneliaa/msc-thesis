from __future__ import annotations

from thesis.preprocessing.cache import TokenCache
from thesis.schemas.cache import CacheQuery, CacheResponse
from thesis.schemas.preprocessing import WindowTransaction


def compute_window_weight(
    window_id: int,
    latest_window_id: int,
    decay_factor: float = 1.0,
) -> float:
    """
    Exponential decay by window distance.
    decay_factor=1.0 means no decay.
    """
    if decay_factor <= 0:
        raise ValueError("decay_factor must be > 0")

    age = latest_window_id - window_id
    if age < 0:
        age = 0

    return decay_factor**age


def select_transactions_from_response(
    response: CacheResponse,
    retention_windows: int | None = None,
    decay_factor: float = 1.0,
) -> list[WindowTransaction]:
    """
    Convert cache response windows into miner-ready transactions.
    """
    windows = sorted(response.windows, key=lambda w: w.window_id)

    if not windows:
        return []

    if retention_windows is not None:
        latest_window_id = windows[-1].window_id
        min_keep = latest_window_id - retention_windows + 1
        windows = [w for w in windows if w.window_id >= min_keep]
    else:
        latest_window_id = windows[-1].window_id

    transactions: list[WindowTransaction] = []

    for window in windows:
        weight = compute_window_weight(
            window_id=window.window_id,
            latest_window_id=latest_window_id,
            decay_factor=decay_factor,
        )

        transactions.append(
            WindowTransaction(
                window_id=window.window_id,
                window_start=window.start_ts,
                window_end=window.end_ts,
                n_alerts=len(window.alert_ids),
                items=set(window.items),
                alert_labels=(
                    set(window.alert_labels)
                    if window.alert_labels is not None
                    else None
                ),
                tx_label=window.tx_label if window.tx_label is not None else None,
                weight=weight,
            )
        )

    return transactions


def select_transactions(
    cache: TokenCache,
    query: CacheQuery,
    retention_windows: int | None = None,
    decay_factor: float = 1.0,
) -> list[WindowTransaction]:
    """
    Query cache and return miner-ready transactions.
    """
    response = cache.query(query)

    return select_transactions_from_response(
        response=response,
        retention_windows=retention_windows,
        decay_factor=decay_factor,
    )
