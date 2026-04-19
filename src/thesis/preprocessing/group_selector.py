from __future__ import annotations

from thesis.preprocessing.cache import TokenCache
from thesis.schemas.cache import CacheQuery, CacheResponse
from thesis.schemas.preprocessing import GroupSnapshot


def create_cache_query(
    allowed_methods: set[str] | None = None,
    only_closed: bool = True,
    allowed_statuses: set[str] | None = None,
    min_start_ts: int | None = None,
    max_end_ts: int | None = None,
    limit: int | None = None,
) -> CacheQuery:
    return CacheQuery(
        allowed_methods=allowed_methods or {"fixed_2s", "alertbert"},
        only_closed=only_closed,
        allowed_statuses=allowed_statuses or {"closed"},
        min_start_ts=min_start_ts,
        max_end_ts=max_end_ts,
        limit=limit,
    )


def select_group_snapshots_from_response(
    response: CacheResponse,
    limit: int | None = None,
    require_closed: bool = True,
) -> list[GroupSnapshot]:
    """
    Convert cache response groups into stable GroupSnapshot objects.
    """
    groups = sorted(response.groups, key=lambda g: (g.end_ts, g.start_ts, g.group_id))

    if require_closed:
        groups = [g for g in groups if g.status == "closed"]

    if not groups:
        return []

    if limit is not None:
        groups = groups[-limit:]

    snapshots: list[GroupSnapshot] = []

    for group in groups:
        snapshots.append(
            GroupSnapshot(
                group_id=str(group.group_id),
                method=group.method,
                version=group.version,
                start_ts=group.start_ts,
                end_ts=group.end_ts,
                alert_ids=list(group.alert_ids),
                n_alerts=group.n_alerts,
                items=set(group.items) if group.items is not None else set(),
                alert_labels=(
                    set(group.alert_labels) if group.alert_labels is not None else None
                ),
                tx_label=group.tx_label if group.tx_label is not None else None,
                status=group.status,
            )
        )

    return snapshots


def select_group_snapshots(
    cache: TokenCache,
    allowed_methods: set[str] | None = None,
    limit: int | None = None,
    min_start_ts: int | None = None,
    max_end_ts: int | None = None,
    require_closed: bool = True,
) -> list[GroupSnapshot]:
    """
    Query cache and return stable group snapshots for mining-prep.
    """

    query = create_cache_query(
        allowed_methods=allowed_methods,
        only_closed=require_closed,
        min_start_ts=min_start_ts,
        max_end_ts=max_end_ts,
        limit=limit,
    )

    response = cache.query(query)

    return select_group_snapshots_from_response(
        response=response,
        limit=limit,
        require_closed=require_closed,
    )
