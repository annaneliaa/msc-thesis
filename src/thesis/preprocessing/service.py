from __future__ import annotations

from typing import TYPE_CHECKING

from thesis.schemas.preprocessing import GroupSnapshot, IncomingAlert, TokenizedAlert
from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.group_selector import (
    select_group_snapshots_from_response,
    create_cache_query,
)
from thesis.preprocessing.group_alerts import (
    # ALERTBERT_METHOD,
    FIXED_WINDOW_METHOD,
    group_alerts,
)

if TYPE_CHECKING:
    from thesis.preprocessing.grouping.alertbert_grouper import AlertBERTGrouper

""""
Set up pipeline methods here for the module.
To be called from CLI commands to expose as less as possible.
"""


def process_alert_batch(
    rows: list[dict],
    scenario: str,
    ingestor: CacheIngestor,
    grouping_mode: str = FIXED_WINDOW_METHOD,
    grouper: AlertBERTGrouper | None = None,
) -> int:
    """Parse, tokenize, group, and ingest one batch of raw alert rows.

    Pass grouping_mode='alertbert' and a pre-loaded AlertBERTGrouper as
    `grouper` to use embedding-based grouping instead of the fixed window.
    """
    tokenized_alerts: list[TokenizedAlert] = []
    for row in rows:
        try:
            alert = IncomingAlert.from_row(row)
            parsed = parse_incoming_alert(alert=alert, scenario=scenario)
            tokenized = tokenize_alert(parsed)
            tokenized_alerts.append(tokenized)
        except Exception as e:
            print(f"Skipping row due to parsing/tokenization error: {e}")
            continue

    alert_groups = group_alerts(tokenized_alerts, method=grouping_mode, grouper=grouper)

    if tokenized_alerts:
        ingestor.ingest_alert_batch(tokenized_alerts, batch_name=scenario)

    if alert_groups:
        ingestor.ingest_groups(tokenized_alerts, alert_groups)

    return len(tokenized_alerts)


def select_groups_from_cache(
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
