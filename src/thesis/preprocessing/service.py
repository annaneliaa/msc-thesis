from thesis.schemas.preprocessing import IncomingAlert, TokenizedAlert
from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.preprocessing.cache import TokenCache
from thesis.schemas.preprocessing import GroupSnapshot
from thesis.preprocessing.group_selector import (
    select_group_snapshots_from_response,
    create_cache_query,
)

""""
Set up pipeline methods here for the module.
To be called from CLI commands to expose as less as possible.
"""


def process_alert_batch(rows: list[dict], scenario: str, ingestor: CacheIngestor):
    tokenize_alerts: list[TokenizedAlert] = []
    for row in rows:
        try:
            alert = IncomingAlert.from_row(row)
            parsed = parse_incoming_alert(alert=alert, scenario=scenario)
            tokenized = tokenize_alert(parsed)
            tokenize_alerts.append(tokenized)
        except Exception as e:
            print(f"Skipping row due to parsing/tokenization error: {e}")
            continue

    if tokenize_alerts:
        ingestor.ingest_alert_batch(tokenize_alerts, batch_name=scenario)

    return len(tokenize_alerts)


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
