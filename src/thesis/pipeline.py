from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from thesis.config import GroupingConfig
from thesis.schemas.preprocessing import (
    IncomingAlert,
    IncomingSuricataGroup,
    TokenizedAlert,
)
from thesis.preprocessing.parsing import parse_incoming_alert, parse_suricata_group_row
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.caching.ingestor import CacheIngestor
from thesis.grouping.group_alerts import (
    ALERTBERT_METHOD,
    FIXED_WINDOW_METHOD,
    FIXED_WINDOW_HOST_METHOD,
    group_alerts,
)

if TYPE_CHECKING:
    from thesis.grouping.alertbert_grouper import AlertBERTGrouper


def build_grouper(grouping: GroupingConfig) -> "AlertBERTGrouper | None":
    """Construct an AlertBERTGrouper from config, or return None for fixed-window mode."""
    if grouping.mode != ALERTBERT_METHOD:
        return None
    from thesis.grouping.alertbert_grouper import AlertBERTGrouper

    cfg = grouping.alertbert
    checkpoint_dir = Path(cfg.models_path) / cfg.model_id
    return AlertBERTGrouper(
        checkpoint_dir=checkpoint_dir,
        delta=cfg.delta,
        theta=cfg.theta,
        dim_reduction=cfg.dim_reduction,
        padding=cfg.padding,
        readout=cfg.readout,
        device=cfg.device,
    )


def process_alert_batch(
    rows: list[dict],
    scenario: str,
    ingestor: CacheIngestor,
    grouping_mode: str = FIXED_WINDOW_METHOD,
    grouper: "AlertBERTGrouper | None" = None,
    window_size: int = 2,
) -> int:
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

    grouping_kwargs: dict = {}
    if grouping_mode in (FIXED_WINDOW_METHOD, FIXED_WINDOW_HOST_METHOD):
        grouping_kwargs["window_size"] = window_size
    alert_groups = group_alerts(
        tokenized_alerts, method=grouping_mode, grouper=grouper, **grouping_kwargs
    )

    if tokenized_alerts:
        ingestor.ingest_alert_batch(tokenized_alerts, batch_name=scenario)

    if alert_groups:
        ingestor.ingest_groups(tokenized_alerts, alert_groups)

    return len(tokenized_alerts)


def process_suricata_group_batch(
    rows: list[dict],
    ingestor: CacheIngestor,
) -> int:
    """
    Parse and ingest a batch of pre-grouped Suricata rows.

    Each row is already a closed group; the grouping step is skipped entirely.
    Tokens are derived from SignatureText via the Suricata-specific tokenizer.
    Returns the number of rows successfully ingested.
    """
    entries = []
    for row in rows:
        try:
            suricata_row = IncomingSuricataGroup.from_row(row)
            entry = parse_suricata_group_row(suricata_row)
            entries.append(entry)
        except Exception as e:
            print(f"Skipping Suricata row due to error: {e}")
            continue

    ingestor.ingest_suricata_group_batch(entries)
    return len(entries)
