from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from thesis.config import GroupingConfig
from thesis.schemas.groups import AlertGroup, GroupSnapshot
from thesis.schemas.preprocessing import (
    IncomingAlert,
    IncomingSuricataGroup,
    TokenizedAlert,
)
from thesis.preprocessing.parsing import parse_incoming_alert, parse_suricata_group_row
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.caching.cache import TokenCache
from thesis.caching.ingestor import CacheIngestor
from thesis.caching.selector import select_group_snapshots
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


# ---------------------------------------------------------------------------
# Convenience helpers used by CLI commands
# ---------------------------------------------------------------------------


def load_alert_rows_from_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Input file must contain a JSON list of alert objects.")
    return payload


def open_scenario_cache(scenario: str, cache_dir: Path | str) -> TokenCache:
    return TokenCache(cache_dir=Path(cache_dir) / scenario)


def run_preprocess_batch(
    scenario: str,
    cache_dir: Path | str,
    rows: list[dict],
    grouping_mode: str,
    grouper: "AlertBERTGrouper | None" = None,
    window_size: int = 2,
) -> int:
    cache = open_scenario_cache(scenario, cache_dir)
    ingestor = CacheIngestor(cache=cache)
    return process_alert_batch(
        rows=rows,
        scenario=scenario,
        ingestor=ingestor,
        grouping_mode=grouping_mode,
        grouper=grouper,
        window_size=window_size,
    )


def select_snapshots(
    cache: TokenCache,
    allowed_methods: set[str] | None = None,
    limit: int | None = None,
    min_start_ts: int | None = None,
    max_end_ts: int | None = None,
    require_closed: bool = True,
) -> list[GroupSnapshot]:
    return select_group_snapshots(
        cache=cache,
        allowed_methods=allowed_methods,
        limit=limit,
        min_start_ts=min_start_ts,
        max_end_ts=max_end_ts,
        require_closed=require_closed,
    )


def select_alert_groups(
    cache: TokenCache,
    allowed_methods: set[str] | None = None,
    limit: int | None = None,
    min_start_ts: int | None = None,
    max_end_ts: int | None = None,
    require_closed: bool = True,
) -> list[AlertGroup]:
    snapshots = select_group_snapshots(
        cache=cache,
        allowed_methods=allowed_methods,
        limit=limit,
        min_start_ts=min_start_ts,
        max_end_ts=max_end_ts,
        require_closed=require_closed,
    )
    return [s.to_alert_group() for s in snapshots]


def snapshot_to_dict(s: GroupSnapshot) -> dict:
    return {
        "group_id": s.group_id,
        "method": s.method,
        "version": s.version,
        "start_ts": s.start_ts,
        "end_ts": s.end_ts,
        "alert_ids": s.alert_ids,
        "n_alerts": s.n_alerts,
        "items": sorted(s.items),
        "sorted_items": [sorted(itemset) for itemset in s.sorted_items],
        "alert_ips": sorted(s.alert_ips),
        "group_label": s.group_label,
        "alert_labels": sorted(s.alert_labels) if s.alert_labels is not None else None,
        "status": s.status,
    }


def alert_group_to_dict(t: AlertGroup) -> dict:
    return {
        "alert_group_id": t.alert_group_id,
        "group_id": t.group_id,
        "method": t.method,
        "start_ts": t.start_ts,
        "end_ts": t.end_ts,
        "n_alerts": t.n_alerts,
        "alert_ids": t.alert_ids,
        "abs_items": sorted(t.abs_items),
        "raw_items": sorted(t.raw_items) if t.raw_items is not None else None,
        "sorted_items": [sorted(itemset) for itemset in t.sorted_items],
        "alert_ips": sorted(t.alert_ips),
        "group_label": t.group_label,
        "alert_labels": sorted(t.alert_labels) if t.alert_labels is not None else None,
        "weight": t.weight,
    }


def save_snapshots_json(snapshots: list[GroupSnapshot], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([snapshot_to_dict(s) for s in snapshots], f, indent=2)


def save_alert_groups_json(alert_groups: list[AlertGroup], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([alert_group_to_dict(t) for t in alert_groups], f, indent=2)


def build_encoded_alert_groups_df(
    alert_groups: list[AlertGroup],
    schema,
    top_k: int | None = None,
) -> pd.DataFrame:
    from thesis.encoders.service import encode_alert_groups_for_schema

    feature_df = encode_alert_groups_for_schema(
        alert_groups=alert_groups,
        schema=schema,
        top_k=top_k,
    )
    meta_df = pd.DataFrame(
        [
            {
                "alert_group_id": t.alert_group_id,
                "group_id": t.group_id,
                "method": t.method,
                "start_ts": t.start_ts,
                "end_ts": t.end_ts,
                "n_alerts": t.n_alerts,
                "group_label": t.group_label,
                "weight": t.weight,
            }
            for t in alert_groups
        ]
    )
    return pd.concat(
        [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1,
    )


def save_encoded_df(
    df: pd.DataFrame,
    schema_name: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    safe_schema_name = schema_name.replace("+", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"alert_groups_{safe_schema_name}.parquet"
    csv_path = out_dir / f"alert_groups_{safe_schema_name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def combine_mining_results(
    eclat_mined_df: pd.DataFrame,
    item_seq_mined_df: pd.DataFrame,
    mining_filters=None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Apply optional filters, merge Eclat itemsets and PrefixSpan sequences into
    a single combined DataFrame sorted by confidence then support.

    Returns (combined_df, n_itemsets, n_sequences) where the counts reflect
    the post-filter sizes before merging.
    """
    from thesis.mining.util import filter_mined_itemsets, filter_mined_sequences

    eclat_df = eclat_mined_df.copy()
    item_seq_df = item_seq_mined_df.copy()

    if mining_filters is not None:
        f = mining_filters.itemsets
        eclat_df = filter_mined_itemsets(
            eclat_df,
            min_k=f.min_k,
            max_k=f.max_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )
        f = mining_filters.item_sequences
        item_seq_df = filter_mined_sequences(
            item_seq_df,
            min_k=f.min_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            min_lift=f.min_lift,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )

    n_itemsets = len(eclat_df)
    n_sequences = len(item_seq_df)

    eclat_df["mining_type"] = "itemset"
    item_seq_df = item_seq_df.rename(columns={"sequence": "itemset"})
    item_seq_df["mining_type"] = "item_sequence"

    cols_to_keep = [
        "itemset",
        "mining_type",
        "support",
        "confidence_attack",
        "confidence_benign",
    ]
    eclat_df = eclat_df[[c for c in cols_to_keep if c in eclat_df.columns]]
    item_seq_df = item_seq_df[[c for c in cols_to_keep if c in item_seq_df.columns]]

    combined_df = pd.concat([eclat_df, item_seq_df], axis=0, ignore_index=True)

    sort_cols = []
    if "confidence_attack" in combined_df.columns:
        sort_cols.append("confidence_attack")
    elif "confidence_benign" in combined_df.columns:
        sort_cols.append("confidence_benign")
    if "support" in combined_df.columns:
        sort_cols.append("support")

    if sort_cols:
        combined_df = combined_df.sort_values(
            by=sort_cols, ascending=False, na_position="last"
        ).reset_index(drop=True)

    return combined_df, n_itemsets, n_sequences
