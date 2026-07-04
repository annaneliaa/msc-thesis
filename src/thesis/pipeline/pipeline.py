from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from thesis.config import GroupingConfig
from thesis.paths import FEATURE_DIR, ROOT
from thesis.schemas.groups import AlertGroup, GroupSnapshot
from thesis.schemas.mining import FeatureSelectionConfig
from thesis.schemas.preprocessing import (
    IncomingAlert,
    IncomingSuricataGroup,
    ParsedSuricataGroup,
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
    CSCAS_PREGROUPED_METHOD,
    CSCAS_TARGET_WINDOW_METHOD,
    CSCAS_TARGET_WINDOW_SECONDS,
    CSCAS_TARGET_SESSION_METHOD,
    CSCAS_TARGET_SESSION_TIMEOUT_SECONDS,
    CSCAS_TARGET_SESSION_LENGTH_SECONDS,
    group_alerts,
    group_cscas_rows_by_target_window,
    group_cscas_rows_by_target_session,
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
# AIT-ADS ingestion (alert-by-alert stream, fixed_window/time_delta grouping)
# ---------------------------------------------------------------------------


def ensure_feature_manifest(scenario: str, root_dir: Path | None = None) -> None:
    """Initialise the feature schema manifest for a scenario if it's missing."""
    from thesis.features.manifest import initialize_feature_manifest

    root_dir = root_dir or FEATURE_DIR
    manifest_path = root_dir / scenario / "manifest.json"
    if manifest_path.exists():
        print(f"  [skip] Feature manifest already exists at {manifest_path}")
        return
    print(f"  Feature manifest not found for '{scenario}', initialising...")
    initialize_feature_manifest(scenario_name=scenario, root_dir=root_dir)
    print(f"  Created feature manifest at {manifest_path}")


def convert_ait_alerts_to_json(
    scenario: str, alerts_json_path: Path | None = None
) -> Path:
    """
    Convert data/alerts_csv/<scenario>_alerts.txt into the canonical
    alerts.json format IncomingAlert.from_row() expects. Cached: returns the
    existing output path if alerts.json is already there.
    """
    if alerts_json_path is not None:
        if not alerts_json_path.exists():
            raise FileNotFoundError(f"Filtered alerts not found: {alerts_json_path}")
        print(f"  [filtered] Using {alerts_json_path}")
        return alerts_json_path

    input_path = ROOT / "data" / "alerts_csv" / f"{scenario}_alerts.txt"
    output_dir = ROOT / "artifacts" / "processed-data" / scenario
    output_path = output_dir / "alerts.json"

    if output_path.exists():
        print(f"  [skip] alerts.json already exists at {output_path}")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw alerts file not found: {input_path}\n"
            "Place the alerts CSV in data/alerts_csv/ before running."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    alerts: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alerts.append(
                {
                    "time": int(row["time"]),
                    "signature": row["name"],
                    "ip": row["ip"],
                    "host": row["host"],
                    "short": row["short"],
                    "label": row["time_label"],
                    "event_label": row["event_label"],
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    print(f"  Wrote {len(alerts)} alerts → {output_path}")
    return output_path


def ingest_ait_alert_batch(
    scenario: str,
    alerts_path: Path,
    cache_dir: Path,
    grouping_mode: str = FIXED_WINDOW_METHOD,
    grouping: GroupingConfig | None = None,
) -> None:
    """Tokenise alerts.json and ingest alerts + groups into the TokenCache at cache_dir."""
    alert_store_dir = cache_dir / "alerts"
    if alert_store_dir.exists() and any(alert_store_dir.glob("*.json")):
        print(f"  [skip] Alert cache already populated at {alert_store_dir}")
        return

    with alerts_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    grouper = build_grouper(grouping) if grouping is not None else None
    cache = TokenCache(cache_dir=cache_dir)
    count = ingest_to_cache(
        scenario=scenario,
        rows=payload,
        cache=cache,
        grouping_mode=grouping_mode,
        grouper=grouper,
        window_size=grouping.window_size if grouping is not None else 2,
    )
    print(f"  Processed {count} alerts into cache.")


def ingest_ait_scenario(
    scenario: str,
    alerts_json_path: Path | None = None,
    cache_dir: Path | None = None,
    grouping: GroupingConfig | None = None,
) -> Path:
    """
    Populate alert_groups_raw.json for one AIT-ADS scenario: convert the raw
    alerts CSV to JSON, tokenise + ingest into the TokenCache, ensure the
    feature manifest exists, then build alert_groups from the closed groups.

    This is the same ingestion prefix run_baseline_experiment() performs
    before training — factored out so scripts that only need alert_groups
    (e.g. run_mining_window_sweep.py) don't have to run a full baseline
    experiment (and train a model) just to populate the cache.
    """
    grouping = grouping or GroupingConfig()
    cache_dir = cache_dir or (
        ROOT / "artifacts" / "cache" / scenario / "groups" / grouping.mode
    )
    out_path = cache_dir / "alert_groups" / "alert_groups_raw.json"

    if out_path.exists():
        print(f"  [skip] alert_groups already exist at {out_path}")
        return out_path

    alerts_path = convert_ait_alerts_to_json(scenario, alerts_json_path)
    ingest_ait_alert_batch(
        scenario,
        alerts_path,
        cache_dir,
        grouping_mode=grouping.mode,
        grouping=grouping,
    )
    ensure_feature_manifest(scenario)
    load_or_build_alert_groups(scenario, cache_dir)
    return out_path


# ---------------------------------------------------------------------------
# CSCAS ingestion (pre-grouped Suricata rows)
# ---------------------------------------------------------------------------


def _cscas_cache_subdir(
    grouping_method: str,
    window_seconds: float,
    session_timeout: float,
    session_length: float,
) -> str:
    """
    Cache directory name for a given CSCAS grouping_method/hyperparameter combo.

    window_seconds only affects CSCAS_TARGET_WINDOW_METHOD baskets, and
    session_timeout/session_length only affect CSCAS_TARGET_SESSION_METHOD
    baskets, so each is folded into the subdir name only for its own method --
    otherwise two ingests with different hyperparameters would write to the
    same directory and silently clobber/skip each other.
    """
    if grouping_method == CSCAS_TARGET_WINDOW_METHOD:
        window_tag = f"{window_seconds:g}".replace(".", "_")
        return f"{grouping_method}_w{window_tag}s"
    if grouping_method == CSCAS_TARGET_SESSION_METHOD:
        timeout_tag = f"{session_timeout:g}".replace(".", "_")
        length_tag = f"{session_length:g}".replace(".", "_")
        return f"{grouping_method}_t{timeout_tag}s_l{length_tag}s"
    return grouping_method


def ingest_cscas_scenario(
    csv_path: Path | None = None,
    cache_dir: Path | None = None,
    grouping_method: str = CSCAS_PREGROUPED_METHOD,
    window_seconds: float = CSCAS_TARGET_WINDOW_SECONDS,
    session_timeout: float = CSCAS_TARGET_SESSION_TIMEOUT_SECONDS,
    session_length: float = CSCAS_TARGET_SESSION_LENGTH_SECONDS,
) -> Path:
    """
    Parse data/cscas/dataset-labeled-anon-ip.csv into alert_groups_raw.json,
    the same artifact load_or_build_alert_groups() produces for AIT scenarios,
    so downstream scripts (e.g. run_mining_window_sweep.py) can consume it via
    --grouping-method <grouping_method>.

    CSCAS rows are already closed, pre-aggregated groups (one signature x
    external-IP cluster per row), unlike AIT's alert-by-alert stream. There is
    no incremental grouping step, so this skips TokenCache/GroupCacheEntry
    storage entirely — writing ~1.4M individual per-group cache files would be
    impractically slow — and builds AlertGroup objects directly in memory.

    grouping_method selects how CSV rows become AlertGroup baskets:
    - CSCAS_PREGROUPED_METHOD ("cscas_pregrouped", default): one basket per
      CSV row, i.e. per single signature. Itemset mining on these baskets can
      only ever decompose one signature's own description into words — it
      cannot discover cross-signature co-occurrence — and sorted_items is
      always empty, so sequence mining is a no-op.
    - CSCAS_TARGET_WINDOW_METHOD ("cscas_target_window"): rows are grouped by
      (internal target IP, fixed time window of window_seconds), so a basket
      can contain multiple distinct signatures fired against the same host
      within a bounded time span. See group_cscas_rows_by_target_window for
      why this is what makes mining meaningful for this dataset, and for the
      detection-latency caveat fixed windows have.
    - CSCAS_TARGET_SESSION_METHOD ("cscas_target_session"): rows are grouped
      by internal target IP using a session-gap scheme (session_timeout,
      session_length) instead of a fixed window -- see
      group_cscas_rows_by_target_session for why this bounds detection
      latency more tightly for the common case of a target that goes quiet.

    Whichever hyperparameters apply to grouping_method are folded into the
    cache subdir name (see _cscas_cache_subdir) so that re-ingesting with
    different hyperparameters writes to a separate directory instead of
    silently reusing another run's stale alert_groups_raw.json.
    """
    scenario = "cscas"
    csv_path = csv_path or (ROOT / "data" / "cscas" / "dataset-labeled-anon-ip.csv")
    cache_dir = cache_dir or (
        ROOT
        / "artifacts"
        / "cache"
        / scenario
        / "groups"
        / _cscas_cache_subdir(
            grouping_method, window_seconds, session_timeout, session_length
        )
    )
    out_path = cache_dir / "alert_groups" / "alert_groups_raw.json"

    if out_path.exists():
        print(f"  [skip] alert_groups already exist at {out_path}")
        return out_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSCAS CSV not found: {csv_path}")

    print(f"[1/2] Reading CSCAS CSV from {csv_path}...")
    rows = pd.read_csv(csv_path).to_dict("records")
    print(f"  {len(rows)} rows loaded.")

    print("[2/2] Parsing rows into alert_groups...")
    parsed_rows: list[ParsedSuricataGroup] = []
    n_skipped = 0
    for row in rows:
        try:
            suricata_row = IncomingSuricataGroup.from_row(row)
            parsed_rows.append(parse_suricata_group_row(suricata_row))
        except Exception:
            n_skipped += 1
            continue

    if n_skipped:
        print(f"  [warn] Skipped {n_skipped} rows due to parsing errors.")

    if grouping_method == CSCAS_PREGROUPED_METHOD:
        alert_groups: list[AlertGroup] = [
            AlertGroup(
                alert_group_id=parsed.group_id,
                group_id=parsed.group_id,
                method=CSCAS_PREGROUPED_METHOD,
                start_ts=parsed.ts,
                end_ts=parsed.ts,
                n_alerts=parsed.n_alerts,
                abs_items=set(parsed.tokens),
                raw_items=set(parsed.tokens),
                sorted_items=[],
                alert_ips={parsed.ext_ip},
                group_label=parsed.label,
                alert_labels=None,
                weight=1.0,
                proto=parsed.proto,
                int_ip=parsed.int_ip,
                int_port=parsed.int_port,
                ext_port=parsed.ext_port,
                int_ip_is_multiple=parsed.int_ip_is_multiple,
                ext_ip_is_multiple=parsed.ext_ip_is_multiple,
            )
            for parsed in parsed_rows
        ]
    elif grouping_method == CSCAS_TARGET_WINDOW_METHOD:
        alert_groups = group_cscas_rows_by_target_window(
            parsed_rows, window_seconds=window_seconds
        )
    elif grouping_method == CSCAS_TARGET_SESSION_METHOD:
        alert_groups = group_cscas_rows_by_target_session(
            parsed_rows,
            session_timeout=session_timeout,
            session_length=session_length,
        )
    else:
        raise ValueError(f"Unsupported CSCAS grouping_method: {grouping_method!r}")

    alert_groups.sort(key=lambda g: g.start_ts)
    save_alert_groups_json(alert_groups, out_path)
    print(f"  Saved {len(alert_groups)} alert_groups → {out_path}")
    return out_path


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
        "proto": t.proto,
        "int_ip": t.int_ip,
        "int_port": t.int_port,
        "ext_port": t.ext_port,
        "int_ip_is_multiple": t.int_ip_is_multiple,
        "ext_ip_is_multiple": t.ext_ip_is_multiple,
    }


def save_snapshots_json(snapshots: list[GroupSnapshot], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([snapshot_to_dict(s) for s in snapshots], f, indent=2)


def save_alert_groups_json(alert_groups: list[AlertGroup], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([alert_group_to_dict(t) for t in alert_groups], f, indent=2)


def load_or_build_alert_groups(scenario: str, cache_dir: Path) -> list[AlertGroup]:
    """
    Load alert_groups_raw.json if it already exists under cache_dir, otherwise
    query the TokenCache at cache_dir for closed groups and build+save it.
    """
    out_path = cache_dir / "alert_groups" / "alert_groups_raw.json"

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            serialized = json.load(f)
        if serialized:
            print(f"  [skip] Loading alert_groups from existing {out_path}")
            alert_groups = [alert_group_from_dict(t) for t in serialized]
            print(f"  Loaded {len(alert_groups)} alert_groups from cache.")
            return alert_groups
        print(f"  [warn] {out_path} is empty, rebuilding alert_groups...")

    cache = TokenCache(cache_dir=cache_dir)
    alert_groups = select_alert_groups(cache=cache, require_closed=True)

    save_alert_groups_json(alert_groups, out_path)
    print(f"  Built {len(alert_groups)} alert_groups → {out_path}")
    return alert_groups


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
    df = pd.concat(
        [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1,
    )
    return df.loc[:, ~df.columns.duplicated(keep="first")]


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


def encode_and_cache_alert_groups(
    scenario: str,
    alert_groups: list[AlertGroup],
    schema_name: str,
    cache_dir: Path,
    feature_selection: FeatureSelectionConfig | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Load the registered feature schema for scenario, encode alert_groups under
    it, and cache the result as parquet under cache_dir. Returns the cached
    parquet if it already exists and no feature_selection override is given.
    """
    from thesis.encoders.service import encode_alert_groups_for_schema
    from thesis.features.schema_registry import FeatureSchemaRegistry
    from thesis.features.util import select_symbolic_features

    safe_name = schema_name.replace("+", "_").replace("/", "_")
    tx_dir = cache_dir / "alert_groups"
    out_path = tx_dir / f"alert_groups_{safe_name}.parquet"

    registry = FeatureSchemaRegistry(root_dir=FEATURE_DIR)
    schema = registry.load(
        scenario_name=scenario,
        schema_name=schema_name,
        schema_version=None,
    )

    if out_path.exists() and feature_selection is None:
        print(f"  [skip] Loading encoded alert_groups from existing {out_path}")
        df = pd.read_parquet(out_path)
        print(f"  Loaded {len(df)} alert_groups from parquet.")
        return df, schema

    if feature_selection is not None and (
        feature_selection.top_k is not None
        or feature_selection.min_utility_score is not None
        or feature_selection.filter_cross_host_or
    ):
        before = len(schema.symbolic.features) if schema.symbolic else 0
        schema = select_symbolic_features(schema, feature_selection)
        after = len(schema.symbolic.features) if schema.symbolic else 0
        print(f"  Feature selection: {before} → {after} symbolic features")

    print("Loaded schema. Encoding alert_group data under schema...")
    feature_df = encode_alert_groups_for_schema(
        alert_groups=alert_groups,
        schema=schema,
        top_k=None,
    )
    meta_df = pd.DataFrame(
        [
            {
                "alert_group_id": t.alert_group_id,
                "group_label": t.group_label,
                "n_alerts": t.n_alerts,
                "weight": t.weight,
            }
            for t in alert_groups
        ]
    )
    df = pd.concat(
        [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1,
    )
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    tx_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Encoded {len(df)} alert_groups under schema '{schema_name}' → {out_path}")
    return df, schema


def alert_group_from_dict(d: dict) -> AlertGroup:
    """Deserialize an AlertGroup from the dict format produced by alert_group_to_dict."""
    return AlertGroup(
        alert_group_id=d["alert_group_id"],
        group_id=d["group_id"],
        method=d["method"],
        start_ts=d["start_ts"],
        end_ts=d["end_ts"],
        n_alerts=d["n_alerts"],
        alert_ids=d.get("alert_ids"),
        abs_items=set(d["abs_items"]),
        raw_items=set(d["raw_items"]) if d.get("raw_items") is not None else None,
        sorted_items=[set(s) for s in d.get("sorted_items", [])],
        alert_ips=set(d.get("alert_ips", [])),
        group_label=d.get("group_label"),
        alert_labels=set(d["alert_labels"])
        if d.get("alert_labels") is not None
        else None,
        weight=d.get("weight", 1.0),
        proto=d.get("proto"),
        int_ip=d.get("int_ip"),
        int_port=d.get("int_port"),
        ext_port=d.get("ext_port"),
        int_ip_is_multiple=d.get("int_ip_is_multiple", False),
        ext_ip_is_multiple=d.get("ext_ip_is_multiple", False),
    )


def load_alert_groups_json(path: Path) -> list[AlertGroup]:
    with path.open("r", encoding="utf-8") as f:
        return [alert_group_from_dict(d) for d in json.load(f)]


def ingest_to_cache(
    scenario: str,
    rows: list[dict],
    cache: TokenCache,
    grouping_mode: str = FIXED_WINDOW_METHOD,
    grouper: "AlertBERTGrouper | None" = None,
    window_size: int = 2,
) -> int:
    """Wrap process_alert_batch with CacheIngestor construction."""
    ingestor = CacheIngestor(cache=cache)
    return process_alert_batch(
        rows=rows,
        scenario=scenario,
        ingestor=ingestor,
        grouping_mode=grouping_mode,
        grouper=grouper,
        window_size=window_size,
    )


def is_single_class_split(
    alert_groups: list[AlertGroup],
    test_frac: float = 0.3,
    train_start: int = 0,
    random_split: bool = False,
    random_seed: int = 42,
) -> bool:
    """Return True if the train or test split would contain only one class."""
    import random as _random

    label_map = {"benign": 0, "attack": 1}
    labels = [
        label_map[t.group_label] for t in alert_groups if t.group_label in label_map
    ]
    n = len(labels)
    if n == 0:
        return True
    if random_split:
        _random.Random(random_seed).shuffle(labels)
    split = int((1 - test_frac) * n)
    if split <= 0 or split >= n:
        return True
    return len(set(labels[train_start:split])) < 2 or len(set(labels[split:])) < 2


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
