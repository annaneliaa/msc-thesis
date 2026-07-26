from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable
from thesis.schemas.groups import GroupingRecord


# --- Fixed window -----------------------------------------------------------
FIXED_WINDOW_METHOD = "fixed_window"
FIXED_WINDOW_HOST_METHOD = "fixed_window_host"
FIXED_WINDOW_SECONDS = 2

# --- Time-delta --------------------------------------------------------------
TIME_DELTA_METHOD = "time_delta"
TIME_DELTA_HOST_METHOD = "time_delta_host"
TIME_DELTA_SECONDS = 2.0

# --- CSCAS grouping ------------------------------------------------------
CSCAS_PREGROUPED_METHOD = "cscas_pregrouped"
CSCAS_METHOD = "cscas_grouping"
# Values below match the authors' own validated production settings
# (Vaarandi & Guerra-Manzanares 2024; "Evaluating explainable AI for
# deep learning-based NIDS alert classification"), not swept.
CSCAS_SESSION_LENGTH_SECONDS = 300.0
CSCAS_SESSION_TIMEOUT_SECONDS = 60.0

# --- AlertBERT grouping -------------------------------------------------------
ALERTBERT_METHOD = "alertbert"

# --- DeepCASE grouping ---------------------------------------------------------
DEEPCASE_METHOD = "deepcase"

# --- Misc / legacy -----------------------------------------------------------
TEMPORAL_METHOD = "temporal_grouping"


@runtime_checkable
class GroupableAlert(Protocol):
    """
    Structural type for whatever alert-like object grouping runs on.

    Grouping only touches identity/timing/host/signature fields, which both
    ParsedAlert (pre-tokenization) and TokenizedAlert expose, so this module
    stays independent of thesis.preprocessing's concrete schemas.
    """

    alert_id: str
    ts: int | float
    host: str | None
    signature: str | None


# =============================================================================
# Shared helpers
# =============================================================================


def _host_key(alert: GroupableAlert) -> str:
    return alert.host or "_unknown"


def _signature_key(alert: GroupableAlert) -> str:
    return alert.signature or "_unknown"


def _split_by_host(alerts: list[GroupableAlert]) -> dict[str, list[GroupableAlert]]:
    by_host: dict[str, list[GroupableAlert]] = defaultdict(list)
    for alert in alerts:
        by_host[_host_key(alert)].append(alert)
    return by_host


def _split_by_host_signature(
    alerts: list[GroupableAlert],
) -> dict[tuple[str, str], list[GroupableAlert]]:
    by_key: dict[tuple[str, str], list[GroupableAlert]] = defaultdict(list)
    for alert in alerts:
        by_key[(_host_key(alert), _signature_key(alert))].append(alert)
    return by_key


def _chain_by_gap(
    sorted_alerts: list[GroupableAlert],
    gap_threshold: float,
    span_cap: float | None = None,
) -> list[tuple[GroupableAlert, str]]:
    """
    Assign each alert in a time-sorted stream to a group anchor based on
    gap-chaining: a new group starts whenever the gap to the previous alert
    exceeds gap_threshold seconds, or (if span_cap is set) the group's total
    span since its anchor would exceed span_cap seconds. Returns a list of
    (alert, anchor_id) pairs in the same order as the input.

    This is the shared core of time_delta (gap_threshold=delta, no cap) and
    cscas_grouping (gap_threshold=session_timeout, span_cap=session_length).
    """
    if not sorted_alerts:
        return []

    anchor_id = sorted_alerts[0].alert_id
    anchor_ts: float = sorted_alerts[0].ts
    prev_ts: float = anchor_ts

    assignments: list[tuple[GroupableAlert, str]] = []
    for alert in sorted_alerts:
        gap_exceeded = alert.ts - prev_ts > gap_threshold
        span_exceeded = span_cap is not None and (alert.ts - anchor_ts > span_cap)
        if gap_exceeded or span_exceeded:
            anchor_id = alert.alert_id
            anchor_ts = alert.ts
        prev_ts = alert.ts
        assignments.append((alert, anchor_id))
    return assignments


# =============================================================================
# Fixed window
# =============================================================================


def fixed_window_group_id(ts: int, window_size: int = FIXED_WINDOW_SECONDS) -> str:
    window_id = ts // window_size
    return f"fixed_window:{window_id}"


def group_alert_fixed_window(
    alert: GroupableAlert,
    window_size: int = FIXED_WINDOW_SECONDS,
) -> GroupingRecord:
    return GroupingRecord(
        alert_id=alert.alert_id,
        group_id=fixed_window_group_id(alert.ts, window_size=window_size),
        method=FIXED_WINDOW_METHOD,
    )


def group_alerts_fixed_window(
    alerts: list[GroupableAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    """
    Global fixed-window grouping: every alert in the same
    (ts // window_size) slot shares a group, regardless of host.
    """
    return [
        group_alert_fixed_window(alert, window_size=window_size) for alert in alerts
    ]


def group_alerts_fixed_window_by_group(
    alerts: list[GroupableAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[GroupableAlert]]:
    groups: dict[str, list[GroupableAlert]] = defaultdict(list)
    for alert in alerts:
        group_id = fixed_window_group_id(alert.ts, window_size=window_size)
        groups[group_id].append(alert)
    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


def group_alerts_fixed_window_host(
    alerts: list[GroupableAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    """
    Per-host fixed-window grouping: same (ts // window_size) calendar
    buckets as group_alerts_fixed_window, but keyed additionally by host so
    alerts from different hosts never share a group even if they land in
    the same window.
    """
    records: list[GroupingRecord] = []
    for host, host_alerts in _split_by_host(alerts).items():
        for alert in host_alerts:
            window_id = alert.ts // window_size
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"fixed_window_host:{host}:{window_id}",
                    method=FIXED_WINDOW_HOST_METHOD,
                )
            )
    return records


def group_alerts_fixed_window_host_by_group(
    alerts: list[GroupableAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[GroupableAlert]]:
    groups: dict[str, list[GroupableAlert]] = defaultdict(list)
    for host, host_alerts in _split_by_host(alerts).items():
        for alert in host_alerts:
            window_id = alert.ts // window_size
            groups[f"fixed_window_host:{host}:{window_id}"].append(alert)
    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


# =============================================================================
# Time-delta
# =============================================================================


def group_alerts_time_delta(
    alerts: list[GroupableAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """
    Global time-delta grouping (Landauer et al., 2022): alerts sorted by
    time; a new group starts whenever the gap to the previous alert
    exceeds delta seconds. No cap on group span or size — a continuous,
    dense stream with no delta-sized gap becomes one (potentially very
    large) group. This is the plain, uncapped method; compare against
    group_alerts_cscas for the capped, host+signature-split variant.
    """
    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    assignments = _chain_by_gap(sorted_alerts, gap_threshold=delta, span_cap=None)
    return [
        GroupingRecord(
            alert_id=alert.alert_id,
            group_id=f"{TIME_DELTA_METHOD}:{anchor_id}",
            method=TIME_DELTA_METHOD,
        )
        for alert, anchor_id in assignments
    ]


def group_alerts_time_delta_by_group(
    alerts: list[GroupableAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> dict[str, list[GroupableAlert]]:
    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    assignments = _chain_by_gap(sorted_alerts, gap_threshold=delta, span_cap=None)
    groups: dict[str, list[GroupableAlert]] = defaultdict(list)
    for alert, anchor_id in assignments:
        groups[f"{TIME_DELTA_METHOD}:{anchor_id}"].append(alert)
    return dict(groups)


def group_alerts_time_delta_host(
    alerts: list[GroupableAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """
    Per-host time-delta grouping: same stream-gap logic as
    group_alerts_time_delta, but run independently per host so a dense
    flood on one host cannot absorb alerts from another via the shared
    time dimension.
    """
    records: list[GroupingRecord] = []
    for host, host_alerts in _split_by_host(alerts).items():
        sorted_alerts = sorted(host_alerts, key=lambda a: a.ts)
        assignments = _chain_by_gap(sorted_alerts, gap_threshold=delta, span_cap=None)
        for alert, anchor_id in assignments:
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"{TIME_DELTA_HOST_METHOD}:{host}:{anchor_id}",
                    method=TIME_DELTA_HOST_METHOD,
                )
            )
    return records


def group_alerts_time_delta_host_by_group(
    alerts: list[GroupableAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> dict[str, list[GroupableAlert]]:
    groups: dict[str, list[GroupableAlert]] = defaultdict(list)
    for host, host_alerts in _split_by_host(alerts).items():
        sorted_alerts = sorted(host_alerts, key=lambda a: a.ts)
        assignments = _chain_by_gap(sorted_alerts, gap_threshold=delta, span_cap=None)
        for alert, anchor_id in assignments:
            groups[f"{TIME_DELTA_HOST_METHOD}:{host}:{anchor_id}"].append(alert)
    return dict(groups)


# =============================================================================
# CSCAS grouping
# =============================================================================


def group_alerts_cscas(
    alerts: list[GroupableAlert],
    session_length: float = CSCAS_SESSION_LENGTH_SECONDS,
    session_timeout: float = CSCAS_SESSION_TIMEOUT_SECONDS,
) -> list[GroupingRecord]:
    """
    Like time_delta_host, but additionally splits streams by signature, so
    an alert group never mixes alerts from different signatures. A new
    session starts whenever the gap to the previous alert exceeds
    session_timeout seconds, or the session's total span would exceed
    session_length seconds (whichever comes first). The host component is
    taken from alert.host and the signature from alert.signature; missing
    values are treated as a single anonymous host/signature.

    Defaults match the authors' own validated production configuration
    (SessionLength=300s, SessionTimeout=60s) rather than an arbitrary or
    swept choice.
    """
    records: list[GroupingRecord] = []
    for (host, signature), key_alerts in _split_by_host_signature(alerts).items():
        sorted_alerts = sorted(key_alerts, key=lambda a: a.ts)
        assignments = _chain_by_gap(
            sorted_alerts, gap_threshold=session_timeout, span_cap=session_length
        )
        for alert, anchor_id in assignments:
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"{CSCAS_METHOD}:{host}:{signature}:{anchor_id}",
                    method=CSCAS_METHOD,
                )
            )
    return records


def group_alerts_cscas_by_group(
    alerts: list[GroupableAlert],
    session_length: float = CSCAS_SESSION_LENGTH_SECONDS,
    session_timeout: float = CSCAS_SESSION_TIMEOUT_SECONDS,
) -> dict[str, list[GroupableAlert]]:
    groups: dict[str, list[GroupableAlert]] = defaultdict(list)
    for (host, signature), key_alerts in _split_by_host_signature(alerts).items():
        sorted_alerts = sorted(key_alerts, key=lambda a: a.ts)
        assignments = _chain_by_gap(
            sorted_alerts, gap_threshold=session_timeout, span_cap=session_length
        )
        for alert, anchor_id in assignments:
            groups[f"{CSCAS_METHOD}:{host}:{signature}:{anchor_id}"].append(alert)
    return dict(groups)


# =============================================================================
# AlertBERT grouping
# =============================================================================


def group_alerts_alertbert(
    alerts: list[GroupableAlert],
    delta: float,
    theta: float,
    **kwargs,
) -> list[GroupingRecord]:
    """
    Groups alerts using a pretrained AlertBERT masked-language-model
    checkpoint (no training happens here). Thin wrapper that defers the
    torch/alertbert import to call time, so importing this module doesn't
    drag those (heavy, and graph-tool-requiring -- see
    thesis.grouping.alertbert_grouping's module docstring for the required
    conda env) dependencies in for callers only using the other methods.

    alerts must additionally expose `.short` (TokenizedAlert does; the
    generic GroupableAlert protocol above doesn't require it since the
    other methods don't need it).
    """
    from thesis.grouping.alertbert_grouping import (
        group_alerts_alertbert as _group_alerts_alertbert,
    )

    return _group_alerts_alertbert(alerts, delta=delta, theta=theta, **kwargs)


# =============================================================================
# DeepCASE grouping
# =============================================================================


def group_alerts_deepcase(
    alerts: list[GroupableAlert],
    train_alerts: list[GroupableAlert],
    train_id: str,
    **kwargs,
) -> list[GroupingRecord]:
    """
    Groups alerts using a DeepCASE ContextBuilder trained on train_alerts
    (no pretrained checkpoint exists for this dataset, unlike AlertBERT, so
    training happens here). Thin wrapper that defers the torch/deepcase
    import to call time, so importing this module doesn't drag those in for
    callers only using the other methods. See
    thesis.grouping.deepcase_grouping's module docstring for the
    shared-vocabulary requirement and caching behavior.

    Build train_id with thesis.grouping.deepcase_grouping.train_id_for_scenarios
    rather than hand-writing it, so the same training scenario set always
    resolves to the same on-disk cache entry regardless of call order.

    alerts and train_alerts must additionally expose `.short` (TokenizedAlert
    does; the generic GroupableAlert protocol above doesn't require it since
    the other methods don't need it).
    """
    from thesis.grouping.deepcase_grouping import (
        group_alerts_deepcase as _group_alerts_deepcase,
    )

    return _group_alerts_deepcase(alerts, train_alerts, train_id, **kwargs)


# =============================================================================
# Legacy / misc
# =============================================================================


def group_alerts_temporal(
    alerts: list[GroupableAlert],
    session_length: float = CSCAS_SESSION_LENGTH_SECONDS,
    session_timeout: float = CSCAS_SESSION_TIMEOUT_SECONDS,
) -> list[GroupingRecord]:
    """
    Like group_alerts_cscas, but chains sessions across the whole alert
    stream instead of splitting first by host/signature: alerts are sorted
    by timestamp and grouped purely on temporal proximity, capped the same
    way as CSCAS (session_timeout gap, session_length span). Not part of
    the 5-method grouping comparison; kept for other uses in the codebase.
    """
    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    assignments = _chain_by_gap(
        sorted_alerts, gap_threshold=session_timeout, span_cap=session_length
    )
    return [
        GroupingRecord(
            alert_id=alert.alert_id,
            group_id=f"{TEMPORAL_METHOD}:{anchor_id}",
            method=TEMPORAL_METHOD,
        )
        for alert, anchor_id in assignments
    ]


# =============================================================================
# Dispatcher
# =============================================================================


def group_alerts(
    alerts: list[GroupableAlert],
    method: str = FIXED_WINDOW_METHOD,
    **kwargs,
) -> list[GroupingRecord]:
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_window(alerts, **kwargs)
    elif method == FIXED_WINDOW_HOST_METHOD:
        return group_alerts_fixed_window_host(alerts, **kwargs)
    elif method == TIME_DELTA_METHOD:
        return group_alerts_time_delta(alerts, **kwargs)
    elif method == TIME_DELTA_HOST_METHOD:
        return group_alerts_time_delta_host(alerts, **kwargs)
    elif method == CSCAS_METHOD:
        return group_alerts_cscas(alerts, **kwargs)
    elif method == ALERTBERT_METHOD:
        return group_alerts_alertbert(alerts, **kwargs)
    elif method == DEEPCASE_METHOD:
        return group_alerts_deepcase(alerts, **kwargs)
    elif method == TEMPORAL_METHOD:
        return group_alerts_temporal(alerts, **kwargs)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")
