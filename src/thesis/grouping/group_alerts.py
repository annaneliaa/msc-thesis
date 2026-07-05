from __future__ import annotations

from collections import defaultdict
from thesis.schemas.groups import GroupingRecord
from thesis.schemas.preprocessing import TokenizedAlert


FIXED_WINDOW_SECONDS = 2
FIXED_WINDOW_METHOD = "fixed_window"
CSCAS_PREGROUPED_METHOD = "cscas_pregrouped"
CSCAS_METHOD = "cscas_grouping"
CSCAS_SESSION_LENGTH_SECONDS = 300.0
CSCAS_SESSION_TIMEOUT_SECONDS = 2.0


def fixed_window_group_id(ts: int, window_size: int = FIXED_WINDOW_SECONDS) -> str:
    window_id = ts // window_size
    return f"fixed_window:{window_id}"


def group_alert_fixed_window(
    alert: TokenizedAlert,
    window_size: int = FIXED_WINDOW_SECONDS,
) -> GroupingRecord:
    return GroupingRecord(
        alert_id=alert.alert_id,
        group_id=fixed_window_group_id(alert.ts, window_size=window_size),
        method=FIXED_WINDOW_METHOD,
    )


def group_alerts_fixed_window(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    return [
        group_alert_fixed_window(alert, window_size=window_size) for alert in alerts
    ]


def group_alerts_fixed_window_by_group(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[TokenizedAlert]]:
    groups: dict[str, list[TokenizedAlert]] = defaultdict(list)

    for alert in alerts:
        group_id = fixed_window_group_id(alert.ts, window_size=window_size)
        groups[group_id].append(alert)

    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


def group_alerts_cscas(
    alerts: list[TokenizedAlert],
    session_length: float = CSCAS_SESSION_LENGTH_SECONDS,
    session_timeout: float = CSCAS_SESSION_TIMEOUT_SECONDS,
) -> list[GroupingRecord]:
    """
    Like time_delta_host, but additionally splits streams by signature, so an
    alert group never mixes alerts from different signatures. A new session
    starts whenever the gap to the previous alert exceeds session_timeout
    seconds, or the session's total span would exceed session_length seconds
    (whichever comes first). The host component is taken from alert.host and
    the signature from alert.signature; missing values are treated as a
    single anonymous host/signature.
    """
    if not alerts:
        return []

    by_key: dict[tuple[str, str], list[TokenizedAlert]] = defaultdict(list)
    for alert in alerts:
        host = alert.host or "_unknown"
        signature = alert.signature or "_unknown"
        by_key[(host, signature)].append(alert)

    records: list[GroupingRecord] = []
    for (host, signature), key_alerts in by_key.items():
        sorted_alerts = sorted(key_alerts, key=lambda a: a.ts)
        anchor_id = sorted_alerts[0].alert_id
        anchor_ts: float = sorted_alerts[0].ts
        prev_ts: float = anchor_ts
        for alert in sorted_alerts:
            if alert.ts - prev_ts > session_timeout or (
                alert.ts - anchor_ts > session_length
            ):
                anchor_id = alert.alert_id
                anchor_ts = alert.ts
            prev_ts = alert.ts
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"cscas_grouping:{host}:{signature}:{anchor_id}",
                    method=CSCAS_METHOD,
                )
            )
    return records


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    **kwargs,
) -> list[GroupingRecord]:
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_window(alerts, **kwargs)
    elif method == CSCAS_METHOD:
        return group_alerts_cscas(alerts, **kwargs)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")
