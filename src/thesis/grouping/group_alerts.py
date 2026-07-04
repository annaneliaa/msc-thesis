from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from thesis.schemas.groups import AlertGroup, GroupingRecord
from thesis.schemas.preprocessing import ParsedSuricataGroup, TokenizedAlert

if TYPE_CHECKING:
    from thesis.grouping.alertbert_grouper import AlertBERTGrouper

FIXED_WINDOW_SECONDS = 2
FIXED_WINDOW_METHOD = "fixed_window"
FIXED_WINDOW_HOST_METHOD = "fixed_window_host"
TIME_DELTA_METHOD = "time_delta"
TIME_DELTA_HOST_METHOD = "time_delta_host"
TIME_DELTA_SECONDS = 2.0
ALERTBERT_METHOD = "alertbert"
CSCAS_PREGROUPED_METHOD = "cscas_pregrouped"
CSCAS_METHOD = "cscas_grouping"
CSCAS_SESSION_LENGTH_SECONDS = 300.0
CSCAS_SESSION_TIMEOUT_SECONDS = 2.0
CSCAS_TARGET_WINDOW_METHOD = "cscas_target_window"
CSCAS_TARGET_WINDOW_SECONDS = 3600.0  # one hour
CSCAS_TARGET_SESSION_METHOD = "cscas_target_session"
CSCAS_TARGET_SESSION_TIMEOUT_SECONDS = 1800.0  # 30 min quiet gap closes a basket
CSCAS_TARGET_SESSION_LENGTH_SECONDS = (
    21600.0  # 6h hard cap for continuously-active targets
)


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


def group_alerts_fixed_window_host(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    """Per-host variant of fixed_window.

    Each alert is assigned to a group identified by (host, window_id) so that
    alerts from different machines are never merged even when they fall in the
    same calendar window. The host component is taken from alert.host; alerts
    with no host value are treated as a single anonymous host.
    """
    records: list[GroupingRecord] = []
    for alert in alerts:
        host = alert.host or "_unknown"
        window_id = alert.ts // window_size
        group_id = f"fixed_window_host:{host}:{window_id}"
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=group_id,
                method=FIXED_WINDOW_HOST_METHOD,
            )
        )
    return records


def group_alerts_time_delta(
    alerts: list[TokenizedAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """Landauer et al. (2022) time-delta method.

    Sorts alerts by timestamp and starts a new group whenever the gap to the
    previous alert exceeds delta seconds. Equivalent to connected-components
    clustering with a time-only distance threshold on a 1-D sorted sequence.
    Unlike fixed_window, groups are bounded by stream gaps rather than absolute
    calendar windows, so a continuous stream with gaps < delta forms one group.
    """
    if not alerts:
        return []

    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    anchor_id = sorted_alerts[0].alert_id
    prev_ts: float = sorted_alerts[0].ts

    records: list[GroupingRecord] = []
    for alert in sorted_alerts:
        if alert.ts - prev_ts > delta:
            anchor_id = alert.alert_id
        prev_ts = alert.ts
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=f"time_delta:{anchor_id}",
                method=TIME_DELTA_METHOD,
            )
        )
    return records


def group_alerts_time_delta_host(
    alerts: list[TokenizedAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """Per-host variant of the Landauer et al. time-delta method.

    Runs the same stream-gap scan as group_alerts_time_delta but independently
    per host, so a dense stream on one machine cannot absorb alerts from other
    machines. The host component is taken from alert.host; alerts with no host
    value are treated as a single anonymous host.
    """
    if not alerts:
        return []

    by_host: dict[str, list[TokenizedAlert]] = defaultdict(list)
    for alert in alerts:
        by_host[alert.host or "_unknown"].append(alert)

    records: list[GroupingRecord] = []
    for host, host_alerts in by_host.items():
        sorted_alerts = sorted(host_alerts, key=lambda a: a.ts)
        anchor_id = sorted_alerts[0].alert_id
        prev_ts: float = sorted_alerts[0].ts
        for alert in sorted_alerts:
            if alert.ts - prev_ts > delta:
                anchor_id = alert.alert_id
            prev_ts = alert.ts
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"time_delta_host:{host}:{anchor_id}",
                    method=TIME_DELTA_HOST_METHOD,
                )
            )
    return records


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


def _drop_unresolvable_int_ip_rows(
    parsed_rows: list[ParsedSuricataGroup], context: str
) -> list[ParsedSuricataGroup]:
    """
    Drop rows with IntIP == -1 (row.int_ip is None), shared by both
    target-keyed CSCAS grouping schemes -- see group_cscas_rows_by_target_window's
    docstring for why these are excluded rather than bucketed by external actor.
    """
    usable_rows = [row for row in parsed_rows if row.int_ip is not None]
    n_dropped = len(parsed_rows) - len(usable_rows)
    if n_dropped:
        print(
            f"  [warn] Dropped {n_dropped}/{len(parsed_rows)} rows with "
            f"IntIP == -1 (session touched multiple internal IPs; not "
            f"resolvable to one target) before {context} grouping."
        )
    return usable_rows


def _build_cscas_target_alert_group(
    int_ip: str,
    rows: list[ParsedSuricataGroup],
    method: str,
    basket_idx: int,
) -> AlertGroup:
    """Build one AlertGroup basket from rows already assigned to (int_ip, basket_idx)."""
    rows_sorted = sorted(rows, key=lambda r: r.ts)
    group_id = f"{method}:{int_ip}:{basket_idx}"

    items: set[str] = set()
    sorted_items: list[set[str]] = []
    alert_ips: set[str] = set()
    alert_labels: set[str] = set()
    n_alerts = 0

    for row in rows_sorted:
        items |= row.tokens
        sorted_items.append(set(row.tokens))
        alert_ips.add(row.ext_ip)
        alert_labels.add(row.label)
        n_alerts += row.n_alerts

    return AlertGroup(
        alert_group_id=group_id,
        group_id=group_id,
        method=method,
        start_ts=rows_sorted[0].ts,
        end_ts=rows_sorted[-1].ts,
        n_alerts=n_alerts,
        abs_items=items,
        raw_items=set(items),
        sorted_items=sorted_items,
        alert_ips=alert_ips,
        group_label=(
            "mixed"
            if "benign" in alert_labels and "attack" in alert_labels
            else ("attack" if "attack" in alert_labels else "benign")
        ),
        alert_labels=alert_labels,
        weight=1.0,
        int_ip=int_ip,
    )


def group_cscas_rows_by_target_window(
    parsed_rows: list[ParsedSuricataGroup],
    window_seconds: float = CSCAS_TARGET_WINDOW_SECONDS,
) -> list[AlertGroup]:
    """
    Aggregate CSCAS rows into baskets keyed by (target IP, fixed time window).

    Every CSCAS CSV row is already a single-signature x single-external-IP
    cluster (see ingest_cscas_scenario / CSCAS_PREGROUPED_METHOD), so mining
    one basket per row can only ever rediscover a single signature's own
    description decomposed into words -- every "co-occurring" token comes
    from the same fixed string, so any itemset mined this way is tautological
    and every subset of it carries identical support.

    Grouping by internal target (the host actually being probed/attacked)
    within a bounded time window instead produces baskets that can span
    multiple distinct signatures fired against the same target. That is what
    makes itemset mining meaningful here -- "did a recon signature and an
    exploit signature both hit this host in this window" -- and it also
    populates sorted_items (one token set per row, timestamp-ordered) so
    PrefixSpan sequence mining has real cross-signature order to work with,
    instead of the always-empty sorted_items CSCAS_PREGROUPED_METHOD produces.

    Rows with IntIP == -1 (~24% of the CSCAS CSV) are dropped rather than
    bucketed by external actor. IntIP == -1 means the (signature, external
    host) session that CSCAS pre-aggregated this row from touched more than
    one internal IP -- which specific IPs is not recoverable from this CSV,
    that granularity was already lost upstream. Falling back to an
    actor-keyed basket for these rows was tried and rejected: those baskets
    turned out *less* likely to be genuinely multi-signature than target-keyed
    ones (4.7% vs 8.7% in a full-dataset check), consistent with the paper's
    own framing of high-fanout single-signature sessions as low-value bulk
    scanning rather than multi-stage attacks. Keeping them out avoids mixing
    two different units of analysis (actor-hours vs. target-hours) into one
    mined corpus, at the cost of excluding those rows from this grouping
    method entirely -- they remain available via CSCAS_PREGROUPED_METHOD.

    Caveat this scheme has and group_cscas_rows_by_target_session doesn't:
    a fixed calendar window forces every basket to wait for its window to
    close before it's "complete," regardless of whether the target is
    actually still active -- e.g. a 6h window means up to a 6h detection
    delay even for a target that only generated one signature and went quiet
    seconds later. See group_cscas_rows_by_target_session for the session-gap
    alternative, which closes a basket as soon as the target goes quiet and
    only pays the full delay for targets that are continuously active.
    """
    if not parsed_rows:
        return []

    usable_rows = _drop_unresolvable_int_ip_rows(parsed_rows, "target-window")

    buckets: dict[tuple[str, int], list[ParsedSuricataGroup]] = defaultdict(list)
    for row in usable_rows:
        window_id = int(row.ts // window_seconds)
        buckets[(row.int_ip, window_id)].append(row)

    groups = [
        _build_cscas_target_alert_group(
            key_ip, rows, CSCAS_TARGET_WINDOW_METHOD, window_id
        )
        for (key_ip, window_id), rows in buckets.items()
    ]
    groups.sort(key=lambda g: g.start_ts)
    return groups


def group_cscas_rows_by_target_session(
    parsed_rows: list[ParsedSuricataGroup],
    session_timeout: float = CSCAS_TARGET_SESSION_TIMEOUT_SECONDS,
    session_length: float = CSCAS_TARGET_SESSION_LENGTH_SECONDS,
) -> list[AlertGroup]:
    """
    Aggregate CSCAS rows into baskets keyed by target IP, using a session-gap
    scheme instead of group_cscas_rows_by_target_window's fixed calendar
    windows -- the same (host, signature) session-gap idea group_alerts_cscas
    uses, but keyed by internal target alone so a basket can still span
    multiple signatures.

    For each internal target IP, rows are sorted by time and a new basket
    starts whenever the gap to the previous row exceeds session_timeout, or
    the current basket's span would exceed session_length (whichever comes
    first) -- mirroring the paper's own SessionTimeout/SessionLength
    semantics ("reported if it has not been updated for more than
    SessionTimeout", "forced after SessionLength expires").

    Why this instead of a fixed window: a fixed window forces every basket,
    including a target that only ever sees one signature, to wait for the
    full window to close before it can be reported -- with a 6h window
    that's a 6h worst case for every target, not just the ones exhibiting
    genuinely extended multi-stage activity. Under this scheme, a target
    that goes quiet closes its basket after session_timeout regardless of
    where it falls in the window; session_length only bounds the minority
    of targets that stay continuously active. The actual latency this
    produces should be measured empirically as basket duration
    (end_ts - start_ts), not assumed from the nominal session_length cap.

    Same IntIP == -1 exclusion and rationale as group_cscas_rows_by_target_window.
    """
    if not parsed_rows:
        return []

    usable_rows = _drop_unresolvable_int_ip_rows(parsed_rows, "target-session")

    by_ip: dict[str, list[ParsedSuricataGroup]] = defaultdict(list)
    for row in usable_rows:
        by_ip[row.int_ip].append(row)

    groups: list[AlertGroup] = []
    for int_ip, ip_rows in by_ip.items():
        rows_sorted = sorted(ip_rows, key=lambda r: r.ts)

        session_rows: list[ParsedSuricataGroup] = [rows_sorted[0]]
        session_start_ts = rows_sorted[0].ts
        prev_ts = rows_sorted[0].ts
        basket_idx = 0

        for row in rows_sorted[1:]:
            if (
                row.ts - prev_ts > session_timeout
                or row.ts - session_start_ts > session_length
            ):
                groups.append(
                    _build_cscas_target_alert_group(
                        int_ip, session_rows, CSCAS_TARGET_SESSION_METHOD, basket_idx
                    )
                )
                basket_idx += 1
                session_rows = []
                session_start_ts = row.ts

            session_rows.append(row)
            prev_ts = row.ts

        groups.append(
            _build_cscas_target_alert_group(
                int_ip, session_rows, CSCAS_TARGET_SESSION_METHOD, basket_idx
            )
        )

    groups.sort(key=lambda g: g.start_ts)
    return groups


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    grouper: AlertBERTGrouper | None = None,
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
        if grouper is None:
            raise ValueError(
                "grouper must be a loaded AlertBERTGrouper when method='alertbert'"
            )
        return grouper.group(alerts)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")
