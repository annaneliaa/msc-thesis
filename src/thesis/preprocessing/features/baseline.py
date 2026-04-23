from typing import Any
import ipaddress

from thesis.schemas.preprocessing import GroupSnapshot


def _count_items_with_prefix(items: set[str], prefix: str) -> int:
    return sum(1 for item in items if item.startswith(prefix))


def _is_internal_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def compute_baseline_features(group: GroupSnapshot) -> dict[str, Any]:
    """
    Compute baseline transaction-level features from a GroupSnapshot.
    """
    items = group.items or set()

    duration_sec = max(0, int(group.end_ts) - int(group.start_ts))
    n_alerts = int(group.n_alerts)
    n_items = len(items)

    ip_values = group.alert_ips
    n_internal_ips = sum(1 for ip in ip_values if _is_internal_ip(ip))
    n_external_ips = sum(1 for ip in ip_values if not _is_internal_ip(ip))

    features: dict[str, Any] = {
        "duration_sec": duration_sec,
        "n_alerts": n_alerts,
        "n_items": n_items,
        "n_hosts": _count_items_with_prefix(items, "host:"),
        "n_shorts": _count_items_with_prefix(items, "short:"),
        "n_sigs": _count_items_with_prefix(items, "sig:"),
        "n_internal_ips": n_internal_ips,
        "n_external_ips": n_external_ips,
        "alerts_per_second": n_alerts / max(1, duration_sec),
    }

    return features
