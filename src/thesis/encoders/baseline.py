from typing import Any, Iterable
import ipaddress
import pandas as pd

from thesis.schemas.groups import AlertGroup


def _count_items_with_prefix(items: set[str], prefix: str) -> int:
    return sum(1 for item in items if item.startswith(prefix))


def _is_internal_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def compute_baseline_features(tx: AlertGroup) -> dict[str, Any]:
    items = set(tx.abs_items or tx.raw_items or [])
    ip_values = list(tx.alert_ips or [])

    duration_sec = max(0, int(tx.end_ts) - int(tx.start_ts))
    n_alerts = int(tx.n_alerts)
    n_items = len(items)

    n_internal_ips = sum(1 for ip in ip_values if _is_internal_ip(ip))
    n_external_ips = sum(1 for ip in ip_values if not _is_internal_ip(ip))

    return {
        "duration_sec": duration_sec,
        # "n_alerts": n_alerts,
        "n_items": n_items,
        "n_hosts": _count_items_with_prefix(items, "host:"),
        "n_shorts": _count_items_with_prefix(items, "short:"),
        "n_sigs": _count_items_with_prefix(items, "sig:"),
        "n_internal_ips": n_internal_ips,
        "n_external_ips": n_external_ips,
        "alerts_per_second": n_alerts / max(1, duration_sec),
    }


class BaselineFeatureEncoder:
    """
    Stateless baseline feature encoder for training and inference.
    """

    def transform_one(self, tx: AlertGroup) -> pd.DataFrame:
        """
        Encode one alert_group into a 1-row feature DataFrame.
        """
        features = compute_baseline_features(tx)
        return pd.DataFrame([features])

    def transform(self, alert_groups: Iterable[AlertGroup]) -> pd.DataFrame:
        """
        Encode many alert_groups into a feature DataFrame.
        """
        rows = [compute_baseline_features(tx) for tx in alert_groups]
        return pd.DataFrame(rows)
