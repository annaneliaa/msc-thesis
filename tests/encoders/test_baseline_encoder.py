"""Base features are picked by dataset: CSCAS pre-grouped rows get the CSV
columns, every AIT-ADS grouping (including the CSCAS-style cscas_grouping)
gets the AIT-ADS summary features."""

import pytest

from thesis.encoders.baseline import compute_baseline_features
from thesis.schemas.groups import AlertGroup


def _group(method: str) -> AlertGroup:
    return AlertGroup(
        alert_group_id="g",
        group_id="g",
        method=method,
        start_ts=1_642_215_600,  # 03:00 UTC
        end_ts=1_642_215_600,
        n_alerts=4,
        group_label="benign",
        raw_items={"host:h1", "short:A", "sig:a", "sig:b"},
        proto=6,
    )


def test_cscas_pregrouped_gets_cscas_features():
    f = compute_baseline_features(_group("cscas_pregrouped"))
    assert set(f) == {
        "proto",
        "ext_port",
        "int_port",
        "n_alerts",
        "signature_matches_per_day",
    }


@pytest.mark.parametrize(
    "method", ["cscas_grouping", "fixed_window", "time_delta", "alertbert"]
)
def test_ait_ads_groupings_get_ait_ads_features(method):
    f = compute_baseline_features(_group(method))
    assert f == {
        "hour_of_day": 3,
        "n_alerts": 4,
        "n_hosts": 1,
        "n_shorts": 1,
        "n_sigs": 2,
    }
