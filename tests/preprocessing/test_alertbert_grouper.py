"""Smoke-tests for the AlertBERT grouping integration.

These tests do NOT require a trained model on disk.  The full-grouper test
is marked xfail unless ALERTBERT_MODEL_ID and ALERTBERT_MODELS_PATH are set
in the environment, so CI stays green without pre-trained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.grouping.alertbert_grouper import (
    ALERTBERT_METHOD,
    _TokenizedAlertDataset,
    AlertBERTGrouper,
)
from thesis.grouping.group_alerts import (
    ALERTBERT_METHOD as DISPATCH_ALERTBERT_METHOD,
    FIXED_WINDOW_METHOD,
    group_alerts,
)
from thesis.schemas.preprocessing import TokenizedAlert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_alert(
    alert_id: str, ts: int, short: str = "SSH", host: str = "h1"
) -> TokenizedAlert:
    return TokenizedAlert(
        alert_id=alert_id,
        ts=ts,
        time_norm=None,
        signature=None,
        ip=None,
        host=host,
        short=short,
    )


SAMPLE_ALERTS = [
    make_alert("a1", ts=1000, short="SSH", host="host1"),
    make_alert("a2", ts=1001, short="FTP", host="host2"),
    make_alert("a3", ts=1003, short="SSH", host="host1"),
    make_alert("a4", ts=2000, short="HTTP", host="host3"),
]


# ---------------------------------------------------------------------------
# _TokenizedAlertDataset unit tests
# ---------------------------------------------------------------------------


class TestTokenizedAlertDataset:
    def setup_method(self):
        self.ds = _TokenizedAlertDataset(SAMPLE_ALERTS)

    def test_len(self):
        assert len(self.ds) == 4

    def test_raw_time_values(self):
        np.testing.assert_array_equal(
            self.ds.data["raw_time"], [1000.0, 1001.0, 1003.0, 2000.0]
        )

    def test_getitem_int(self):
        item = self.ds[0]
        assert item["raw_time"] == 1000.0
        assert item["short"] == "SSH"
        assert item["host"] == "host1"

    def test_getitem_int_cyclic(self):
        # negative index wraps cyclically (AlertBERT padding uses this)
        item_neg = self.ds[-1]
        item_pos = self.ds[len(self.ds) - 1]
        assert item_neg["raw_time"] == item_pos["raw_time"]

    def test_getitem_slice(self):
        result = self.ds[1:3]
        np.testing.assert_array_equal(result["raw_time"], [1001.0, 1003.0])

    def test_getitem_slice_cyclic(self):
        # slice starting before index 0 — used by the first readout window
        result = self.ds[-2:2]
        assert len(result["raw_time"]) == 4  # wraps: indices -2,-1,0,1 → 2,3,0,1

    def test_getitem_empty_slice(self):
        result = self.ds[5:5]
        assert len(result["raw_time"]) == 0

    def test_iter(self):
        items = list(self.ds)
        assert len(items) == 4
        assert items[0]["raw_time"] == 1000.0

    def test_keys(self):
        assert set(self.ds.keys) == {"raw_time", "short", "host"}

    def test_missing_short_host_becomes_unk(self):
        alert = TokenizedAlert(
            alert_id="x",
            ts=0,
            time_norm=None,
            signature=None,
            ip=None,
            host=None,
            short=None,
        )
        ds = _TokenizedAlertDataset([alert])
        assert ds[0]["short"] == "<UNK>"
        assert ds[0]["host"] == "<UNK>"


# ---------------------------------------------------------------------------
# group_alerts dispatcher tests (no model needed)
# ---------------------------------------------------------------------------


def test_fixed_window_still_works():
    records = group_alerts(SAMPLE_ALERTS, method=FIXED_WINDOW_METHOD)
    assert len(records) == 4
    assert all(r.method == FIXED_WINDOW_METHOD for r in records)


def test_alertbert_dispatch_requires_grouper():
    with pytest.raises(ValueError, match="grouper"):
        group_alerts(SAMPLE_ALERTS, method=ALERTBERT_METHOD)


def test_alertbert_method_constant_consistent():
    assert ALERTBERT_METHOD == DISPATCH_ALERTBERT_METHOD


def test_grouper_validates_theta_delta():
    with pytest.raises(ValueError, match="theta"):
        AlertBERTGrouper(
            checkpoint_dir="/tmp/dummy",
            delta=3.0,
            theta=1.0,  # theta < delta → invalid
        )


def test_grouper_group_empty_returns_empty():
    grouper = AlertBERTGrouper(
        checkpoint_dir="/tmp/dummy",
        delta=2.0,
        theta=2.0,
    )
    assert grouper.group([]) == []
