import pandas as pd

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.common.util import sequence_contains_pattern
from thesis.mining.temporal.lookback import (
    attach_lookback_steps,
    lookback_steps,
    HostAlertIndex,
)
from thesis.schemas.features import SymbolicFeature, SymbolicFeatureSchema
from thesis.schemas.groups import AlertGroup

T0 = 1_642_204_800  # a minute boundary


def _alerts():
    return pd.DataFrame(
        [
            (T0 - 900, "web", "OLD"),  # 15 min before: outside a 10-min window
            (T0 - 300, "web", "SCAN"),
            (T0 - 290, "web", "SCAN2"),  # same minute bucket as SCAN
            (T0 - 120, "mail", "OTHERHOST"),  # different host
            (T0 - 60, "web", "LOGIN"),
            (T0, "web", "EXPLOIT"),  # the group itself
            (T0 + 30, "web", "EXPLOIT"),
            (T0 + 600, "web", "LATER"),  # after the group ends
        ],
        columns=["time", "host", "short"],
    )


def _group(host="web", start=T0, end=T0 + 30):
    return AlertGroup(
        alert_group_id="g",
        group_id="g",
        method="cscas_grouping",
        start_ts=start,
        end_ts=end,
        n_alerts=2,
        raw_items={f"host:{host}", "short:EXPLOIT"},
    )


def test_lookback_window_host_and_buckets():
    steps = lookback_steps(_group(), HostAlertIndex(_alerts()), window_minutes=10)
    assert steps == [
        {"short:SCAN", "short:SCAN2"},
        {"short:LOGIN"},
        {"short:EXPLOIT"},  # T0 and T0+30 share a minute bucket
    ]


def test_lookback_handles_missing_host_and_open_group():
    index = HostAlertIndex(_alerts())
    assert lookback_steps(_group(host="nohost"), index) == []
    open_group = _group(end=None)
    assert lookback_steps(open_group, index, window_minutes=1)[-1] == {"short:EXPLOIT"}


def test_encoder_uses_lookback_sequence_when_attached():
    g = _group()
    attach_lookback_steps([g], _alerts(), window_minutes=10)
    pattern = ("short:SCAN", "short:LOGIN", "short:EXPLOIT")
    assert sequence_contains_pattern(g.context_steps, pattern)
    schema = SymbolicFeatureSchema(
        schema_name="s",
        schema_version="1",
        features=[
            SymbolicFeature(
                feature_name="seq__x",
                itemset=pattern,
                source_label="attack",
                mining_type="item_sequence",
            ),
            SymbolicFeature(
                feature_name="seq__rev",
                itemset=pattern[::-1],
                source_label="attack",
                mining_type="item_sequence",
            ),
        ],
    )
    row = SymbolicFeatureEncoder(schema).transform_one(g).iloc[0]
    assert row["seq__x"] == 1
    assert row["seq__rev"] == 0
