from datetime import datetime, timezone

import pytest

from thesis.monitor.monitor import run_monitor_window
from thesis.monitor.state import MonitorState
from thesis.schemas.dynamic_schema import DynamicCompoundRule, DynamicSchema
from thesis.schemas.groups import AlertGroup

_MINED_AT = datetime(2022, 1, 26, tzinfo=timezone.utc)
_WINDOW_START = datetime(2022, 2, 1, tzinfo=timezone.utc)
_WINDOW_END = datetime(2022, 2, 2, tzinfo=timezone.utc)


def _make_alert_group(group_id: str, label: str, **overrides) -> AlertGroup:
    defaults = dict(
        alert_group_id=group_id,
        group_id=group_id,
        method="cscas_pregrouped",
        start_ts=1_642_636_800,
        end_ts=1_642_636_800,
        n_alerts=1,
        group_label=label,
        category="POLICY",
        ruleset="ET",
        proto=6,
        scas=0,
        cve_refs=set(),
        qualifiers=set(),
        signature_matches_per_day=10.0,
        similarity=0.5,
        signature_id_similarity=0.5,
        attr_similarities={},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )
    defaults.update(overrides)
    return AlertGroup(**defaults)


def _make_schema(version: int = 1) -> DynamicSchema:
    rule = DynamicCompoundRule(
        rule_id="rule:exploit",
        conditions=(("category", "==", "EXPLOIT"),),
        prediction="attack",
        confidence=0.9,
        support_attack=0.9,
        support_benign=0.02,
        n_samples=100,
        schema_version=version,
        mined_at=_MINED_AT,
    )
    return DynamicSchema(
        version=version,
        mined_at=_MINED_AT,
        mining_window_start=_MINED_AT,
        mining_window_end=_MINED_AT,
        base_attack_rate=0.5,
        single_predicates=[],
        compound_rules=[rule],
    )


def _drifting_groups() -> list[AlertGroup]:
    # 20 attack + 10 benign, all category=EXPLOIT -> observed_confidence
    # 20/30 = 0.667 vs the rule's mined confidence 0.9 -> drift 0.233 > 0.10.
    return [
        _make_alert_group(f"a{i}", "attack", category="EXPLOIT") for i in range(20)
    ] + [_make_alert_group(f"b{i}", "benign", category="EXPLOIT") for i in range(10)]


def test_run_monitor_window_no_drift_is_no_action():
    schema = _make_schema()
    # observed_confidence matches mined confidence exactly: 90/100 attack.
    stable_groups = [
        _make_alert_group(f"a{i}", "attack", category="EXPLOIT") for i in range(90)
    ] + [_make_alert_group(f"b{i}", "benign", category="EXPLOIT") for i in range(10)]
    state = MonitorState(scenario_name="cscas", deployed_schema_version=1)

    snapshot = run_monitor_window(
        schema, state, stable_groups, _WINDOW_START, _WINDOW_END
    )

    assert snapshot.signal_2_elevated is False
    assert snapshot.n_elevated == 0
    assert snapshot.trigger_remine is False
    assert snapshot.action == "NO_ACTION"
    assert snapshot.n_incoming_groups == 100
    assert snapshot.n_labeled_groups == 100
    assert snapshot.window_start == _WINDOW_START
    assert snapshot.window_end == _WINDOW_END


def test_run_monitor_window_consecutive_drift_triggers_remine():
    schema = _make_schema()
    state = MonitorState(scenario_name="cscas", deployed_schema_version=1)
    groups = _drifting_groups()

    snapshot_1 = run_monitor_window(schema, state, groups, _WINDOW_START, _WINDOW_END)
    assert snapshot_1.signal_2_elevated is True
    assert snapshot_1.trigger_remine is False
    assert snapshot_1.action == "SOFT_ALERT"
    assert state.consecutive_signal_2_elevated == 1

    snapshot_2 = run_monitor_window(schema, state, groups, _WINDOW_START, _WINDOW_END)
    assert snapshot_2.trigger_remine is False
    assert snapshot_2.action == "SOFT_ALERT"
    assert state.consecutive_signal_2_elevated == 2

    snapshot_3 = run_monitor_window(schema, state, groups, _WINDOW_START, _WINDOW_END)
    assert snapshot_3.trigger_remine is True
    assert snapshot_3.action == "REMINE_AND_RETRAIN"
    assert state.consecutive_signal_2_elevated == 3
    assert state.windows_observed == 3
    assert snapshot_3.state_after is state


def test_run_monitor_window_rejects_stale_state_for_new_schema_version():
    schema = _make_schema(version=2)
    state = MonitorState(scenario_name="cscas", deployed_schema_version=1)

    with pytest.raises(ValueError):
        run_monitor_window(schema, state, [], _WINDOW_START, _WINDOW_END)
