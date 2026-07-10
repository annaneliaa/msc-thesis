from datetime import datetime, timezone

import pytest

from thesis.monitor.signals import compute_psi, compute_signal_1, compute_signal_2
from thesis.schemas.dynamic_schema import (
    DynamicCompoundRule,
    DynamicSchema,
    DynamicSinglePredicate,
)
from thesis.schemas.groups import AlertGroup

_MINED_AT = datetime(2022, 1, 26, tzinfo=timezone.utc)


def _make_alert_group(group_id: str, label: str | None, **overrides) -> AlertGroup:
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


def test_compute_psi_zero_when_equal():
    assert compute_psi(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_compute_psi_below_elevated_threshold():
    # fired = 0.1*ln(2), not_fired = -0.1*ln(0.8/0.9) -- hand-computed.
    psi = compute_psi(0.1, 0.2)
    assert psi == pytest.approx(0.081093, abs=1e-5)
    assert psi < 0.1


def test_compute_psi_crosses_elevated_not_significant_threshold():
    psi = compute_psi(0.05, 0.15)
    assert psi == pytest.approx(0.120984, abs=1e-5)
    assert 0.1 < psi < 0.2


def _make_schema(
    single_predicates=None, compound_rules=None, base_attack_rate=0.5
) -> DynamicSchema:
    return DynamicSchema(
        version=1,
        mined_at=_MINED_AT,
        mining_window_start=_MINED_AT,
        mining_window_end=_MINED_AT,
        base_attack_rate=base_attack_rate,
        single_predicates=single_predicates or [],
        compound_rules=compound_rules or [],
    )


def _exploit_predicate(**overrides) -> DynamicSinglePredicate:
    defaults = dict(
        predicate_id="cat:category=EXPLOIT",
        predicate_type="categorical",
        field="category",
        operator="==",
        value="EXPLOIT",
        attack_support=0.8,
        benign_support=0.1,
        growth_rate=8.0,
        direction="attack",
        n_attack=80,
        n_benign=10,
        p_value=0.001,
        schema_version=1,
        mined_at=_MINED_AT,
    )
    defaults.update(overrides)
    return DynamicSinglePredicate(**defaults)


def test_compute_signal_1_observed_rate_matches_incoming_fires():
    schema = _make_schema(
        single_predicates=[_exploit_predicate()], base_attack_rate=0.5
    )
    groups = [
        _make_alert_group(f"g{i}", "attack", category="EXPLOIT" if i < 3 else "SNMP")
        for i in range(10)
    ]

    results = compute_signal_1(schema, groups)

    assert len(results) == 1
    result = results[0]
    assert result.predicate_id == "cat:category=EXPLOIT"
    assert result.p_expected == pytest.approx(0.8 * 0.5 + 0.1 * 0.5)
    assert result.p_observed == pytest.approx(0.3)
    assert result.n_observed == 10
    assert result.psi == pytest.approx(
        compute_psi(result.p_expected, result.p_observed)
    )


def test_compute_signal_1_empty_window_observed_rate_zero():
    schema = _make_schema(single_predicates=[_exploit_predicate()])
    results = compute_signal_1(schema, [])
    assert results[0].p_observed == 0.0
    assert results[0].n_observed == 0


def _exploit_rule(**overrides) -> DynamicCompoundRule:
    defaults = dict(
        rule_id="rule:abc",
        conditions=(("category", "==", "EXPLOIT"),),
        prediction="attack",
        confidence=0.9,
        support_attack=0.9,
        support_benign=0.02,
        n_samples=100,
        schema_version=1,
        mined_at=_MINED_AT,
    )
    defaults.update(overrides)
    return DynamicCompoundRule(**defaults)


def test_compute_signal_2_calibration_drift_detected():
    schema = _make_schema(compound_rules=[_exploit_rule(confidence=0.9)])
    groups = [
        _make_alert_group(f"a{i}", "attack", category="EXPLOIT") for i in range(12)
    ] + [_make_alert_group(f"b{i}", "benign", category="EXPLOIT") for i in range(8)]

    results = compute_signal_2(schema, groups, min_samples=5)

    assert len(results) == 1
    result = results[0]
    assert result.n_matching == 20
    assert result.observed_confidence == pytest.approx(0.6)
    assert result.drift == pytest.approx(0.3)
    assert result.elevated is True


def test_compute_signal_2_below_min_samples_skips():
    schema = _make_schema(compound_rules=[_exploit_rule(confidence=0.9)])
    groups = [
        _make_alert_group(f"a{i}", "attack", category="EXPLOIT") for i in range(3)
    ]

    results = compute_signal_2(schema, groups, min_samples=5)

    assert results[0].observed_confidence is None
    assert results[0].drift is None
    assert results[0].elevated is False
    assert results[0].n_matching == 3


def test_compute_signal_2_ignores_non_matching_groups():
    schema = _make_schema(compound_rules=[_exploit_rule(confidence=0.9)])
    groups = (
        [_make_alert_group(f"a{i}", "attack", category="EXPLOIT") for i in range(9)]
        + [
            _make_alert_group(f"a{i}", "attack", category="EXPLOIT")
            for i in range(9, 10)
        ]
        + [_make_alert_group("other", "benign", category="SNMP")]
    )

    results = compute_signal_2(schema, groups, min_samples=5)

    assert results[0].n_matching == 10
    assert results[0].observed_confidence == pytest.approx(1.0)
    assert results[0].drift == pytest.approx(0.1)
