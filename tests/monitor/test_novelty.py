import dataclasses
from datetime import datetime, timezone

import pytest

from thesis.mining.common.attribute_features import compute_candidate_attribute_features
from thesis.monitor.novelty import (
    build_novelty_reference,
    compute_novelty,
    firing_pattern,
)
from thesis.schemas.dynamic_schema import DynamicSchema, DynamicSinglePredicate
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


def _make_schema(single_predicates) -> DynamicSchema:
    return DynamicSchema(
        version=1,
        mined_at=_MINED_AT,
        mining_window_start=_MINED_AT,
        mining_window_end=_MINED_AT,
        base_attack_rate=0.5,
        single_predicates=single_predicates,
        compound_rules=[],
    )


def _schema(version: int = 1):
    return _make_schema(
        single_predicates=[
            _exploit_predicate(),
            _exploit_predicate(
                predicate_id="num:signature_matches_per_day_gt",
                predicate_type="numeric_threshold",
                field="signature_matches_per_day",
                operator=">",
                value=100.0,
            ),
        ]
    )


def test_firing_pattern_lists_holding_predicates():
    schema = _schema()
    g = _make_alert_group(
        "g", "attack", category="EXPLOIT", signature_matches_per_day=500.0
    )
    feats = compute_candidate_attribute_features(g)
    assert firing_pattern(schema, feats) == frozenset(
        {"cat:category=EXPLOIT", "num:signature_matches_per_day_gt"}
    )


def test_novelty_rates_against_reference():
    schema = _schema()
    ref_groups = [
        _make_alert_group("r1", "attack", category="EXPLOIT"),  # {EXPLOIT}
        _make_alert_group("r2", "benign", category="SNMP"),  # {} (no fire)
        _make_alert_group("r3", "benign", category="SNMP"),  # {}
        _make_alert_group("r4", "benign", category="SNMP"),  # {}
    ]
    ref = build_novelty_reference(schema, ref_groups)
    assert ref.no_fire_rate == pytest.approx(0.75)
    assert len(ref.patterns) == 2

    window = [
        _make_alert_group("w1", "attack", category="EXPLOIT"),  # seen
        # EXPLOIT + high volume: never seen together -> novel
        _make_alert_group(
            "w2", "attack", category="EXPLOIT", signature_matches_per_day=500.0
        ),
        _make_alert_group(
            "w3", "benign", category="SNMP", signature_matches_per_day=500.0
        ),  # novel
        _make_alert_group("w4", "benign", category="SNMP"),  # seen (no fire)
    ]
    sig = compute_novelty(schema, ref, window)
    assert sig.n_groups == 4
    assert sig.pattern_novelty_rate == pytest.approx(0.5)
    assert sig.no_fire_rate == pytest.approx(0.25)
    assert sig.reference_no_fire_rate == pytest.approx(0.75)
    assert sig.n_novel_patterns == 2
    assert sig.pattern_novelty_rate_attack == pytest.approx(0.5)
    assert sig.pattern_novelty_rate_benign == pytest.approx(0.5)


def test_novelty_unlabeled_window_has_no_per_class_rates():
    schema = _schema()
    ref = build_novelty_reference(schema, [_make_alert_group("r", "benign")])
    sig = compute_novelty(schema, ref, [_make_alert_group("w", None)])
    assert sig.pattern_novelty_rate_attack is None
    assert sig.pattern_novelty_rate_benign is None


def test_novelty_rejects_reference_from_other_schema_version():
    ref = build_novelty_reference(_schema(), [])
    other = dataclasses.replace(_schema(), version=2)
    with pytest.raises(ValueError):
        compute_novelty(other, ref, [])
