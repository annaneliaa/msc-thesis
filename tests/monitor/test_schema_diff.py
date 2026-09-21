from __future__ import annotations

from datetime import datetime, timezone

from thesis.monitor.schema_diff import diff_dynamic_schemas, zero_diff
from thesis.schemas.dynamic_schema import (
    DynamicCompoundRule,
    DynamicSchema,
    DynamicSinglePredicate,
)

_NOW = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _pred(pid: str, direction: str, version: int) -> DynamicSinglePredicate:
    return DynamicSinglePredicate(
        predicate_id=pid,
        predicate_type="binary",
        field="category",
        operator="==",
        value="EXPLOIT",
        attack_support=0.8,
        benign_support=0.1,
        growth_rate=8.0,
        direction=direction,
        n_attack=10,
        n_benign=2,
        p_value=0.01,
        schema_version=version,
        mined_at=_NOW,
    )


def _rule(rid: str, prediction: str, version: int) -> DynamicCompoundRule:
    return DynamicCompoundRule(
        rule_id=rid,
        conditions=(("category", "==", "EXPLOIT"),),
        prediction=prediction,
        confidence=0.9,
        support_attack=0.9,
        support_benign=0.1,
        n_samples=20,
        schema_version=version,
        mined_at=_NOW,
    )


def _schema(version, preds, rules) -> DynamicSchema:
    return DynamicSchema(
        version=version,
        mined_at=_NOW,
        mining_window_start=_NOW,
        mining_window_end=_NOW,
        base_attack_rate=0.5,
        single_predicates=preds,
        compound_rules=rules,
    )


def test_diff_detects_added_removed_and_unchanged():
    old = _schema(1, [_pred("cat:a", "attack", 1), _pred("cat:b", "attack", 1)], [])
    new = _schema(2, [_pred("cat:a", "attack", 2), _pred("cat:c", "attack", 2)], [])

    diff = diff_dynamic_schemas(old, new)
    assert diff.predicates_added == 1  # cat:c
    assert diff.predicates_removed == 1  # cat:b
    assert diff.predicates_changed == 0  # cat:a unchanged (direction stable)
    assert diff.predicates_before == 2
    assert diff.predicates_after == 2
    assert diff.churn_frac == (1 + 1 + 0) / 2


def test_diff_detects_direction_flip_as_changed():
    old = _schema(1, [_pred("cat:a", "attack", 1)], [])
    new = _schema(2, [_pred("cat:a", "benign", 2)], [])

    diff = diff_dynamic_schemas(old, new)
    assert diff.predicates_added == 0
    assert diff.predicates_removed == 0
    assert diff.predicates_changed == 1
    assert diff.churn_frac == 1.0


def test_diff_rules_tracked_alongside_but_not_in_churn_frac():
    old = _schema(1, [_pred("cat:a", "attack", 1)], [_rule("r1", "attack", 1)])
    new = _schema(
        2,
        [_pred("cat:a", "attack", 2)],
        [_rule("r1", "benign", 2), _rule("r2", "attack", 2)],
    )

    diff = diff_dynamic_schemas(old, new)
    assert diff.rules_added == 1  # r2
    assert diff.rules_removed == 0
    assert diff.rules_changed == 1  # r1 flipped attack->benign
    assert diff.rules_before == 1
    # churn_frac is predicate-only -- zero predicate churn despite rule churn.
    assert diff.churn_frac == 0.0


def test_diff_empty_before_gives_nan_churn():
    old = _schema(1, [], [])
    new = _schema(2, [_pred("cat:a", "attack", 2)], [])

    diff = diff_dynamic_schemas(old, new)
    assert diff.predicates_added == 1
    assert diff.predicates_before == 0
    import math

    assert math.isnan(diff.churn_frac)


def test_zero_diff_matches_no_op_retrain():
    schema = _schema(1, [_pred("cat:a", "attack", 1)], [_rule("r1", "attack", 1)])
    diff = zero_diff(schema)
    assert diff.predicates_added == 0
    assert diff.predicates_removed == 0
    assert diff.predicates_changed == 0
    assert diff.predicates_before == diff.predicates_after == 1
    assert diff.rules_before == diff.rules_after == 1
    assert diff.churn_frac == 0.0
