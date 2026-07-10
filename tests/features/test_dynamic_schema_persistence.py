from datetime import datetime, timezone
from pathlib import Path

from thesis.features.dynamic_schema_persistence import (
    load_dynamic_schema,
    save_dynamic_schema,
)
from thesis.schemas.dynamic_schema import (
    DynamicCompoundRule,
    DynamicSchema,
    DynamicSinglePredicate,
)

_MINED_AT = datetime(2022, 1, 26, 6, 23, 0, 123456, tzinfo=timezone.utc)


def _sample_schema() -> DynamicSchema:
    predicate = DynamicSinglePredicate(
        predicate_id="cat:category=EXPLOIT",
        predicate_type="categorical",
        field="category",
        operator="==",
        value="EXPLOIT",
        attack_support=0.71,
        benign_support=0.008,
        growth_rate=88.75,
        direction="attack",
        n_attack=1254,
        n_benign=9823,
        p_value=0.0001,
        schema_version=1,
        mined_at=_MINED_AT,
    )
    rule = DynamicCompoundRule(
        rule_id="rule:abc123",
        conditions=(
            ("category", "==", "EXPLOIT"),
            ("signature_matches_per_day", "<=", 50.3),
        ),
        prediction="attack",
        confidence=0.891,
        support_attack=0.34,
        support_benign=0.021,
        n_samples=412,
        schema_version=1,
        mined_at=_MINED_AT,
    )
    return DynamicSchema(
        version=1,
        mined_at=_MINED_AT,
        mining_window_start=datetime(2022, 1, 20, tzinfo=timezone.utc),
        mining_window_end=datetime(2022, 1, 21, tzinfo=timezone.utc),
        base_attack_rate=0.42,
        single_predicates=[predicate],
        compound_rules=[rule],
        deployed_at=datetime(2022, 1, 26, 7, 0, 0, tzinfo=timezone.utc),
        superseded_at=None,
    )


def test_save_and_load_round_trip(tmp_path: Path):
    schema = _sample_schema()
    path = tmp_path / "1.json"
    save_dynamic_schema(schema, path)

    loaded = load_dynamic_schema(path)

    assert loaded == schema


def test_round_trip_preserves_none_deployed_at(tmp_path: Path):
    schema = _sample_schema()
    schema_without_deploy = DynamicSchema(
        version=schema.version,
        mined_at=schema.mined_at,
        mining_window_start=schema.mining_window_start,
        mining_window_end=schema.mining_window_end,
        base_attack_rate=schema.base_attack_rate,
        single_predicates=schema.single_predicates,
        compound_rules=schema.compound_rules,
    )
    path = tmp_path / "2.json"
    save_dynamic_schema(schema_without_deploy, path)

    loaded = load_dynamic_schema(path)

    assert loaded.deployed_at is None
    assert loaded.superseded_at is None
