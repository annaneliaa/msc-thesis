from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis.schemas.dynamic_schema import (
    DynamicCompoundRule,
    DynamicSchema,
    DynamicSinglePredicate,
)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def save_dynamic_schema(schema: DynamicSchema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(schema)
    payload["mined_at"] = _iso(schema.mined_at)
    payload["mining_window_start"] = _iso(schema.mining_window_start)
    payload["mining_window_end"] = _iso(schema.mining_window_end)
    payload["deployed_at"] = _iso(schema.deployed_at)
    payload["superseded_at"] = _iso(schema.superseded_at)
    for pred_payload, pred in zip(
        payload["single_predicates"], schema.single_predicates
    ):
        pred_payload["mined_at"] = _iso(pred.mined_at)
    for rule_payload, rule in zip(payload["compound_rules"], schema.compound_rules):
        rule_payload["mined_at"] = _iso(rule.mined_at)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_dynamic_schema(path: Path) -> DynamicSchema:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    single_predicates = [
        DynamicSinglePredicate(
            predicate_id=item["predicate_id"],
            predicate_type=item["predicate_type"],
            field=item["field"],
            operator=item["operator"],
            value=item["value"],
            attack_support=item["attack_support"],
            benign_support=item["benign_support"],
            growth_rate=item["growth_rate"],
            direction=item["direction"],
            n_attack=item["n_attack"],
            n_benign=item["n_benign"],
            p_value=item["p_value"],
            schema_version=item["schema_version"],
            mined_at=datetime.fromisoformat(item["mined_at"]),
        )
        for item in payload["single_predicates"]
    ]

    compound_rules = [
        DynamicCompoundRule(
            rule_id=item["rule_id"],
            conditions=tuple(tuple(c) for c in item["conditions"]),
            prediction=item["prediction"],
            confidence=item["confidence"],
            support_attack=item["support_attack"],
            support_benign=item["support_benign"],
            n_samples=item["n_samples"],
            schema_version=item["schema_version"],
            mined_at=datetime.fromisoformat(item["mined_at"]),
        )
        for item in payload["compound_rules"]
    ]

    return DynamicSchema(
        version=payload["version"],
        mined_at=datetime.fromisoformat(payload["mined_at"]),
        mining_window_start=datetime.fromisoformat(payload["mining_window_start"]),
        mining_window_end=datetime.fromisoformat(payload["mining_window_end"]),
        base_attack_rate=payload["base_attack_rate"],
        single_predicates=single_predicates,
        compound_rules=compound_rules,
        deployed_at=(
            datetime.fromisoformat(payload["deployed_at"])
            if payload.get("deployed_at") is not None
            else None
        ),
        superseded_at=(
            datetime.fromisoformat(payload["superseded_at"])
            if payload.get("superseded_at") is not None
            else None
        ),
    )
