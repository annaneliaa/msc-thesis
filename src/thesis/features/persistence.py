import json
from dataclasses import asdict
from pathlib import Path

from thesis.schemas.features import (
    AttributePredicate,
    SymbolicFeatureSchema,
    SymbolicFeature,
)


def save_symbolic_feature_schema(schema: SymbolicFeatureSchema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)


def load_symbolic_feature_schema(path: Path) -> SymbolicFeatureSchema:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    predicates = payload.get("predicates")

    return SymbolicFeatureSchema(
        schema_name=payload["schema_name"],
        schema_version=payload["schema_version"],
        features=[
            SymbolicFeature(
                feature_name=x["feature_name"],
                itemset=tuple(x["itemset"]),
                source_label=x["source_label"],
                support=x.get("support"),
                confidence_attack=x.get("confidence_attack"),
                confidence_benign=x.get("confidence_benign"),
                mining_type=x.get("mining_type"),
                utility_score=x.get("utility_score", 1.0),
                clauses=(
                    tuple(tuple(c) for c in x["clauses"])
                    if x.get("clauses") is not None
                    else None
                ),
                p_value=x.get("p_value"),
            )
            for x in payload["features"]
        ],
        predicates=(
            [
                AttributePredicate(
                    token=p["token"],
                    attribute=p["attribute"],
                    operator=p["operator"],
                    value=p["value"],
                )
                for p in predicates
            ]
            if predicates is not None
            else None
        ),
    )
