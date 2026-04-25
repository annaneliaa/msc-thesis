import json
from dataclasses import asdict
from pathlib import Path

from thesis.schemas.features import SymbolicFeatureSchema, SymbolicFeature


def save_symbolic_feature_schema(schema: SymbolicFeatureSchema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)


def load_symbolic_feature_schema(path: Path) -> SymbolicFeatureSchema:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return SymbolicFeatureSchema(
        features=[
            SymbolicFeature(
                feature_name=x["feature_name"],
                itemset=tuple(x["itemset"]),
                source_label=x["source_label"],
                support=x.get("support"),
                confidence_attack=x.get("confidence_attack"),
                confidence_benign=x.get("confidence_benign"),
            )
            for x in payload["features"]
        ],
    )
