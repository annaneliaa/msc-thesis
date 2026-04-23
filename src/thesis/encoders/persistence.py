import json
from pathlib import Path

from thesis.schemas.features import SymbolicFeature, SymbolicFeatureSchema


def load_symbolic_feature_schema(path: str | Path) -> SymbolicFeatureSchema:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return SymbolicFeatureSchema(
        schema_name=payload["schema_name"],
        schema_version=payload["schema_version"],
        features=[
            SymbolicFeature(
                feature_name=feature["feature_name"],
                itemset=tuple(feature["itemset"]),
                source_label=feature["source_label"],
                support=feature.get("support"),
                confidence_attack=feature.get("confidence_attack"),
                confidence_benign=feature.get("confidence_benign"),
            )
            for feature in payload["features"]
        ],
    )
