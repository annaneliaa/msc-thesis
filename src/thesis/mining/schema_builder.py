from __future__ import annotations

import ast
import re
import pandas as pd

from thesis.schemas.features import SymbolicFeature, SymbolicFeatureSchema


def _parse_itemset(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(x) for x in value)

    if isinstance(value, list):
        return tuple(str(x) for x in value)

    if isinstance(value, set):
        return tuple(str(x) for x in sorted(value))

    if isinstance(value, str):
        parsed = ast.literal_eval(value)

        if isinstance(parsed, tuple):
            return tuple(str(x) for x in parsed)
        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed)
        if isinstance(parsed, set):
            return tuple(str(x) for x in sorted(parsed))

    raise ValueError(f"Unsupported itemset value: {value!r}")


def _sanitize_token(token: str) -> str:
    token = token.replace(":", "_")
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", token)
    return token.strip("_").lower()


def _make_feature_name(itemset: tuple[str, ...], prefix: str = "sym") -> str:
    parts = [_sanitize_token(x) for x in itemset]
    return f"{prefix}__{'__'.join(parts)}"


def build_symbolic_feature_schema(
    df: pd.DataFrame,
    schema_name: str,
    schema_version: str,
    source_label: str,
    max_features: int | None = None,
) -> SymbolicFeatureSchema:
    """
    Convert mined itemsets dataframe into a symbolic feature schema.
    Expects at least an 'itemset' column.
    """
    if max_features is not None:
        df = df.head(max_features)

    features: list[SymbolicFeature] = []

    for _, row in df.iterrows():
        itemset = _parse_itemset(row["itemset"])

        features.append(
            SymbolicFeature(
                feature_name=_make_feature_name(itemset),
                itemset=itemset,
                source_label=source_label,
                support=float(row["support"])
                if "support" in row and pd.notna(row["support"])
                else None,
                confidence_attack=float(row["confidence_attack"])
                if "confidence_attack" in row and pd.notna(row["confidence_attack"])
                else None,
                confidence_benign=float(row["confidence_benign"])
                if "confidence_benign" in row and pd.notna(row["confidence_benign"])
                else None,
            )
        )

    return SymbolicFeatureSchema(
        schema_name=schema_name,
        schema_version=schema_version,
        features=features,
    )
