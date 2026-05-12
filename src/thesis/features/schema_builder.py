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


def _make_or_feature_name(
    clauses: tuple[tuple[str, ...], ...], prefix: str = "or"
) -> str:
    clause_parts = ["_".join(_sanitize_token(x) for x in c) for c in clauses]
    return f"{prefix}__{'__OR__'.join(clause_parts)}"


def build_symbolic_feature_schema(
    df: pd.DataFrame,
    source_label: str,
    schema_name: str,
    schema_version: str,
) -> SymbolicFeatureSchema:
    """
    Convert mined itemsets dataframe into a symbolic feature schema.
    Expects at least an 'itemset' column.
    """
    features: list[SymbolicFeature] = []
    seen_names: set[str] = set()

    for _, row in df.iterrows():
        is_or = (
            "clauses" in row.index
            and row["clauses"] is not None
            and not (isinstance(row["clauses"], float))  # NaN from concat
        )

        if is_or:
            clauses: tuple[tuple[str, ...], ...] = row["clauses"]
            feature_name = _make_or_feature_name(clauses)
            if feature_name in seen_names:
                continue
            seen_names.add(feature_name)
            features.append(
                SymbolicFeature(
                    feature_name=feature_name,
                    itemset=(),
                    source_label=source_label,
                    clauses=clauses,
                    confidence_attack=(
                        float(row["confidence_attack"])
                        if "confidence_attack" in row
                        and pd.notna(row["confidence_attack"])
                        else None
                    ),
                    confidence_benign=(
                        float(row["confidence_benign"])
                        if "confidence_benign" in row
                        and pd.notna(row["confidence_benign"])
                        else None
                    ),
                    mining_type="or_itemset",
                    utility_score=1.0,
                )
            )
        else:
            itemset = _parse_itemset(row["itemset"])
            feature_name = _make_feature_name(itemset)
            if feature_name in seen_names:
                continue
            seen_names.add(feature_name)
            features.append(
                SymbolicFeature(
                    feature_name=feature_name,
                    itemset=itemset,
                    source_label=source_label,
                    support=(
                        float(row["support"])
                        if "support" in row and pd.notna(row["support"])
                        else None
                    ),
                    confidence_attack=(
                        float(row["confidence_attack"])
                        if "confidence_attack" in row
                        and pd.notna(row["confidence_attack"])
                        else None
                    ),
                    confidence_benign=(
                        float(row["confidence_benign"])
                        if "confidence_benign" in row
                        and pd.notna(row["confidence_benign"])
                        else None
                    ),
                    mining_type=(
                        str(row["mining_type"])
                        if "mining_type" in row and pd.notna(row["mining_type"])
                        else None
                    ),
                    utility_score=1.0,
                )
            )

    n_input = len(df)
    n_kept = len(features)
    n_dropped = n_input - n_kept
    if n_dropped:
        print(
            f"  [schema_builder] Dropped {n_dropped} duplicate feature names ({n_kept}/{n_input} kept)"
        )

    return SymbolicFeatureSchema(
        schema_name=schema_name,
        schema_version=schema_version,
        features=features,
    )


def build_or_feature_schema(
    df: pd.DataFrame,
    source_label: str,
    schema_name: str,
    schema_version: str,
) -> SymbolicFeatureSchema:
    """
    Convert OR-pattern DataFrame (output of mine_or_disjunctions) into a schema.
    Expects a 'clauses' column of tuple[tuple[str, ...], ...].
    """
    features: list[SymbolicFeature] = []
    seen_names: set[str] = set()

    for _, row in df.iterrows():
        clauses: tuple[tuple[str, ...], ...] = row["clauses"]
        feature_name = _make_or_feature_name(clauses)

        if feature_name in seen_names:
            continue
        seen_names.add(feature_name)

        features.append(
            SymbolicFeature(
                feature_name=feature_name,
                itemset=(),
                source_label=source_label,
                clauses=clauses,
                confidence_attack=(
                    float(row["confidence_attack"])
                    if "confidence_attack" in row and pd.notna(row["confidence_attack"])
                    else None
                ),
                confidence_benign=(
                    float(row["confidence_benign"])
                    if "confidence_benign" in row and pd.notna(row["confidence_benign"])
                    else None
                ),
                mining_type="or_itemset",
                utility_score=1.0,
            )
        )

    return SymbolicFeatureSchema(
        schema_name=schema_name,
        schema_version=schema_version,
        features=features,
    )
