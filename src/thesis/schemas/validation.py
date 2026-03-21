from __future__ import annotations

import pandas as pd

from thesis.schemas.definitions import SCHEMAS


class SchemaValidationError(ValueError):
    """Raised when a dataframe does not match a registered schema."""


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> None:
    if schema_name not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema_name}")

    expected = SCHEMAS[schema_name]

    missing_cols = [col for col in expected if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Schema '{schema_name}' missing required columns: {missing_cols}"
        )

    dtype_errors: list[str] = []
    for col, expected_dtype in expected.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            dtype_errors.append(
                f"Column '{col}' expected dtype '{expected_dtype}' but got '{actual_dtype}'"
            )

    if dtype_errors:
        raise SchemaValidationError(
            f"Schema '{schema_name}' dtype mismatches: " + "; ".join(dtype_errors)
        )
