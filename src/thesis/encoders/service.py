from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.encoders.baseline import BaselineFeatureEncoder
from thesis.encoders.symbolic import SymbolicFeatureEncoder


# def _resolve_symbolic_schema_path(
#     scenario_name: str,
#     schema_name: str,
# ) -> Path:
#     return (
#         Path("artifacts")
#         / "features"
#         / scenario_name
#         / f"{schema_name}.json"
#     )


def _resolve_symbolic_schema_path(
    scenario_name: str,
    schema_name: str | None = None,
) -> Path:
    base_dir = Path("artifacts") / "features" / scenario_name

    if not base_dir.exists():
        raise FileNotFoundError(f"No feature directory found at {base_dir}")

    # grab any symbolic schema
    candidates = sorted(base_dir.glob("symbolic*.json"))

    if not candidates:
        raise FileNotFoundError(
            f"No symbolic schema found in {base_dir} (expected symbolic*.json)"
        )

    # TEMP: pick the first one
    return candidates[0]


def encode_transactions(
    scenario_name: str,
    transactions: Iterable[Transaction],
    schema_name: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    if schema_name == "baseline":
        return BaselineFeatureEncoder().transform(transactions)

    if schema_name.startswith("symbolic"):
        symbolic_schema_path = _resolve_symbolic_schema_path(
            scenario_name=scenario_name,
            schema_name=schema_name,
        )

        if not symbolic_schema_path.exists():
            raise FileNotFoundError(
                f"Symbolic schema not found at {symbolic_schema_path}"
            )

        encoder = SymbolicFeatureEncoder.from_path(
            schema_path=symbolic_schema_path,
            top_k=top_k,
        )
        return encoder.transform(transactions)

    raise ValueError(f"Unsupported schema_name: {schema_name}")


def encode_transaction(
    scenario_name: str,
    transaction: Transaction,
    schema_name: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    if schema_name == "baseline":
        return BaselineFeatureEncoder().transform_one(transaction)

    if schema_name.startswith("symbolic"):
        symbolic_schema_path = _resolve_symbolic_schema_path(
            scenario_name=scenario_name,
            schema_name=schema_name,
        )

        if not symbolic_schema_path.exists():
            raise FileNotFoundError(
                f"Symbolic schema not found at {symbolic_schema_path}"
            )

        encoder = SymbolicFeatureEncoder.from_path(
            schema_path=symbolic_schema_path,
            top_k=top_k,
        )
        return encoder.transform_one(transaction)

    raise ValueError(f"Unsupported schema_name: {schema_name}")
