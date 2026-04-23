from collections.abc import Iterable
import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.encoder.baseline import BaselineFeatureEncoder


def encode_transactions(
    transactions: Iterable[Transaction],
    schema_name: str,
) -> pd.DataFrame:
    """
    Encode a collection of transactions under the requested feature schema.

    Currently supported:
    - baseline

    Returns a row-based DataFrame ready for training or inference.
    """
    if schema_name == "baseline":
        encoder = BaselineFeatureEncoder()
        return encoder.transform(transactions)

    raise ValueError(f"Unsupported schema_name: {schema_name}")


def encode_transaction(
    transaction: Transaction,
    schema_name: str,
) -> pd.DataFrame:
    """
    Encode a single transaction into a 1-row DataFrame.
    """
    if schema_name == "baseline":
        encoder = BaselineFeatureEncoder()
        return encoder.transform_one(transaction)

    raise ValueError(f"Unsupported schema_name: {schema_name}")
