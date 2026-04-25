from collections.abc import Iterable

import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.schemas.features import FeatureSchema
from thesis.encoders.baseline import BaselineFeatureEncoder
from thesis.encoders.symbolic import SymbolicFeatureEncoder


def encode_transactions_for_schema(
    transactions: Iterable[Transaction],
    schema: FeatureSchema,
    top_k: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if schema.base is not None:
        frames.append(BaselineFeatureEncoder().transform(transactions))

    if schema.symbolic is not None:
        encoder = SymbolicFeatureEncoder(
            schema=schema.symbolic,
            top_k=top_k,
        )
        frames.append(encoder.transform(transactions))

    if schema.dynamic is not None:
        raise NotImplementedError("DynamicFeatureEncoder not implemented yet.")

    if not frames:
        raise ValueError(f"Schema '{schema.schema_name}' contains no feature groups.")

    return pd.concat(
        [frame.reset_index(drop=True) for frame in frames],
        axis=1,
    )
