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
    transactions_list = list(transactions)

    frames: list[pd.DataFrame] = []

    if schema.base is not None:
        baseline_frame = BaselineFeatureEncoder().transform(transactions_list)

        baseline_frame = baseline_frame.reindex(
            columns=schema.base.features,
            fill_value=0,
        )

        frames.append(baseline_frame)

    if schema.dynamic is not None:
        raise NotImplementedError("DynamicFeatureEncoder not implemented yet.")

    if schema.symbolic is not None:
        symbolic_encoder = SymbolicFeatureEncoder(
            feature_schema=schema.symbolic,
            top_k=top_k,
        )

        symbolic_frame = symbolic_encoder.transform(transactions_list)

        symbolic_feature_names = [
            feature.feature_name
            for feature in (
                schema.symbolic.features[:top_k]
                if top_k is not None
                else schema.symbolic.features
            )
        ]

        symbolic_frame = symbolic_frame.reindex(
            columns=symbolic_feature_names,
            fill_value=0,
        )

        frames.append(symbolic_frame)

    if not frames:
        raise ValueError(f"Schema '{schema.schema_name}' contains no feature groups.")

    encoded = pd.concat(
        [frame.reset_index(drop=True) for frame in frames],
        axis=1,
    )

    return encoded
