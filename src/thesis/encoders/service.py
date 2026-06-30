from collections.abc import Iterable

import pandas as pd

from thesis.schemas.preprocessing import AlertGroup
from thesis.schemas.features import FeatureSchema
from thesis.encoders.baseline import BaselineFeatureEncoder
from thesis.encoders.symbolic import SymbolicFeatureEncoder


def encode_alert_groups_for_schema(
    alert_groups: Iterable[AlertGroup],
    schema: FeatureSchema,
    top_k: int | None = None,
) -> pd.DataFrame:
    alert_groups_list = list(alert_groups)

    frames: list[pd.DataFrame] = []

    if schema.base is not None:
        baseline_frame = BaselineFeatureEncoder().transform(alert_groups_list)

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

        symbolic_frame = symbolic_encoder.transform(alert_groups_list)

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

    # Remove any duplicate columns that may have slipped through
    n_before = encoded.shape[1]
    encoded = encoded.loc[:, ~encoded.columns.duplicated(keep="first")]
    n_dropped = n_before - encoded.shape[1]
    if n_dropped:
        print(
            f"  [encoder] Dropped {n_dropped} duplicate columns ({encoded.shape[1]}/{n_before} kept)"
        )

    return encoded
