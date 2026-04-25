from collections.abc import Iterable
from pathlib import Path
import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.schemas.features import SymbolicFeatureSchema
from thesis.features.persistence import load_symbolic_feature_schema


def _transaction_items(tx: Transaction) -> set[str]:
    return set(tx.abs_items or tx.raw_items or [])


class SymbolicFeatureEncoder:
    """
    Stateless symbolic feature encoder driven by a mined feature schema.
    """

    def __init__(
        self,
        feature_schema: SymbolicFeatureSchema,
        top_k: int | None = None,
    ) -> None:
        self.feature_schema = feature_schema
        self.features = (
            feature_schema.features[:top_k]
            if top_k is not None
            else feature_schema.features
        )

        self.compiled_features = [
            (feature.feature_name, set(feature.itemset)) for feature in self.features
        ]

    @classmethod
    def from_path(
        cls,
        schema_path: str | Path,
        top_k: int | None = None,
    ) -> "SymbolicFeatureEncoder":
        feature_schema = load_symbolic_feature_schema(schema_path)
        return cls(feature_schema=feature_schema, top_k=top_k)

    def transform_one(self, tx: Transaction) -> pd.DataFrame:
        items = _transaction_items(tx)
        row: dict[str, int] = {}

        for feature_name, itemset in self.compiled_features:
            row[feature_name] = int(itemset.issubset(items))

        return pd.DataFrame([row])

    def transform(self, transactions: Iterable[Transaction]) -> pd.DataFrame:
        rows = []

        for tx in transactions:
            items = _transaction_items(tx)
            row: dict[str, int] = {}

            for feature_name, itemset in self.compiled_features:
                row[feature_name] = int(itemset.issubset(items))

            rows.append(row)

        return pd.DataFrame(rows)
