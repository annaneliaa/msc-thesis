from collections.abc import Iterable
import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.schemas.features import SymbolicFeatureSchema


def _transaction_items(tx: Transaction) -> set[str]:
    return set(tx.abs_items or tx.raw_items or [])


class SymbolicFeatureEncoder:
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

    def transform_one(self, tx: Transaction) -> pd.DataFrame:
        items = _transaction_items(tx)

        row = {
            feature_name: int(itemset.issubset(items))
            for feature_name, itemset in self.compiled_features
        }

        return pd.DataFrame([row])

    def transform(self, transactions: Iterable[Transaction]) -> pd.DataFrame:
        rows = []

        for tx in transactions:
            items = _transaction_items(tx)

            row = {
                feature_name: int(itemset.issubset(items))
                for feature_name, itemset in self.compiled_features
            }

            rows.append(row)

        return pd.DataFrame(rows)
