from collections.abc import Iterable
import pandas as pd

from thesis.schemas.preprocessing import Transaction
from thesis.schemas.features import SymbolicFeatureSchema, SymbolicFeature
from thesis.mining.repeat_encoding import encode_sequence_of_itemsets


def _transaction_items(tx: Transaction) -> set[str]:
    base = set(tx.abs_items or tx.raw_items or [])
    if tx.sorted_items:
        encoded = encode_sequence_of_itemsets(tx.sorted_items)
        base.update(item for itemset in encoded for item in itemset)
    return base


def _compile_feature(
    feature: "SymbolicFeature",
) -> tuple[str, list[frozenset[str]]]:
    """Return (name, clauses) where each clause is one AND-set.

    A plain AND feature produces a single-element clause list so the
    evaluator loop is uniform: fire if *any* clause is a subset.
    """
    if feature.clauses is not None:
        return feature.feature_name, [frozenset(c) for c in feature.clauses]
    return feature.feature_name, [frozenset(feature.itemset)]


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

        self.compiled_features = [_compile_feature(f) for f in self.features]

    def transform_one(self, tx: Transaction) -> pd.DataFrame:
        items = _transaction_items(tx)

        row = {
            name: int(any(clause.issubset(items) for clause in clauses))
            for name, clauses in self.compiled_features
        }

        return pd.DataFrame([row])

    def transform(self, transactions: Iterable[Transaction]) -> pd.DataFrame:
        rows = []

        for tx in transactions:
            items = _transaction_items(tx)

            row = {
                name: int(any(clause.issubset(items) for clause in clauses))
                for name, clauses in self.compiled_features
            }

            rows.append(row)

        return pd.DataFrame(rows)
