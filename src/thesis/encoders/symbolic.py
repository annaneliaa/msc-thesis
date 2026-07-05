from collections.abc import Iterable
import pandas as pd

from thesis.schemas.groups import AlertGroup
from thesis.schemas.features import (
    AttributePredicate,
    SymbolicFeatureSchema,
    SymbolicFeature,
)
from thesis.mining.attribute_features import compute_candidate_attribute_features
from thesis.mining.repeat_encoding import encode_sequence_of_itemsets
from thesis.mining.token_abstraction import abstract_mail_hosts


def _alert_group_items(tx: AlertGroup) -> set[str]:
    base = set(tx.raw_items or [])
    if tx.sorted_items:
        encoded = encode_sequence_of_itemsets(tx.sorted_items)
        base.update(item for itemset in encoded for item in itemset)
    return abstract_mail_hosts(base)


def _attribute_predicate_tokens(
    tx: AlertGroup, predicates: list[AttributePredicate] | None
) -> set[str]:
    """
    Evaluate each decision-tree-rule predicate's (attribute, operator, value)
    condition against this alert group's own attribute values, so the same
    set-containment evaluation loop below works for decision-tree rules
    exactly as it does for co-occurrence itemsets.
    """
    if not predicates:
        return set()

    feats = compute_candidate_attribute_features(tx)
    tokens: set[str] = set()
    for pred in predicates:
        value = feats.get(pred.attribute)
        if value is None:
            continue
        if pred.operator == "==":
            fires = value == pred.value
        elif pred.operator == "!=":
            fires = value != pred.value
        elif pred.operator == ">":
            fires = value > pred.value
        elif pred.operator == "<=":
            fires = value <= pred.value
        else:
            fires = False
        if fires:
            tokens.add(pred.token)
    return tokens


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
        self.predicates = feature_schema.predicates

    def _items_for(self, tx: AlertGroup) -> set[str]:
        return _alert_group_items(tx) | _attribute_predicate_tokens(tx, self.predicates)

    def transform_one(self, tx: AlertGroup) -> pd.DataFrame:
        items = self._items_for(tx)

        row = {
            name: int(any(clause.issubset(items) for clause in clauses))
            for name, clauses in self.compiled_features
        }

        return pd.DataFrame([row])

    def transform(self, alert_groups: Iterable[AlertGroup]) -> pd.DataFrame:
        rows = []

        for tx in alert_groups:
            items = self._items_for(tx)

            row = {
                name: int(any(clause.issubset(items) for clause in clauses))
                for name, clauses in self.compiled_features
            }

            rows.append(row)

        return pd.DataFrame(rows)
