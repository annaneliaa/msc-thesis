from collections.abc import Iterable
import pandas as pd

from thesis.schemas.groups import AlertGroup
from thesis.schemas.features import (
    SEQUENCE_MINING_TYPES,
    AttributePredicate,
    SymbolicFeatureSchema,
    SymbolicFeature,
)
from thesis.mining.common.attribute_features import compute_candidate_attribute_features
from thesis.mining.temporal.prefixspan.repeat_encoding import (
    encode_sequence_of_itemsets,
)
from thesis.mining.common.token_abstraction import abstract_mail_hosts
from thesis.mining.common.util import sequence_contains_pattern
from thesis.monitor.predicate_eval import evaluate_condition


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
        if evaluate_condition(feats, pred.attribute, pred.operator, pred.value):
            tokens.add(pred.token)
    return tokens


def _alert_group_steps(tx: AlertGroup) -> list[set[str]]:
    """The group's itemsets in time order, with the same mail-host abstraction
    the miner applied to the patterns. Used for sequence features: the
    per-host look-back sequence when one was built (context_steps), else the
    group's own per-alert itemsets (sorted_items)."""
    steps = tx.context_steps if tx.context_steps is not None else tx.sorted_items
    return [abstract_mail_hosts(set(step)) for step in (steps or [])]


def _compile_feature(
    feature: "SymbolicFeature",
) -> tuple[str, list[frozenset[str]]] | tuple[str, tuple[str, ...]]:
    """Return (name, clauses) for set features, (name, pattern) for sequences.

    Set features: each clause is one AND-set; a plain AND feature produces a
    single-element clause list so the evaluator loop is uniform: fire if *any*
    clause is a subset.
    Sequence features (mining_type in SEQUENCE_MINING_TYPES): the ordered
    pattern; fire if it is a subsequence of the group's alert steps, with the
    same matching rule the miner counted support with.
    """
    if feature.mining_type in SEQUENCE_MINING_TYPES:
        return feature.feature_name, tuple(feature.itemset)
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

        self.compiled_features = []
        self.compiled_sequences = []
        for f in self.features:
            if f.mining_type in SEQUENCE_MINING_TYPES:
                self.compiled_sequences.append(_compile_feature(f))
            else:
                self.compiled_features.append(_compile_feature(f))
        self._column_order = [f.feature_name for f in self.features]
        self.predicates = feature_schema.predicates

    def _items_for(self, tx: AlertGroup) -> set[str]:
        return _alert_group_items(tx) | _attribute_predicate_tokens(tx, self.predicates)

    def _row(self, tx: AlertGroup) -> dict[str, int]:
        items = self._items_for(tx)
        row = {
            name: int(any(clause.issubset(items) for clause in clauses))
            for name, clauses in self.compiled_features
        }
        if self.compiled_sequences:
            steps = _alert_group_steps(tx)
            for name, pattern in self.compiled_sequences:
                row[name] = int(sequence_contains_pattern(steps, pattern))
        return {name: row[name] for name in self._column_order if name in row}

    def transform_one(self, tx: AlertGroup) -> pd.DataFrame:
        return pd.DataFrame([self._row(tx)])

    def transform(self, alert_groups: Iterable[AlertGroup]) -> pd.DataFrame:
        return pd.DataFrame([self._row(tx) for tx in alert_groups])
