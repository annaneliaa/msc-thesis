from __future__ import annotations

from typing import Any, Sequence


def evaluate_condition(
    feats: dict[str, Any], field: str, operator: str, value: Any
) -> bool:
    """
    One (field, operator, value) condition against a precomputed
    compute_candidate_attribute_features() dict. Extracted from
    encoders/symbolic.py::_attribute_predicate_tokens's inline if/elif chain
    so the monitor and the encoder share one evaluation rule instead of
    drifting apart.

    For a SET_VALUED_CATEGORICAL_FIELDS field (AIT-ADS's short/sig/host --
    see attribute_features.py), `observed` is a set of the values present
    in this alert group rather than a single value, so "==" means
    membership ("value is one of the values this group has") and "!="
    means non-membership -- a plain `observed == value` would always be
    False for a set compared against a scalar. Every other (scalar-valued)
    field is unaffected; this branch only ever fires when `observed`
    actually is a set.
    """
    observed = feats.get(field)
    if observed is None:
        return False
    if isinstance(observed, (set, frozenset)):
        if operator == "==":
            return value in observed
        if operator == "!=":
            return value not in observed
        return False
    if operator == "==":
        return observed == value
    if operator == "!=":
        return observed != value
    if operator == ">":
        return observed > value
    if operator == "<=":
        return observed <= value
    return False


def evaluate_all_conditions(
    feats: dict[str, Any], conditions: Sequence[tuple[str, str, Any]]
) -> bool:
    """AND of evaluate_condition over a compound rule's conditions."""
    return all(
        evaluate_condition(feats, field, op, value) for field, op, value in conditions
    )
