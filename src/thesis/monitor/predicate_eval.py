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
    """
    observed = feats.get(field)
    if observed is None:
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
