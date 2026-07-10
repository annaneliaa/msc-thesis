from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from thesis.mining.attribute_contrast_mining import predicate_support_stats
from thesis.mining.attribute_features import BINARY_CATEGORICAL_FIELDS
from thesis.schemas.dynamic_schema import (
    DynamicCompoundRule,
    DynamicSchema,
    DynamicSinglePredicate,
)
from thesis.schemas.features import AttributePredicate


def _direction(confidence_attack: float, confidence_benign: float) -> str:
    return "attack" if confidence_attack > confidence_benign else "benign"


def _categorical_predicate_type(field: str) -> str:
    return "binary" if field in BINARY_CATEGORICAL_FIELDS else "categorical"


def _numeric_op_word(operator: str) -> str:
    return "le" if operator == "<=" else "gt"


def _assign_numeric_predicate_ids(
    numeric_preds: list[AttributePredicate],
) -> dict[str, str]:
    """
    token -> predicate_id for numeric-threshold predicates, keyed on
    (field, operator) with the mined threshold value dropped -- so the same
    conceptual predicate is trackable across Vk versions for Signal-1 PSI
    purposes. A decision tree can independently split on the same
    (field, operator) in more than one leaf path (each leaf path only dedups
    same-field/same-direction bounds *within itself*, not across leaves --
    see decision_tree_rule_mining._merge_predicate_into_path), so on
    collision within one Vk, sort by ascending value and append a stable
    ordinal suffix to all but the first. Only the un-suffixed
    (single-occurrence) case has full cross-Vk trackability.
    """
    groups: dict[tuple[str, str], list[AttributePredicate]] = {}
    for pred in numeric_preds:
        key = (pred.attribute, _numeric_op_word(pred.operator))
        groups.setdefault(key, []).append(pred)

    id_map: dict[str, str] = {}
    for (field, op_word), preds in groups.items():
        base_id = f"num:{field}_{op_word}"
        ordered = sorted(preds, key=lambda p: p.value)
        for i, pred in enumerate(ordered):
            id_map[pred.token] = base_id if i == 0 else f"{base_id}#{i + 1}"
    return id_map


def _rule_id(conditions: tuple[tuple[str, str, Any], ...]) -> str:
    canonical = sorted(conditions, key=lambda c: (c[0], c[1], str(c[2])))
    digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:12]
    return f"rule:{digest}"


def build_dynamic_schema(
    contrast_stats_df: pd.DataFrame,
    leaf_rules_df: pd.DataFrame,
    predicate_alphabet: list[AttributePredicate],
    column_predicate_map: dict[str, tuple[str, Any]],
    X_num: pd.DataFrame,
    y: pd.Series,
    version: int,
    mining_window_start: datetime,
    mining_window_end: datetime,
    mined_at: datetime | None = None,
) -> DynamicSchema:
    """
    Build a deployment-scoped DynamicSchema (Vk) from one mining run's raw
    stats -- the pre-concatenation frames attribute_mining_job.py produces
    internally, not its already-merged mined_df, since the split
    attack/benign counts needed here don't survive that concatenation.

    deployed_at/superseded_at are left None: those are registry-owned, set
    only by DynamicSchemaRegistry.deploy().
    """
    mined_at = mined_at or datetime.now(timezone.utc)
    base_attack_rate = float(y.mean()) if len(y) else 0.0

    single_predicates: list[DynamicSinglePredicate] = []

    # --- categorical/binary single predicates (contrast-set survivors) ---
    singles_df = contrast_stats_df[
        contrast_stats_df["itemset"].apply(lambda itemset: len(itemset) == 1)
    ]
    for _, row in singles_df.iterrows():
        column = row["itemset"][0]
        mapping = column_predicate_map.get(column)
        if mapping is None:
            continue
        field, value = mapping
        single_predicates.append(
            DynamicSinglePredicate(
                predicate_id=f"cat:{field}={value}",
                predicate_type=_categorical_predicate_type(field),
                field=field,
                operator="==",
                value=value,
                attack_support=float(row["confidence_attack"]),
                benign_support=float(row["confidence_benign"]),
                growth_rate=float(row["growth_rate"]),
                direction=_direction(
                    row["confidence_attack"], row["confidence_benign"]
                ),
                n_attack=int(row["n_attack"]),
                n_benign=int(row["n_benign"]),
                p_value=row["p_value"],
                schema_version=version,
                mined_at=mined_at,
            )
        )

    # --- numeric-threshold single predicates (from decision-tree splits) ---
    numeric_preds = [p for p in predicate_alphabet if p.operator in (">", "<=")]
    numeric_id_map = _assign_numeric_predicate_ids(numeric_preds)

    y_arr = y.to_numpy()
    attack_mask = y_arr == 1
    benign_mask = y_arr == 0
    n_attack = int(attack_mask.sum())
    n_benign = int(benign_mask.sum())

    for pred in numeric_preds:
        if pred.attribute not in X_num.columns:
            continue
        col = X_num[pred.attribute].to_numpy()
        fires = col > pred.value if pred.operator == ">" else col <= pred.value
        stats = predicate_support_stats(
            fires, attack_mask, benign_mask, n_attack, n_benign
        )
        single_predicates.append(
            DynamicSinglePredicate(
                predicate_id=numeric_id_map[pred.token],
                predicate_type="numeric_threshold",
                field=pred.attribute,
                operator=pred.operator,
                value=pred.value,
                attack_support=stats["attack_support"],
                benign_support=stats["benign_support"],
                growth_rate=stats["growth_rate"],
                direction=_direction(stats["attack_support"], stats["benign_support"]),
                n_attack=stats["n_attack"],
                n_benign=stats["n_benign"],
                p_value=stats["p_value"],
                schema_version=version,
                mined_at=mined_at,
            )
        )

    # --- compound rules (decision-tree leaves) ---
    alphabet_by_token = {p.token: p for p in predicate_alphabet}
    compound_rules: list[DynamicCompoundRule] = []
    for _, row in leaf_rules_df.iterrows():
        conditions = tuple(
            (
                alphabet_by_token[token].attribute,
                alphabet_by_token[token].operator,
                alphabet_by_token[token].value,
            )
            for token in row["itemset"]
            if token in alphabet_by_token
        )
        confidence_attack = float(row["confidence_attack"])
        confidence_benign = float(row["confidence_benign"])
        prediction = _direction(confidence_attack, confidence_benign)
        compound_rules.append(
            DynamicCompoundRule(
                rule_id=_rule_id(conditions),
                conditions=conditions,
                prediction=prediction,
                confidence=max(confidence_attack, confidence_benign),
                support_attack=confidence_attack,
                support_benign=confidence_benign,
                n_samples=int(row["support_count"]),
                schema_version=version,
                mined_at=mined_at,
            )
        )

    return DynamicSchema(
        version=version,
        mined_at=mined_at,
        mining_window_start=mining_window_start,
        mining_window_end=mining_window_end,
        base_attack_rate=base_attack_rate,
        single_predicates=single_predicates,
        compound_rules=compound_rules,
    )
