from __future__ import annotations

from dataclasses import dataclass

from thesis.schemas.dynamic_schema import DynamicSchema


@dataclass(slots=True)
class SchemaDiff:
    """Predicate/rule-set churn between two DynamicSchema versions -- the
    "how much did the schema actually change" number a remine's cost gets
    regressed against (system-operationality Experiment 3: Retraining /
    remining cost). added/removed are plain set differences on predicate_id
    / rule_id; "changed" is deliberately narrow -- a predicate/rule present
    in both versions whose *direction*/prediction flipped (attack<->benign)
    -- rather than a magnitude threshold on attack_support/confidence,
    which would need an arbitrary epsilon to justify. churn_frac follows
    the spec literally: predicate-level only, (added+removed+changed) /
    predicates_before -- rules_* are reported alongside for reference but
    don't feed the ratio."""

    predicates_added: int
    predicates_removed: int
    predicates_changed: int
    predicates_before: int
    predicates_after: int
    rules_added: int
    rules_removed: int
    rules_changed: int
    rules_before: int
    rules_after: int
    churn_frac: float


def diff_dynamic_schemas(old: DynamicSchema, new: DynamicSchema) -> SchemaDiff:
    """Diff `old` (Vk) against `new` (Vk+1) -- order matters, added/removed
    are relative to `old`."""
    old_preds = {p.predicate_id: p for p in old.single_predicates}
    new_preds = {p.predicate_id: p for p in new.single_predicates}
    added = set(new_preds) - set(old_preds)
    removed = set(old_preds) - set(new_preds)
    common = set(old_preds) & set(new_preds)
    changed = {
        pid for pid in common if old_preds[pid].direction != new_preds[pid].direction
    }

    old_rules = {r.rule_id: r for r in old.compound_rules}
    new_rules = {r.rule_id: r for r in new.compound_rules}
    r_added = set(new_rules) - set(old_rules)
    r_removed = set(old_rules) - set(new_rules)
    r_common = set(old_rules) & set(new_rules)
    r_changed = {
        rid
        for rid in r_common
        if old_rules[rid].prediction != new_rules[rid].prediction
    }

    n_pred_before = len(old_preds)
    churn_frac = (
        (len(added) + len(removed) + len(changed)) / n_pred_before
        if n_pred_before
        else float("nan")
    )

    return SchemaDiff(
        predicates_added=len(added),
        predicates_removed=len(removed),
        predicates_changed=len(changed),
        predicates_before=n_pred_before,
        predicates_after=len(new_preds),
        rules_added=len(r_added),
        rules_removed=len(r_removed),
        rules_changed=len(r_changed),
        rules_before=len(old_rules),
        rules_after=len(new_rules),
        churn_frac=churn_frac,
    )


def zero_diff(schema: DynamicSchema) -> SchemaDiff:
    """The no-op diff for a RETRAIN_ONLY event -- schema/Vk is untouched,
    so churn is 0 by construction, not merely "not computed"."""
    n_pred = len(schema.single_predicates)
    n_rules = len(schema.compound_rules)
    return SchemaDiff(
        predicates_added=0,
        predicates_removed=0,
        predicates_changed=0,
        predicates_before=n_pred,
        predicates_after=n_pred,
        rules_added=0,
        rules_removed=0,
        rules_changed=0,
        rules_before=n_rules,
        rules_after=n_rules,
        churn_frac=0.0,
    )
