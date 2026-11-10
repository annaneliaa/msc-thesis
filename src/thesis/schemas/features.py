from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class BaseFeatureSchema:
    features: list[str] = field(default_factory=list)
    # Which base encoder encode_alert_groups_for_schema uses:
    # "baseline" -> encoders.baseline (the deployment-realistic reduced
    # column set); "cscas_full" -> encoders.cscas_full (the CSCAS paper's
    # full 41-column set incl. SCAS + offline *Similarity scores -- a
    # non-deployable reference ceiling, used by the temporal-decay
    # experiment). Defaults to "baseline" so every existing caller is
    # unchanged.
    kind: str = "baseline"


# mining_type values whose `itemset` is an ordered sequence of steps, not a
# set: the encoder fires them only when each step is a subset of a strictly
# later step of the group's sequence (context_steps, else sorted_items).
#   item_sequence:    one item per step, itemset = (item, item, ...)
#   itemset_sequence: several items per step; each element of `itemset` is
#                     one step, its items joined with STEP_ITEM_SEP
#                     (sorted), e.g. ("short:A & short:B", "short:C").
# Storing steps as joined strings keeps the existing schema JSON format.
SEQUENCE_MINING_TYPES = frozenset({"item_sequence", "itemset_sequence"})
STEP_ITEM_SEP = " & "


def join_step(items) -> str:
    """One itemset_sequence step as a schema token (sorted, joined)."""
    return STEP_ITEM_SEP.join(sorted(items))


def split_step(step: str) -> frozenset[str]:
    """Inverse of join_step."""
    return frozenset(step.split(STEP_ITEM_SEP))


@dataclass(frozen=True, slots=True)
class SymbolicFeature:
    feature_name: str
    itemset: tuple[str, ...]
    source_label: Literal["benign", "attack", "mixed"]
    support: float | None = None
    confidence_attack: float | None = None
    confidence_benign: float | None = None
    # "itemset", "item_sequence", "itemset_sequence", "contrast_categorical",
    # or "decision_tree_rule"
    mining_type: str | None = None
    utility_score: float = 1.0
    # OR-of-AND patterns: each inner tuple is one AND-clause.
    # None means this is a plain AND feature (backward-compatible).
    clauses: tuple[tuple[str, ...], ...] | None = None
    # Chi-square significance for contrast-set predicates (mining_type=
    # "contrast_categorical"); None when the significance filter wasn't used
    # or isn't applicable. growth_rate is not stored -- it's trivially
    # confidence_attack / confidence_benign.
    p_value: float | None = None


@dataclass(frozen=True, slots=True)
class AttributePredicate:
    """
    One atomic condition an alert-group attribute can be tested against, e.g.
    (attribute="category", operator="==", value="EXPLOIT") or
    (attribute="signature_matches_per_day", operator=">", value=847.0).

    This is the alphabet decision-tree-rule itemsets are built from. It lets
    the encoder reconstruct each condition against a *new* alert group's
    compute_candidate_attribute_features() output at encode/inference time,
    not just recognise the token string.
    """

    token: str
    attribute: str
    operator: str  # "==", "!=", ">", "<="
    value: float | str | bool


@dataclass(frozen=True, slots=True)
class SymbolicFeatureSchema:
    schema_name: str
    schema_version: str
    features: list[SymbolicFeature] = field(default_factory=list)
    # Alphabet of atomic conditions referenced by any decision_tree_rule
    # feature's itemset tokens. None for schemas built entirely from the old
    # co-occurrence path, which needs no such alphabet.
    predicates: list[AttributePredicate] | None = None


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    schema_name: str
    schema_version: str
    base: BaseFeatureSchema | None = None
    symbolic: SymbolicFeatureSchema | None = None

    def feature_names(self) -> list[str]:
        names: list[str] = []

        if self.base is not None:
            names.extend(self.base.features)

        if self.symbolic is not None:
            names.extend(f.feature_name for f in self.symbolic.features)

        return names
