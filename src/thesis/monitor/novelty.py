"""
Novelty signal (observe-only, not wired into classify_action): how much of a
window's traffic falls outside what the deployed schema saw at mining time,
measured in the schema's own input space.

For each alert group, its *firing pattern* is the set of the DynamicSchema's
single predicates that hold for it. The reference is the set of patterns
seen on the groups the schema was mined on (W_src's train split). Per window:

* pattern_novelty_rate: share of groups whose firing pattern never occurred
  in the reference -- traffic the schema has no precedent for.
* no_fire_rate: share of groups on which no predicate fires at all -- traffic
  the schema has nothing to say about. Compared against the same rate on the
  reference (reference_no_fire_rate), so a shift is visible.

Deliberately not SignatureID novelty: the model never sees SignatureID (it's
a class shortcut on CSCAS), so signature novelty would measure change the
model doesn't feel, and on CSCAS new signatures are mostly attacks, which
would turn the signal into an attack-rate proxy. The DeepCASE counterpart is
its own-input novelty: the share of events with predict code -2 (event type
unknown or without training clusters).

The signal itself is label-free. The per-class rates (novelty among attack /
benign groups) are logged only for the frozen-pi check -- if novelty is
driven by attacks alone, it's an attack-rate proxy, not drift detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from thesis.mining.common.attribute_features import compute_candidate_attribute_features
from thesis.monitor.predicate_eval import evaluate_condition
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup


def firing_pattern(schema: DynamicSchema, feats: dict[str, Any]) -> frozenset[str]:
    """predicate_ids of the schema's single predicates that hold for one
    group's candidate attribute features."""
    return frozenset(
        p.predicate_id
        for p in schema.single_predicates
        if evaluate_condition(feats, p.field, p.operator, p.value)
    )


def firing_patterns(
    schema: DynamicSchema, groups: Iterable[AlertGroup]
) -> list[frozenset[str]]:
    return [
        firing_pattern(schema, compute_candidate_attribute_features(g)) for g in groups
    ]


@dataclass(slots=True)
class NoveltyReference:
    schema_version: int
    patterns: frozenset[frozenset[str]]
    no_fire_rate: float
    n_groups: int


def build_novelty_reference(
    schema: DynamicSchema, reference_groups: Sequence[AlertGroup]
) -> NoveltyReference:
    pats = firing_patterns(schema, reference_groups)
    n = len(pats)
    return NoveltyReference(
        schema_version=schema.version,
        patterns=frozenset(pats),
        no_fire_rate=sum(1 for p in pats if not p) / n if n else 0.0,
        n_groups=n,
    )


@dataclass(slots=True)
class NoveltySignal:
    n_groups: int
    pattern_novelty_rate: float
    no_fire_rate: float
    reference_no_fire_rate: float
    n_reference_patterns: int
    n_novel_patterns: int  # distinct unseen patterns in this window
    # Logging only (frozen-pi check); None when the window has no such label.
    pattern_novelty_rate_attack: float | None
    pattern_novelty_rate_benign: float | None


def compute_novelty(
    schema: DynamicSchema,
    reference: NoveltyReference,
    groups: Sequence[AlertGroup],
) -> NoveltySignal:
    if reference.schema_version != schema.version:
        raise ValueError(
            f"NoveltyReference was built for schema version "
            f"{reference.schema_version}, got {schema.version}"
        )
    return novelty_from_patterns(reference, firing_patterns(schema, groups), groups)


def novelty_from_patterns(
    reference: NoveltyReference,
    pats: Sequence[frozenset[str]],
    groups: Sequence[AlertGroup],
) -> NoveltySignal:
    """compute_novelty on already-computed firing patterns (aligned with
    `groups`), so a caller that evaluated the predicates anyway doesn't pay
    for compute_candidate_attribute_features twice."""
    n = len(pats)
    novel = [p not in reference.patterns for p in pats]

    def _rate_for(label: str) -> float | None:
        flags = [nv for nv, g in zip(novel, groups) if g.group_label == label]
        return sum(flags) / len(flags) if flags else None

    return NoveltySignal(
        n_groups=n,
        pattern_novelty_rate=sum(novel) / n if n else 0.0,
        no_fire_rate=sum(1 for p in pats if not p) / n if n else 0.0,
        reference_no_fire_rate=reference.no_fire_rate,
        n_reference_patterns=len(reference.patterns),
        n_novel_patterns=len({p for p, nv in zip(pats, novel) if nv}),
        pattern_novelty_rate_attack=_rate_for("attack"),
        pattern_novelty_rate_benign=_rate_for("benign"),
    )
