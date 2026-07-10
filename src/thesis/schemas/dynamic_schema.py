from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class DynamicSinglePredicate:
    """
    One surviving single predicate from a mining run, carried with the full
    contrast-set statistics that back it (as opposed to
    schemas.features.AttributePredicate, which is just the bare condition
    used to re-evaluate a rule at encode time).
    """

    predicate_id: str
    predicate_type: Literal["categorical", "binary", "numeric_threshold"]
    field: str
    operator: str  # "==" for categorical/binary; ">" or "<=" for numeric_threshold
    value: float | str | bool
    attack_support: float  # P(predicate=True | Label=1)
    benign_support: float  # P(predicate=True | Label=0)
    growth_rate: float
    direction: Literal["attack", "benign"]
    n_attack: int
    n_benign: int
    p_value: float | None
    schema_version: int
    mined_at: datetime


@dataclass(frozen=True, slots=True)
class DynamicCompoundRule:
    """One decision-tree leaf rule, with its full condition path and stats."""

    rule_id: str
    conditions: tuple[tuple[str, str, float | str | bool], ...]  # (field, op, value)
    prediction: Literal["attack", "benign"]
    confidence: float
    support_attack: float
    support_benign: float
    n_samples: int
    schema_version: int
    mined_at: datetime


@dataclass(frozen=True, slots=True)
class DynamicSchema:
    """
    A deployment-scoped schema version (Vk): the single-predicate registry +
    compound-rule registry a drift monitor compares live alert traffic
    against, plus deployment bookkeeping. Distinct from
    schemas.features.SymbolicFeatureSchema, which is experiment-scoped (one
    version per mining run, no deployment/supersession concept).

    deployed_at/superseded_at are registry-owned: build_dynamic_schema
    (features.dynamic_schema_builder) always produces them as None; only
    DynamicSchemaRegistry.deploy() sets them.
    """

    version: int
    mined_at: datetime
    mining_window_start: datetime
    mining_window_end: datetime
    base_attack_rate: float
    single_predicates: list[DynamicSinglePredicate] = field(default_factory=list)
    compound_rules: list[DynamicCompoundRule] = field(default_factory=list)
    deployed_at: datetime | None = None
    superseded_at: datetime | None = None
