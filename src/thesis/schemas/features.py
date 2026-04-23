from dataclasses import dataclass
from typing import List, Literal


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    features: List[str]


@dataclass(slots=True)
class SymbolicFeature:
    feature_name: str
    itemset: tuple[str, ...]
    source_label: Literal["benign", "attack", "mixed"]
    support: float | None = None
    confidence_attack: float | None = None
    confidence_benign: float | None = None


@dataclass(slots=True)
class SymbolicFeatureSchema:
    schema_name: str
    schema_version: str
    features: list[SymbolicFeature]
