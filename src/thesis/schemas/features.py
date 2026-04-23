from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class BaseFeatureSchema:
    features: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DynamicFeatureSchema:
    features: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SymbolicFeature:
    feature_name: str
    itemset: tuple[str, ...]
    source_label: Literal["benign", "attack", "mixed"]
    support: float | None = None
    confidence_attack: float | None = None
    confidence_benign: float | None = None


@dataclass(frozen=True, slots=True)
class SymbolicFeatureSchema:
    features: list[SymbolicFeature] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    schema_name: str
    schema_version: str
    base: BaseFeatureSchema | None = None
    dynamic: DynamicFeatureSchema | None = None
    symbolic: SymbolicFeatureSchema | None = None

    def feature_names(self) -> list[str]:
        names: list[str] = []

        if self.base is not None:
            names.extend(self.base.features)

        if self.dynamic is not None:
            names.extend(self.dynamic.features)

        if self.symbolic is not None:
            names.extend(f.feature_name for f in self.symbolic.features)

        return names
