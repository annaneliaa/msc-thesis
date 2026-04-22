from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    features: List[str]
