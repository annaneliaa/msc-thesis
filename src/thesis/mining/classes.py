from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SymbolicMemory:
    """
    Tracks which symbolic features are useful over time.
    Scores decay over time; selected features gain score.
    """

    decay: float = 0.8
    reward: float = 1.0
    min_score: float = 0.8
    scores: Dict[str, float] = field(default_factory=dict)

    # a feature fades out if it stops being useful
    def step_decay(self):
        for k in list(self.scores.keys()):
            self.scores[k] *= self.decay
            if self.scores[k] < 1e-6:
                del self.scores[k]

    # if a feature is selected repeatedly the score grows ands stays above min_score
    def reward_feats(self, feats: List[str]):
        for f in feats:
            self.scores[f] = self.scores.get(f, 0.0) + self.reward

    # A feature is considered memory-active if score_f ≥ min_score
    def active(self) -> List[str]:
        return [f for f, s in self.scores.items() if s >= self.min_score]

    def active_with_threshold(self, tau: float) -> List[str]:
        return [f for f, s in self.scores.items() if s >= tau]


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    features: List[str]
