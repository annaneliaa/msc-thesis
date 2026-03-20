from dataclasses import dataclass

from thesis.config import Settings


@dataclass
class DummyModel:
    model_name: str
    model_version: str

    def predict(self, text: str) -> tuple[int, float]:
        score = 0.9 if "attack" in text.lower() else 0.1 # just a dummy set up for now
        label = int(score > 0.5)
        return label, score


def load_model(settings: Settings) -> DummyModel:
    return DummyModel(
        model_name=settings.model.model_name,
        model_version=settings.model.model_version,
    )