from dataclasses import dataclass

from thesis.config import Settings
from thesis.registry.encoders import get_encoder_path


@dataclass
class DummyEncoder:
    encoder_name: str
    encoder_version: str

    def predict(self, text: str) -> tuple[int, float]:
        score = 0.9 if "attack" in text.lower() else 0.1  # just a dummy set up for now
        label = int(score > 0.5)
        return label, score


def load_model(settings: Settings) -> DummyEncoder:
    path = get_encoder_path(
        settings.encoder.encoder_name, settings.encoder.encoder_version
    )

    print("Loading model from path: ", path)

    return DummyEncoder(
        encoder_name=settings.encoder.encoder_name,
        model_version=settings.encoder.encoder_version,
    )
