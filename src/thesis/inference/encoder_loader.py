from dataclasses import dataclass
import pandas as pd

from thesis.registry.encoders import get_encoder_path


@dataclass
class BaselineEncoder:
    encoder_name: str
    encoder_version: str

    def transform_row(self, row: dict) -> pd.DataFrame:
        """
        Convert one incoming alert_group row into a 1-row DataFrame.
        For now this is identity-style encoding for baseline features.
        """
        return pd.DataFrame([row])


def load_encoder(encoder_name: str, encoder_version: str) -> BaselineEncoder:
    path = get_encoder_path(
        encoder_name,
        encoder_version,
    )

    print("Loading encoder from path:", path)

    return BaselineEncoder(
        encoder_name=encoder_name,
        encoder_version=encoder_version,
    )
