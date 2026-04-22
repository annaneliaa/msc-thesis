from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SklearnTabularModel:
    model_name: str
    model_version: str
    schema_name: str
    features: list[str]
    model: object

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_used = X[self.features].fillna(0)
        return self.model.predict_proba(X_used)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
