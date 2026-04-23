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

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_used = X.copy()

        for feature in self.features:
            if feature not in X_used.columns:
                X_used[feature] = 0

        return X_used[self.features].fillna(0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_used = self.prepare_features(X)
        return self.model.predict_proba(X_used)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
