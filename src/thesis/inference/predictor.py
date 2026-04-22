import pandas as pd
import numpy as np

from thesis.schemas.models import ModelArtifact


class SklearnTabularModel:
    def __init__(self, artifact: ModelArtifact):
        self.model = artifact.model
        self.schema_name = artifact.schema_name
        self.features = artifact.features
        self.model_type = artifact.model_type
        self.model_version = artifact.model_version

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_used = X[self.features].fillna(0)
        return self.model.predict_proba(X_used)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
