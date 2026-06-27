"""Sklearn-compatible one-class anomaly detectors for binary feature vectors."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor


class BernoulliOneClass(BaseEstimator):
    """
    One-class anomaly detector for binary features using Bernoulli likelihood.

    Fits P(feature_i=1) from benign-only training data with Laplace smoothing,
    then scores each transaction by its log-likelihood under the benign model.
    A transaction with low log-likelihood (unusual binary pattern) is flagged anomalous.

    sklearn interface: fit / decision_function / predict (-1 = anomaly, +1 = normal).
    Provides analytic shap_values() — no approximation overhead.
    """

    def __init__(self, contamination: float = 0.05, alpha: float = 1.0):
        self.contamination = contamination
        self.alpha = alpha

    def fit(self, X, y=None):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=float)
        # Binarise: count features (x > 1) are treated as "present" (1).
        # The Bernoulli formula is only valid for x ∈ {0, 1}; counts > 1 make
        # (1 - x) negative, inverting the absence term and corrupting scores.
        X = np.clip(X, 0, 1)
        n = X.shape[0]
        p = (X.sum(axis=0) + self.alpha) / (n + 2 * self.alpha)
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)
        self.log_p_ = np.log(p)
        self.log_1p_ = np.log(1 - p)
        train_scores = self._score(X)
        self.threshold_ = np.percentile(train_scores, 100 * self.contamination)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, 0, 1)
        return X @ self.log_p_ + (1 - X) @ self.log_1p_

    def decision_function(self, X) -> np.ndarray:
        """Log P(x | benign model). Higher = more normal (sklearn convention)."""
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._score(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        """Returns -1 for anomaly, +1 for normal."""
        scores = self.decision_function(X)
        out = np.ones(len(scores), dtype=int)
        out[scores < self.threshold_] = -1
        return out

    def shap_values(self, X) -> np.ndarray:
        """
        Analytic attribution values for the anomaly score (= -log_likelihood).

        For each sample: contribution of feature i = -(log_lik_i - E[log_lik_i])
        where E[log_lik_i] is the expected contribution under the benign marginal.

        Positive value → feature pushes toward anomalous (deviates from benign pattern).
        Shape: (n_samples, n_features).
        """
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.clip(np.asarray(X, dtype=float), 0, 1)
        p = np.exp(self.log_p_)
        per_feature = X * self.log_p_ + (1 - X) * self.log_1p_
        baseline = p * self.log_p_ + (1 - p) * self.log_1p_
        return -(per_feature - baseline)

    @property
    def log_odds_(self) -> np.ndarray:
        """Log odds ratio per feature: log(p/(1-p)). More positive = more common in benign."""
        return self.log_p_ - self.log_1p_


class BinaryAutoencoder(BaseEstimator):
    """
    Reconstruction-error anomaly detector for binary features.

    Trains an MLP autoencoder (X → X) on benign-only data. Scores each transaction
    by its per-sample mean squared reconstruction error — high error means the
    binary pattern is unusual relative to what the autoencoder learned from benign traffic.

    sklearn interface: fit / decision_function / predict (-1 = anomaly, +1 = normal).
    Provides shap_values() as per-feature reconstruction error deviation from the
    benign training baseline (interpretable and fast, not a true Shapley computation).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32, 64),
        max_iter: int = 500,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=float)
        self._net = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self._net.fit(X, X)
        X_hat = self._net.predict(X)
        self.train_feature_errors_ = np.mean((X - X_hat) ** 2, axis=0)
        train_errors = np.mean((X - X_hat) ** 2, axis=1)
        self.threshold_ = np.percentile(train_errors, 100 * (1 - self.contamination))
        return self

    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._net.predict(X)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean((X - self._reconstruct(X)) ** 2, axis=1)

    def decision_function(self, X) -> np.ndarray:
        """Negative reconstruction error. Higher = more normal (sklearn convention)."""
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        return -self._reconstruction_error(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        """Returns -1 for anomaly, +1 for normal."""
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        errors = self._reconstruction_error(np.asarray(X, dtype=float))
        out = np.ones(len(errors), dtype=int)
        out[errors > self.threshold_] = -1
        return out

    def shap_values(self, X) -> np.ndarray:
        """
        Per-feature reconstruction error deviation from the benign training baseline.

        value_i = (x_i - x_hat_i)^2 - train_feature_errors_i

        Positive = feature reconstructed worse than typical benign → pushes toward anomalous.
        Shape: (n_samples, n_features).
        """
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=float)
        X_hat = self._reconstruct(X)
        return (X - X_hat) ** 2 - self.train_feature_errors_
