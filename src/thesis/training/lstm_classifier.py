import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) — treat each feature as a timestep
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)  # (batch,) logits


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible LSTM classifier for tabular binary classification."""

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 1,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        n_neg = float((y_t == 0).sum())
        n_pos = float((y_t == 1).sum())
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)

        self.classes_ = np.array([0, 1])
        self.net_ = _LSTMNet(self.hidden_size, self.n_layers)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.net_.train()
        n = len(X_t)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                optimizer.step()

        self.net_.eval()
        return self

    def predict_proba(self, X):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            proba = torch.sigmoid(self.net_(X_t)).numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
