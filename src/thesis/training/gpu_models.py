"""GPU-backed sklearn-compatible classifiers: a small torch feedforward NN and
a cuML RandomForest wrapper. Follow the same BaseEstimator wrapper pattern as
training/anomaly_models.py (BernoulliOneClass/BinaryAutoencoder) so both slot
into the existing fit/predict_proba/joblib-persistence/SHAP contract that
training/train.py and training/explain.py already dispatch on.

cuml is imported lazily inside CuMLRandomForestClassifierWrapper (not at
module top level) since RAPIDS is Linux+NVIDIA-only -- this module (and
therefore model_factory.py, which imports it) must stay importable on a
plain Mac dev machine with no cuml installed at all.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from thesis.grouping._device import resolve_device


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Simple torch feedforward NN, GPU-trained via resolve_device()
    (mps -> cuda -> cpu). Architecture mirrors the existing sklearn "mlp"
    factory (hidden_layer_sizes=(128, 64)) for a fair comparison; the only
    reason for a second NN entry at all is that sklearn's MLPClassifier has
    no GPU path.

    Persistence: the fitted module is always moved back to CPU at the end of
    fit() so joblib.dump (training/persistence.py) produces an artifact
    loadable on a machine without the same GPU. predict/predict_proba move
    it back onto resolve_device() lazily.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64),
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        validation_fraction: float = 0.1,
        early_stopping_patience: int = 10,
        device: str | None = None,
        random_state: int = 42,
        pos_weight: float | None = None,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        # Class-imbalance weighting for BCEWithLogitsLoss, set by the active
        # pool-sampling condition (model_factory.py passes
        # scale_pos_weight through as this constructor kwarg) -- read at
        # fit() time rather than passed to fit() directly, matching sklearn's
        # convention of keeping hyperparameters on __init__, not fit().
        self.pos_weight = pos_weight
        # Permutation importance shouldn't fan this out across processes --
        # each worker would need its own CUDA context / re-pickle a
        # GPU-resident module. See train.py's _n_jobs line.
        self._gpu_model = True

    def _build_module(self, n_features: int):
        import torch.nn as nn

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import pandas as pd
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        device = resolve_device(self.device)

        n = len(X)
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.validation_fraction)) if n > 10 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train_t = torch.from_numpy(X[train_idx]).to(device)
        y_train_t = torch.from_numpy(y[train_idx]).to(device)
        if n_val:
            X_val_t = torch.from_numpy(X[val_idx]).to(device)
            y_val_t = torch.from_numpy(y[val_idx]).to(device)

        module = self._build_module(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)
        pos_weight_t = (
            torch.tensor([self.pos_weight], dtype=torch.float32, device=device)
            if self.pos_weight is not None
            else None
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        best_val_loss = float("inf")
        best_state = None
        patience_left = self.early_stopping_patience

        n_train = len(X_train_t)
        for _epoch in range(self.epochs):
            module.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]
                optimizer.zero_grad()
                logits = module(xb).squeeze(-1)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            if not n_val:
                continue

            module.eval()
            with torch.no_grad():
                val_logits = module(X_val_t).squeeze(-1)
                val_loss = float(loss_fn(val_logits, y_val_t))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in module.state_dict().items()
                }
                patience_left = self.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            module.load_state_dict(best_state)

        module.eval()
        self.model_ = module.cpu()
        return self

    def _forward_proba(self, X) -> np.ndarray:
        import pandas as pd
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        device = resolve_device(self.device)
        module = self.model_.to(device)
        module.eval()
        with torch.no_grad():
            logits = module(torch.from_numpy(X).to(device)).squeeze(-1)
            p1 = torch.sigmoid(logits).cpu().numpy()
        self.model_ = module.cpu()
        return p1

    def predict_proba(self, X) -> np.ndarray:
        p1 = self._forward_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self._forward_proba(X) >= 0.5).astype(int)


class CuMLRandomForestClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wraps cuml.ensemble.RandomForestClassifier for GPU-accelerated
    RandomForest training. n_estimators/max_depth match the existing sklearn
    "rf" factory for comparability -- cuML's own default (max_depth=16, at
    least through cuml 26.06) is NOT unlimited like sklearn's, so leaving it
    unset would silently cap tree depth relative to "rf" and skew any
    rf-vs-rf_gpu comparison. Pinned explicitly here rather than left to
    cuML's default, which is also scheduled to change (to None) in a future
    release -- see cuml.ensemble.RandomForestClassifier's own FutureWarning.

    shap.TreeExplainer does not understand cuML's tree format, so this
    exposes get_shap_values() explicitly (using shap's model-agnostic
    Permutation explainer against predict_proba) -- explain.py's dispatch
    checks hasattr(model, "get_shap_values") first, before ever considering
    TreeExplainer, so this path is picked automatically.

    cuml's RandomForestClassifier has no class_weight equivalent -- see
    model_factory.get_model_factory's per-model handling for the
    class_weighted pool condition's fallback.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._gpu_model = True

    def _to_numpy(self, arr) -> np.ndarray:
        if hasattr(arr, "to_numpy"):
            return arr.to_numpy()
        if hasattr(arr, "get"):  # cupy array
            return arr.get()
        return np.asarray(arr)

    def fit(self, X, y):
        import pandas as pd
        from cuml.ensemble import RandomForestClassifier as CuMLRF

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        self._model = CuMLRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self.feature_importances_ = self._to_numpy(self._model.feature_importances_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._to_numpy(self._model.predict_proba(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_shap_values(self, bg_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
        import shap

        explainer = shap.Explainer(self.predict_proba, bg_arr)
        sv = explainer(x_arr)
        return sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
