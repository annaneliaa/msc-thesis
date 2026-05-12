from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from thesis.schemas.preprocessing import GroupingRecord, TokenizedAlert

from alertbert.model_eval_utils import (
    load_data_tools,
    load_models,
    load_reports,
)
from alertbert.models import AlertBERT, MaskedLangModelInferenceWrapper

ALERTBERT_METHOD = "alertbert"

# Path to the vendored AlertBERT repo (external/AlertBERT relative to thesis root)
_ALERTBERT_ROOT = Path(__file__).resolve().parents[6] / "external" / "AlertBERT"


class _TokenizedAlertDataset:
    """AlertDataset-compatible view over a list of TokenizedAlerts.

    Mirrors the interface of alertbert.aitads.AlertDataset: a .data dict of
    numpy arrays, .keys, __len__, and __getitem__ with cyclic-index slice
    support (needed by AlertBERT's sliding readout window).
    """

    def __init__(self, alerts: list[TokenizedAlert]) -> None:
        self.data: dict[str, np.ndarray] = {
            "raw_time": np.array([a.ts for a in alerts], dtype=np.float64),
            "short": np.array([a.short or "" for a in alerts]),
            "host": np.array([a.host or "" for a in alerts]),
        }
        self.keys = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data["raw_time"])

    def __getitem__(self, idx: int | slice | np.ndarray) -> dict:
        if isinstance(idx, int):
            try:
                return {k: self.data[k][idx] for k in self.keys}
            except IndexError:
                return {k: self.data[k][idx % len(self)] for k in self.keys}
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            idx_array = np.arange(start=start, stop=stop, step=step)
        elif isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        if len(idx_array) == 0:
            return {k: self.data[k][np.array([], dtype=int)] for k in self.keys}
        if idx_array[0] < 0 or idx_array[-1] >= len(self):
            idx_array = idx_array % len(self)
        return {k: self.data[k][idx_array] for k in self.keys}

    def __iter__(self):
        for i in range(len(self)):
            yield {k: self.data[k][i] for k in self.keys}


class AlertBERTGrouper:
    """Offline-inference grouper backed by a trained AlertBERT MLM.

    Lifecycle
    ---------
    1. Instantiate with model_id + hyperparams (does NOT load weights yet).
    2. Call load() once at startup, or let group() lazy-load on first call.
    3. Call group(alerts) per batch; returns one GroupingRecord per alert.

    The alerts passed to group() are sorted internally by timestamp before
    embedding, so insertion order does not matter.

    Small-batch note
    ----------------
    AlertBERT's readout window defaults to 2048 alerts. When a batch is
    smaller than the window the implementation falls back to cyclic-index
    padding (the same alerts repeated). This is fine for smoke-tests; for
    thesis experiments pass full-scenario batches (thousands of alerts).
    """

    def __init__(
        self,
        model_id: str,
        models_path: str | Path,
        delta: float = 2.0,
        theta: float = 2.0,
        dim_reduction: int = 2,
        device: str = "cpu",
    ) -> None:
        if theta < delta:
            raise ValueError(
                f"theta ({theta}) must be >= delta ({delta}): theta scales the "
                "max cosine distance, so values below delta are unreachable."
            )
        self.model_id = model_id
        self.models_path = str(models_path)
        self.delta = delta
        self.theta = theta
        self.dim_reduction = dim_reduction
        self.device = device
        self._alertbert = None

    def load(self) -> None:
        """Load model weights, vocabs, and collate function from disk."""
        ab_root = str(_ALERTBERT_ROOT)
        if ab_root not in sys.path:
            sys.path.insert(0, ab_root)

        reports, model_param_dicts = load_reports([self.model_id], self.models_path)
        # label_vocabs={} → only the model's own feature vocabs end up in the
        # inference collate_fn; we don't need ground-truth label decoding here.
        data_tools = load_data_tools(
            [self.model_id], model_param_dicts, self.models_path, label_vocabs={}
        )
        models = load_models(
            model_param_dicts, self.models_path, data_tools, self.device
        )
        inference_model = MaskedLangModelInferenceWrapper(
            models[self.model_id],
            layers=("embedding", "encoder"),
        )
        self._alertbert = AlertBERT(
            model=inference_model,
            collate_fn=data_tools[self.model_id]["inf_coll_fn"],
            dim_reduction=self.dim_reduction,
            delta=self.delta,
            theta=self.theta,
        )

    def group(self, alerts: list[TokenizedAlert]) -> list[GroupingRecord]:
        """Embed and cluster alerts. Returns one GroupingRecord per alert."""
        if not alerts:
            return []
        if self._alertbert is None:
            self.load()

        sorted_alerts = sorted(alerts, key=lambda a: a.ts)
        dataset = _TokenizedAlertDataset(sorted_alerts)
        cluster_labels: np.ndarray = self._alertbert.forward(dataset)

        return [
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=f"alertbert:{self.model_id}:{label}",
                method=ALERTBERT_METHOD,
            )
            for alert, label in zip(sorted_alerts, cluster_labels)
        ]
