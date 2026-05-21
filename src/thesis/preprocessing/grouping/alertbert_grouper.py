from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from thesis.schemas.preprocessing import GroupingRecord, TokenizedAlert

ALERTBERT_METHOD = "alertbert"


class _InferenceDataset:
    """AlertDataset-compatible wrapper around a list[TokenizedAlert].

    Stores string arrays for short/host so BaseSequenceCollate can apply vocab
    lookups, and a float array for raw_time. Supports cyclic integer/slice indexing
    to match the AlertBERT chunking contract.
    """

    def __init__(self, alerts: list[TokenizedAlert]) -> None:
        sorted_alerts = sorted(alerts, key=lambda a: a.ts)
        self.keys = ["short", "host", "raw_time"]
        self.data = {
            "short": np.array(
                [a.short or "<UNK>" for a in sorted_alerts], dtype=object
            ),
            "host": np.array([a.host or "<UNK>" for a in sorted_alerts], dtype=object),
            "raw_time": np.array(
                [float(a.ts) for a in sorted_alerts], dtype=np.float32
            ),
        }
        self._alert_ids: list[str] = [a.alert_id for a in sorted_alerts]

    def __len__(self) -> int:
        return len(self.data["raw_time"])

    def __getitem__(self, idx: int | slice | np.ndarray) -> dict:
        if isinstance(idx, int):
            try:
                return {k: self.data[k][idx] for k in self.keys}
            except IndexError:
                return {k: self.data[k][idx % len(self)] for k in self.keys}

        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            idx_array = np.arange(start, stop, step)
        elif isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            raise TypeError(f"Expected int, slice, or ndarray, got {type(idx)}")

        if len(idx_array) > 0 and (idx_array[0] < 0 or idx_array[-1] >= len(self)):
            idx_array = idx_array % len(self)
        return {k: self.data[k][idx_array] for k in self.keys}

    def __iter__(self):
        for i in range(len(self)):
            yield {k: self.data[k][i] for k in self.keys}


class AlertBERTGrouper:
    """Groups a list[TokenizedAlert] using a pre-trained AlertBERT checkpoint.

    The model is loaded lazily on first non-empty call to .group(). This means
    instantiation is cheap and theta/delta validation errors surface immediately,
    while missing-model errors surface only when grouping is actually attempted.

    Parameters
    ----------
    checkpoint_dir:
        Path to a saved_models/{model_id}/ directory containing
        report.json, model.pt, vocab_short.json, vocab_host.json.
    delta:
        Time-gap threshold (seconds) for pre-clustering consecutive alerts
        into coarse time windows, and the maximum combined distance for
        merging two alerts into the same group.
    theta:
        Cosine-distance scale factor. Must be >= delta.
        Controls how much semantic similarity matters vs. time proximity.
        When theta == delta, embeddings are ignored and pre-clusters become
        final groups. Increase theta relative to delta to make the model
        split temporally-close but semantically-different alerts.
    dim_reduction:
        Number of PCA dimensions to reduce embeddings to before clustering.
        Set to 0 to disable.
    padding:
        Number of context alerts added on each side of a readout chunk so
        that boundary alerts get richer contextual embeddings. These are
        discarded after embedding.
    readout:
        Number of alerts per embedding chunk. Together with padding this
        must not exceed the model's context window (default 4096).
    device:
        PyTorch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        delta: float = 2.0,
        theta: float = 6.0,
        dim_reduction: int = 2,
        padding: int = 1024,
        readout: int = 2048,
        device: str = "cpu",
    ) -> None:
        if theta < delta:
            raise ValueError(
                f"theta ({theta}) must be >= delta ({delta}): theta controls the "
                "cosine-distance scale and must be at least as large as the time-gap delta"
            )
        self._checkpoint_dir = Path(checkpoint_dir)
        self._delta = delta
        self._theta = theta
        self._dim_reduction = dim_reduction
        self._padding = padding
        self._readout = readout
        self._device_str = device
        self._grouper = None  # loaded lazily by _load_model()

    def _load_model(self) -> None:
        if self._grouper is not None:
            return

        # graph_tool (transitive via alertbert.models) references numpy.float128
        # which doesn't exist on macOS — patch before importing.
        if not hasattr(np, "float128"):
            np.float128 = np.longdouble  # type: ignore[attr-defined]

        from alertbert.models import (  # noqa: PLC0415
            AlertBERT,
            MaskedLanguageModel,
            MaskedLangModelInferenceWrapper,
        )
        from alertbert.preprocessing import (  # noqa: PLC0415
            BaseSequenceCollate,
            default_collate_fn,
            load_feature_vocabs,
        )

        # Defined here so AlertBERT is in scope — graph_tool must be available.
        class _AlertBERTFixed(AlertBERT):
            """AlertBERT subclass with a corrected get_embeddings.

            The upstream version has an UnboundLocalError when the dataset is smaller
            than readout (the loop variable `i` is referenced after a loop that never
            ran). This override handles that case and fixes the remainder-index arithmetic.
            """

            def get_embeddings(
                self,
                data: _InferenceDataset,
                apply_dim_red: bool = True,
                add_time: bool = True,
            ) -> np.ndarray:
                n = len(data)
                embeddings_list: list[np.ndarray] = []

                if n <= self.readout:
                    batch = self.collate_fn([data[0:n]])
                    batch = self.model(batch)
                    embeddings_list.append(batch["output"][0].cpu().numpy())
                else:
                    num_full_chunks = n // self.readout
                    remainder = n % self.readout

                    for i in range(num_full_chunks):
                        start = i * self.readout - self.padding
                        stop = (i + 1) * self.readout + self.padding
                        batch = self.collate_fn([data[start:stop]])
                        batch = self.model(batch)
                        embeddings_list.append(
                            batch["output"][0, self.padding : -self.padding]
                            .cpu()
                            .numpy()
                        )
                        if num_full_chunks >= 500 and (i + 1) % 200 == 0:
                            pct = 100 * (i + 1) / num_full_chunks
                            print(
                                f"        [{pct:.0f}%] {i + 1}/{num_full_chunks} embedding chunks",
                                flush=True,
                            )

                    if remainder:
                        start = num_full_chunks * self.readout - self.padding
                        stop = n + self.padding
                        batch = self.collate_fn([data[start:stop]])
                        batch = self.model(batch)
                        embeddings_list.append(
                            batch["output"][0, self.padding : -self.padding]
                            .cpu()
                            .numpy()
                        )

                embeddings = np.concatenate(embeddings_list)

                if apply_dim_red and self.dim_reduction is not None:
                    n_components = self.dim_reduction.n_components
                    if n > n_components:
                        embeddings = self.dim_reduction.fit_transform(embeddings)

                if add_time:
                    embeddings = np.concatenate(
                        (embeddings, data.data["raw_time"].reshape(-1, 1)),
                        axis=1,
                    )

                return embeddings

        device = torch.device(self._device_str)

        with (self._checkpoint_dir / "report.json").open() as f:
            report = json.load(f)
        params = report["params"]

        vocab_features = list(set(params["features"]) | set(params["targets"]))
        vocabs = load_feature_vocabs(
            str(self._checkpoint_dir), vocab_features, params["min_freq"]
        )

        collate_fn_map = {**vocabs, params["encoding"]: default_collate_fn}
        collate_fn = BaseSequenceCollate(collate_fn_map)

        model = MaskedLanguageModel(params=params, vocabs=vocabs)
        model.load_state_dict(
            torch.load(
                self._checkpoint_dir / "model.pt",
                weights_only=True,
                map_location=device,
            )
        )
        model.to(device)
        model.eval()

        inference_wrapper = MaskedLangModelInferenceWrapper(model)

        self._grouper = _AlertBERTFixed(
            model=inference_wrapper,
            collate_fn=collate_fn,
            dim_reduction=self._dim_reduction,
            delta=self._delta,
            theta=self._theta,
            padding=self._padding,
            readout=self._readout,
        )

    # Default time-window size for batched grouping (seconds).
    # Groups are at most a few seconds wide (delta << window), so splitting
    # here is safe and avoids O(n²) memory for large scenarios.
    _WINDOW_SECONDS: int = 6 * 3600  # 6 hours

    # Maximum alerts per sub-chunk. Windows larger than this are split by count
    # into equal sub-chunks before passing to _group_chunk. Since groups are
    # bounded by delta (≈2 s) and each sub-chunk spans many minutes, no group
    # is ever split across a sub-chunk boundary.
    _MAX_CHUNK_ALERTS: int = 30_000

    def group(self, alerts: list[TokenizedAlert]) -> list[GroupingRecord]:
        """Embed and cluster alerts, returning one GroupingRecord per alert.

        Alerts are processed in time windows of _WINDOW_SECONDS to bound the
        O(n²) distance-matrix memory used by the upstream clustering step.
        Since groups span at most a few seconds (delta << window), no group
        is ever split across a boundary.

        Group IDs are stable across runs: each cluster is identified by the
        alert_id of its earliest (lowest ts) member.
        """
        if not alerts:
            return []

        self._load_model()
        sorted_alerts = sorted(alerts, key=lambda a: a.ts)
        t_start = sorted_alerts[0].ts
        t_end = sorted_alerts[-1].ts
        span_h = (t_end - t_start) / 3600

        if t_end - t_start <= self._WINDOW_SECONDS:
            print(
                f"  [AlertBERT] Grouping {len(sorted_alerts)} alerts "
                f"(span {span_h:.1f}h, single window)..."
            )
            return self._group_chunk(sorted_alerts, window_label="1/1")

        # Count non-empty windows up front for progress display.
        n_windows = sum(
            1
            for ws in range(int(t_start), int(t_end) + 1, self._WINDOW_SECONDS)
            if any(ws <= a.ts < ws + self._WINDOW_SECONDS for a in sorted_alerts)
        )
        print(
            f"  [AlertBERT] Grouping {len(sorted_alerts)} alerts "
            f"(span {span_h:.1f}h) in {n_windows} windows of "
            f"{self._WINDOW_SECONDS // 3600}h each..."
        )

        results: list[GroupingRecord] = []
        window_idx = 0
        window_start = t_start
        while window_start <= t_end:
            window_end = window_start + self._WINDOW_SECONDS
            chunk = [a for a in sorted_alerts if window_start <= a.ts < window_end]
            if chunk:
                window_idx += 1
                label = f"{window_idx}/{n_windows}"
                if len(chunk) > self._MAX_CHUNK_ALERTS:
                    n_sub = (
                        len(chunk) + self._MAX_CHUNK_ALERTS - 1
                    ) // self._MAX_CHUNK_ALERTS
                    sub_size = len(chunk) // n_sub
                    print(
                        f"    window {label}: {len(chunk)} alerts → {n_sub} sub-chunks of ~{sub_size}",
                        flush=True,
                    )
                    for j in range(n_sub):
                        sub_start = j * sub_size
                        sub_end = sub_start + sub_size if j < n_sub - 1 else len(chunk)
                        sub_chunk = chunk[sub_start:sub_end]
                        sub_label = f"{label}.{j + 1}/{n_sub}"
                        print(
                            f"      sub-chunk {sub_label}: {len(sub_chunk)} alerts",
                            flush=True,
                        )
                        results.extend(
                            self._group_chunk(sub_chunk, window_label=sub_label)
                        )
                else:
                    print(f"    window {label}: {len(chunk)} alerts", flush=True)
                    results.extend(self._group_chunk(chunk, window_label=label))
            window_start = window_end
        return results

    def _group_chunk(
        self, alerts: list[TokenizedAlert], window_label: str = ""
    ) -> list[GroupingRecord]:
        dataset = _InferenceDataset(alerts)

        with torch.no_grad():
            cluster_labels: np.ndarray = self._grouper(dataset)

        n_clusters = len(set(int(x) for x in cluster_labels))
        if window_label:
            print(f"      → {n_clusters} groups formed", flush=True)

        cluster_to_anchor: dict[int, str] = {}
        for i, raw_label in enumerate(cluster_labels):
            label = int(raw_label)
            if label not in cluster_to_anchor:
                cluster_to_anchor[label] = dataset._alert_ids[i]

        return [
            GroupingRecord(
                alert_id=dataset._alert_ids[i],
                group_id=f"alertbert:{cluster_to_anchor[int(cluster_labels[i])]}",
                method=ALERTBERT_METHOD,
            )
            for i in range(len(dataset))
        ]


_TokenizedAlertDataset = _InferenceDataset
