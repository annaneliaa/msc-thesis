"""
AlertBERT alert grouping: groups alerts using embeddings from a pretrained
AlertBERT masked-language-model checkpoint (external/AlertBERT/saved_models).
Inference only -- no training happens here.

Checkpoint / scenario compatibility
------------------------------------
The default checkpoint (`mlm_1l_4h_16d_original_1_60k`) was trained on the
"original" (least-augmented) AIT-ADS-A config using the scenario split
defined in `alertbert.aitads.aitads_train_val_test_split`:

    train = shaw, wardbeck, wheeler, wilson
    val   = harrison, santos
    test  = fox, russellmitchell

This matches the 4 scenarios this thesis groups on, so running this
checkpoint against shaw/wardbeck/wheeler/wilson is in-training-distribution
for the checkpoint. Do not use fox/russellmitchell to pick delta/theta for
this checkpoint -- they were held out during AlertBERT's own training.

Of the three head-count variants trained on "original" (1h/2h/4h -- same
data, same layer/dim config, differ only in attention heads), 4h has the
best validation macro-F1 on the "short" alert-type target (0.548 vs.
0.472/0.456 for 1h/2h; see each checkpoint's report.json), so it's the
default here.

Runtime requirement
--------------------
`alertbert.models` imports `graph_tool` at module level for its connected
-components clustering step. graph-tool is conda/mamba-only (not
pip-installable) and is NOT part of this project's plain `venv/`. It IS
available (with the `alertbert` package installed editable) in the
`thesis-alertbert` conda environment -- use that env to call anything in
this module:

    conda activate thesis-alertbert

macOS note: that environment hits an OpenMP double-init abort (SIGABRT)
from numpy/torch both bundling libomp unless `KMP_DUPLICATE_LIB_OK=TRUE`
is set in the environment before Python starts.

delta/theta sweep
------------------
Sweep delta and theta over the same log-scale grid used for time_delta
(see thesis.baselines.grouping.time_delta's `time_delta_grid` / `TIME_DELTA_VALUES`:
`a * 2**i` for `i` in `range(-7, 14)`, `a` in `(1, 1.5)`) -- this mirrors
the AlertBERT paper's own evaluation protocol
(`alertbert.eval_grouping.timedelta_roc_traj_*`). theta must be >= delta;
`alertbert.models.AlertBERT` raises `AssertionError` otherwise.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from alertbert.aitads import AlertDataset, aitads_train_external_mail_hosts
from alertbert.models import AlertBERT as _AlertBertGroupingModel
from alertbert.models import MaskedLangModelInferenceWrapper, MaskedLanguageModel
from alertbert.preprocessing import (
    BaseSequenceCollate,
    default_collate_fn,
    load_feature_vocabs,
)

from thesis.grouping._device import resolve_device
from thesis.paths import ROOT
from thesis.schemas.groups import GroupingRecord

ALERTBERT_METHOD = "alertbert"

_SAVED_MODELS_DIR = ROOT / "external" / "AlertBERT" / "saved_models"

# See module docstring for why 4h over 1h/2h.
DEFAULT_CHECKPOINT = "mlm_1l_4h_16d_original_1_60k"

# Matches alertbert.eval_grouping's own grouping-model protocol.
ALERTBERT_DIM_REDUCTION = 2
ALERTBERT_LAYERS = ("embedding", "encoder")


@runtime_checkable
class AlertBertGroupableAlert(Protocol):
    """
    Structural type for alerts this method can run on. AlertBERT was
    trained on the short+host feature pair, so -- unlike the generic
    GroupableAlert protocol in thesis.grouping.group_alerts -- this needs
    `.short` too. TokenizedAlert satisfies this already.
    """

    alert_id: str
    ts: int | float
    host: str | None
    short: str | None


class _ScipyAlertBERT(_AlertBertGroupingModel):
    """
    AlertBERT grouping model with connected components computed via scipy
    instead of graph-tool. Both are exact (not approximate) connected
    -components algorithms; scipy is just slower on very large graphs.
    Override kept in case a future environment has graph-tool but callers
    still want the portable path.
    """

    def get_connected_components(
        self,
        coords_0: np.ndarray,
        coords_1: np.ndarray,
        n_nodes: int,
        library: str = "scipy",  # noqa: ARG002
    ) -> tuple[np.ndarray, int]:
        return super().get_connected_components(
            coords_0, coords_1, n_nodes, library="scipy"
        )


class _DeviceCollate:
    """
    Wraps a collate_fn so its output TensorDict lands on `device` before the
    model consumes it. BaseSequenceCollate.__call__ always builds its
    TensorDict on CPU (no device argument anywhere in the vendored
    preprocessing code), while the model itself is moved to `device` in
    _load_checkpoint below -- without this, AlertBERT.get_embeddings's
    `self.model(batch)` call raises a device-mismatch RuntimeError as soon
    as `device` is anything other than cpu. TensorDict.to(device) moves
    every contained tensor at once.
    """

    def __init__(self, collate_fn: BaseSequenceCollate, device: torch.device) -> None:
        self._collate_fn = collate_fn
        self._device = device

    def __call__(self, batch):
        return self._collate_fn(batch).to(self._device)


class _LoadedCheckpoint:
    __slots__ = ("inference_model", "collate_fn", "params")

    def __init__(
        self,
        inference_model: MaskedLangModelInferenceWrapper,
        collate_fn: BaseSequenceCollate,
        params: dict,
    ) -> None:
        self.inference_model = inference_model
        self.collate_fn = collate_fn
        self.params = params


_checkpoint_cache: dict[tuple[str, str], _LoadedCheckpoint] = {}


def _load_checkpoint(checkpoint: str, device: torch.device) -> _LoadedCheckpoint:
    """Loads (and caches) a checkpoint's model + vocabs for inference."""
    cache_key = (checkpoint, str(device))
    if cache_key in _checkpoint_cache:
        return _checkpoint_cache[cache_key]

    ckpt_dir = _SAVED_MODELS_DIR / checkpoint
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"AlertBERT checkpoint not found: {ckpt_dir}. "
            f"Available checkpoints: {sorted(p.name for p in _SAVED_MODELS_DIR.iterdir() if p.is_dir())}"
        )

    with (ckpt_dir / "report.json").open() as f:
        params = json.load(f)["params"]

    features = set(params["features"]) | set(params["targets"])
    vocabs = load_feature_vocabs(str(ckpt_dir), sorted(features), params["min_freq"])
    if "host" in features:
        vocabs["host"].remove(aitads_train_external_mail_hosts)

    model = MaskedLanguageModel(params=params, vocabs=vocabs)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", weights_only=True, map_location=device)
    )
    model.to(device)
    model.eval()

    inference_model = MaskedLangModelInferenceWrapper(model, layers=ALERTBERT_LAYERS)

    collate_fn_map = dict(vocabs)
    collate_fn_map[params["encoding"]] = default_collate_fn
    collate_fn = _DeviceCollate(BaseSequenceCollate(collate_fn_map), device)

    loaded = _LoadedCheckpoint(inference_model, collate_fn, params)
    _checkpoint_cache[cache_key] = loaded
    return loaded


def _to_alert_dataset(
    alerts: list[AlertBertGroupableAlert],
) -> tuple[AlertDataset, list[AlertBertGroupableAlert]]:
    """
    Builds an in-memory AlertDataset from already-tokenized alerts,
    bypassing AlertDataset's file-loading __init__ (which expects AIT-ADS's
    own on-disk JSON format). raw_time is shifted to start at 0 (matching
    how the augmented training data's per-scenario files start near 0) so
    values stay small -- AlertDataset's rotary time encoding takes raw_time
    as a continuous position, and epoch-second magnitudes (~1.7e9) would
    only add imprecision without changing the relative gaps that matter.
    """
    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    ts = np.array([a.ts for a in sorted_alerts], dtype=np.float64)
    # float32 to match the model's own parameter dtype (default_collate_fn
    # would otherwise produce a float64 tensor, which fails to matmul
    # against the model's float32 weights).
    raw_time = (ts - ts.min() if len(ts) else ts).astype(np.float32)

    dataset = AlertDataset.__new__(AlertDataset)
    dataset.data = {
        "short": np.array([a.short or "<UNK>" for a in sorted_alerts], dtype=object),
        "host": np.array([a.host or "<UNK>" for a in sorted_alerts], dtype=object),
        "raw_time": raw_time,
    }
    dataset.keys = ["short", "host", "raw_time"]
    return dataset, sorted_alerts


def group_alerts_alertbert(
    alerts: list[AlertBertGroupableAlert],
    delta: float,
    theta: float,
    checkpoint: str = DEFAULT_CHECKPOINT,
    dim_reduction: int = ALERTBERT_DIM_REDUCTION,
    device: str | torch.device | None = None,
) -> list[GroupingRecord]:
    """
    Groups alerts using a pretrained AlertBERT checkpoint (no training
    here). See module docstring for checkpoint/scenario compatibility and
    the delta/theta sweep grid. Requires the `thesis-alertbert` conda env
    (graph-tool + the alertbert package); see module docstring.

    `device` (None/"auto"/"cpu"/"cuda"/"mps"/...) is resolved once via
    thesis.grouping._device.resolve_device -- see that module for the
    mps->cuda->cpu auto-detect order.
    """
    if not alerts:
        return []

    loaded = _load_checkpoint(checkpoint, resolve_device(device))
    dataset, sorted_alerts = _to_alert_dataset(alerts)

    grouping_model = _ScipyAlertBERT(
        model=loaded.inference_model,
        collate_fn=loaded.collate_fn,
        dim_reduction=dim_reduction,
        delta=delta,
        theta=theta,
    )
    cluster_ids = grouping_model(dataset)

    anchor_by_cluster: dict[int, str] = {}
    records: list[GroupingRecord] = []
    for alert, cluster_id in zip(sorted_alerts, cluster_ids):
        cluster_id = int(cluster_id)
        anchor_id = anchor_by_cluster.setdefault(cluster_id, alert.alert_id)
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=f"{ALERTBERT_METHOD}:{anchor_id}",
                method=ALERTBERT_METHOD,
            )
        )
    return records


def group_alerts_alertbert_by_group(
    alerts: list[AlertBertGroupableAlert],
    delta: float,
    theta: float,
    checkpoint: str = DEFAULT_CHECKPOINT,
    dim_reduction: int = ALERTBERT_DIM_REDUCTION,
    device: str | torch.device | None = None,
) -> dict[str, list[AlertBertGroupableAlert]]:
    records = group_alerts_alertbert(
        alerts,
        delta=delta,
        theta=theta,
        checkpoint=checkpoint,
        dim_reduction=dim_reduction,
        device=device,
    )
    alert_by_id = {a.alert_id: a for a in alerts}
    groups: dict[str, list[AlertBertGroupableAlert]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(alert_by_id[record.alert_id])
    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}
