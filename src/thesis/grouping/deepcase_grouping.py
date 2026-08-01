"""
DeepCASE alert grouping: trains a DeepCASE ContextBuilder (attention-based
encoder-decoder) on this thesis's own train-scenario alerts, then uses its
Interpreter (DBSCAN over attention vectors) to cluster whichever alerts are
being grouped.

    van Ede, T., Aghakhani, H., Spahn, N., Bortolameotti, R., Cova, M.,
    Continella, A., van Steen, M., Peter, A., Kruegel, C. & Vigna, G. (2022).
    DeepCASE: Semi-Supervised Contextual Analysis of Security Events.
    IEEE Symposium on Security and Privacy (S&P).

Unlike AlertBERT (thesis.grouping.alertbert_grouping), there's no pretrained
checkpoint here -- DeepCASE's ContextBuilder must be trained on this
project's own data (train-scenario alerts), then applied to whatever
scenario is being grouped. `deepcase` is a pinned git dependency
(see pyproject.toml, MIT licensed) rather than vendored into external/, since
we only use it as a library and never modify its source.

Only Interpreter.cluster() is used (not .fit()/.predict()/DeepCASE.fit()),
since those require a priori per-sample maliciousness scores that this
grouping comparison doesn't have or need -- purity/reduction are computed
externally from ground-truth labels, the same way as every other method in
group_alerts.py.

Shared vocabulary requirement
------------------------------
ContextBuilder's vocabulary size is fixed at construction (`input_size`/
`output_size`), so train and target alerts must be mapped through the same
event-id vocabulary. We build one combined DataFrame from
`train_alerts + target_alerts` and call `Preprocessor.sequence()` on it once,
then split the returned tensors back into train/target halves. Preprocessor
groups by raw `machine` (host) value across the whole combined frame, so
this assumes host names don't collide between train and target alerts --
true for this project's per-scenario AIT-ADS host naming (fox's hosts are
named distinctly from shaw's, etc.).

Row order is preserved by Preprocessor.sequence() (confirmed by reading its
source: `events`/`context` are built by indexing the input DataFrame
directly, then scattered back into that same order via the DataFrame's own
index -- not reordered by its internal per-host sort/groupby), so target
alerts line up positionally with Interpreter.cluster()'s returned cluster
ids without needing to track it separately.

ContextBuilder.load() bug workaround
--------------------------------------
The installed deepcase package's own `ContextBuilder.load()` classmethod
hardcodes `num_layers=1` and `LSTM=False` when reconstructing a model from a
saved state dict (literal `# TODO` in its source) -- silently wrong for any
other configuration. Our disk cache instead stores the exact constructor
kwargs alongside the state dict and reconstructs the model directly,
bypassing that classmethod entirely.

train_id stability
--------------------
`train_id` only identifies the *disk cache entry* -- it is not derived from
train_alerts' actual content, so two calls with the same train_id but
different train_alerts would silently reuse a stale trained model, and two
calls with the same train_alerts but different train_id strings each get
their own (redundant) cache entry. Always build it with
`train_id_for_scenarios()` from the same scenario-name list you loaded
train_alerts from, rather than hand-writing it, so the same training
scenario set always resolves to the same cache entry regardless of the
order the caller happened to list/load them in. This does not detect
changes to the underlying alert data itself (e.g. a re-tokenization that
changes alert content but not the scenario list) -- delete the relevant
artifacts/cache/deepcase/<key>/ directory if that ever happens.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter import Interpreter
from deepcase.interpreter.cluster import Cluster
from deepcase.interpreter.utils import group_by
from deepcase.preprocessing import Preprocessor

from thesis.grouping._device import resolve_device
from thesis.paths import CACHE_DIR
from thesis.schemas.groups import GroupingRecord

DEEPCASE_METHOD = "deepcase"

_CACHE_DIR = CACHE_DIR / "deepcase"

# Matches every published DeepCASE example's defaults.
DEFAULT_CONTEXT_LENGTH = 10
DEFAULT_TIMEOUT = 86400.0
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 1
DEFAULT_LSTM = False
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPS = 0.1
DEFAULT_MIN_SAMPLES = 5
DEFAULT_THRESHOLD = 0.2
DEFAULT_CLUSTER_ITERATIONS = 100
DEFAULT_CLUSTER_BATCH_SIZE = 1024
# Fixed single seed per training config (not multi-seed averaging) -- SGD
# training is otherwise non-deterministic run to run.
DEFAULT_SEED = 0


def train_id_for_scenarios(scenarios: list[str]) -> str:
    """
    Canonical, order-independent train_id for a set of training scenarios,
    e.g. train_id_for_scenarios(["wilson", "shaw", "wardbeck", "wheeler"])
    == train_id_for_scenarios(["shaw", "wardbeck", "wheeler", "wilson"])
    == "shaw+wardbeck+wheeler+wilson". Use this instead of hand-writing a
    train_id string -- see the "train_id stability" note in the module
    docstring for why an ad hoc string risks either silently reusing a
    stale cache entry or fragmenting the cache with redundant entries for
    what is actually the same training set.
    """
    slug = "+".join(sorted({s.strip().lower() for s in scenarios}))
    return slug.replace("/", "_").replace("\\", "_")


@runtime_checkable
class DeepCaseGroupableAlert(Protocol):
    """
    Structural type for alerts this method can run on. DeepCASE was designed
    around an "event type" + "machine" pair, so -- like AlertBERT's protocol
    variant -- this needs `.short` in addition to the generic GroupableAlert
    protocol in thesis.grouping.group_alerts. TokenizedAlert satisfies this
    already.
    """

    alert_id: str
    ts: int | float
    host: str | None
    short: str | None


def _cache_key(
    train_id: str,
    context_length: int,
    timeout: float,
    hidden_size: int,
    num_layers: int,
    LSTM: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> str:
    return (
        f"{train_id}__len{context_length}_to{timeout:g}_hs{hidden_size}"
        f"_nl{num_layers}_lstm{int(LSTM)}_ep{epochs}_bs{batch_size}_lr{learning_rate:g}"
        f"_seed{seed}"
    )


_context_builder_cache: dict[str, ContextBuilder] = {}


def _build_tensors(
    train_alerts: list[DeepCaseGroupableAlert],
    target_alerts: list[DeepCaseGroupableAlert],
    context_length: int,
    timeout: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Builds (train_X, train_y, target_X, target_y, mapping) by running
    Preprocessor.sequence() once over train_alerts + target_alerts combined,
    so both share one event-id vocabulary, then splitting the result back
    into train/target halves using the known split index (safe because
    Preprocessor.sequence() preserves input row order -- see module
    docstring).

    Preprocessor.sequence() always returns CPU tensors, so every split is
    moved to `device` here -- the one choke point both halves pass through.
    train_X/train_y would technically work uncopied too (ContextBuilder.fit()
    moves its inputs internally), but target_X/target_y genuinely need this:
    Interpreter.attended_context() -> ContextBuilder.query()/.predict() never
    move their input, so a GPU-resident context_builder fed CPU target
    tensors raises a device-mismatch RuntimeError without this.
    """
    combined = train_alerts + target_alerts
    df = pd.DataFrame(
        {
            "timestamp": [float(a.ts) for a in combined],
            "machine": [a.host or "_unknown" for a in combined],
            "event": [a.short or "_unknown" for a in combined],
        }
    )

    context, events, _, mapping = Preprocessor(
        length=context_length, timeout=timeout
    ).sequence(df)

    n_train = len(train_alerts)
    return (
        context[:n_train].to(device),
        events[:n_train].to(device),
        context[n_train:].to(device),
        events[n_train:].to(device),
        mapping,
    )


def _fit_or_load_context_builder(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    mapping: dict,
    train_id: str,
    context_length: int,
    timeout: float,
    hidden_size: int,
    num_layers: int,
    LSTM: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ContextBuilder:
    key = _cache_key(
        train_id,
        context_length,
        timeout,
        hidden_size,
        num_layers,
        LSTM,
        epochs,
        batch_size,
        learning_rate,
        seed,
    )
    if key in _context_builder_cache:
        return _context_builder_cache[key]

    cache_dir = _CACHE_DIR / key
    meta_path = cache_dir / "meta.json"
    state_path = cache_dir / "state_dict.pt"

    if meta_path.exists() and state_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        context_builder = ContextBuilder(
            input_size=meta["input_size"],
            output_size=meta["output_size"],
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            max_length=meta["max_length"],
            bidirectional=meta["bidirectional"],
            LSTM=meta["LSTM"],
        )
        context_builder.load_state_dict(
            torch.load(state_path, map_location=device, weights_only=True)
        )
        context_builder.to(device)
        context_builder.eval()
        _context_builder_cache[key] = context_builder
        return context_builder

    # Seeded before construction (not just .fit()) so weight initialisation
    # is reproducible too -- fixed single seed per config, per the project's
    # choice not to average multiple SGD training runs.
    torch.manual_seed(seed)

    input_size = output_size = len(mapping)
    context_builder = ContextBuilder(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        max_length=context_length,
        bidirectional=False,
        LSTM=LSTM,
    )
    context_builder.to(device)
    context_builder.fit(
        X=train_X,
        y=train_y.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    context_builder.eval()

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(context_builder.state_dict(), state_path)
    with meta_path.open("w") as f:
        json.dump(
            {
                "input_size": input_size,
                "output_size": output_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "max_length": context_length,
                "bidirectional": False,
                "LSTM": LSTM,
            },
            f,
        )

    _context_builder_cache[key] = context_builder
    return context_builder


def _cluster_ids_to_records(
    alerts: list[DeepCaseGroupableAlert],
    cluster_ids: np.ndarray,
    mask: np.ndarray,
) -> list[GroupingRecord]:
    """
    Maps DBSCAN cluster ids (aligned with `alerts`' order) to GroupingRecords
    via the anchor-id convention used across group_alerts.py (first alert_id
    seen for a cluster becomes its anchor). Cluster id -1 (DBSCAN noise or
    below-confidence-threshold, both folded into -1 by DeepCASE) gets no
    such anchor -- each -1 alert becomes its own singleton group, since
    folding every -1 alert into one group would misrepresent purity. These
    records also get is_outlier=True, so metrics that need to distinguish a
    genuine rejection from an ordinary one-alert group (e.g. coverage) can.

    `mask` (aligned with `alerts`, from Interpreter.attended_context) is what
    lets the two -1 causes be told apart for `reason`, since cluster_ids
    alone collapses both to the same value: an alert absent from `mask`
    never cleared the confidence threshold and was never even handed to
    DBSCAN ("deepcase_low_confidence"); an alert present in `mask` but
    still clustered -1 was scored by DBSCAN and called noise
    ("deepcase_dbscan_noise").
    """
    anchor_by_cluster: dict[int, str] = {}
    records: list[GroupingRecord] = []
    for alert, cluster_id, in_mask in zip(alerts, cluster_ids, mask):
        cluster_id = int(cluster_id)
        is_outlier = cluster_id == -1
        if is_outlier:
            anchor_id = alert.alert_id
            reason = (
                "deepcase_low_confidence" if not in_mask else "deepcase_dbscan_noise"
            )
        else:
            anchor_id = anchor_by_cluster.setdefault(cluster_id, alert.alert_id)
            reason = None
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=f"{DEEPCASE_METHOD}:{anchor_id}",
                method=DEEPCASE_METHOD,
                is_outlier=is_outlier,
                reason=reason,
            )
        )
    return records


def _compute_attended_vectors(
    context_builder: ContextBuilder,
    features: int,
    target_X: torch.Tensor,
    target_y: torch.Tensor,
    threshold: float,
    iterations: int,
    batch_size: int,
) -> tuple:
    """
    Runs the (expensive: `iterations`-step attention-query optimization)
    attention extraction once, returning (vectors, mask, y) for reuse across
    an eps sweep via _cluster_from_vectors. eps/min_samples are irrelevant
    to this step (they only affect the later DBSCAN step), so the throwaway
    Interpreter built here uses placeholder values.
    """
    y = target_y.reshape(-1, 1)
    interpreter = Interpreter(
        context_builder=context_builder,
        features=features,
        eps=1.0,
        min_samples=1,
        threshold=threshold,
    )
    vectors, mask = interpreter.attended_context(
        X=target_X,
        y=y,
        threshold=threshold,
        iterations=iterations,
        batch_size=batch_size,
    )
    return vectors, mask, y


def _cluster_from_vectors(
    vectors, mask: torch.Tensor, y: torch.Tensor, eps: float, min_samples: int
) -> np.ndarray:
    """
    Replicates Interpreter.cluster()'s post-attended_context logic (DBSCAN
    per target-event group, offsetting labels to stay globally unique,
    scattering back through mask) starting from precomputed vectors/mask
    instead of recomputing them -- Interpreter.cluster() itself always
    recomputes attention internally and has no way to accept precomputed
    vectors, so this is necessary to make an eps sweep cheap.
    """
    dbscan = Cluster(p=1)
    indices_y = group_by(
        y[mask].squeeze(1).cpu().numpy(), key=lambda x: x.data.tobytes()
    )

    result = np.full(int(mask.sum()), -1, dtype=int)
    for _event, context_mask in indices_y:
        clusters = dbscan.dbscan(
            X=vectors[context_mask], eps=eps, min_samples=min_samples
        )
        clusters[clusters != -1] += max(0, result.max() + 1)
        result[context_mask] = clusters

    clusters = np.full(mask.shape[0], -1, dtype=int)
    clusters[mask.cpu().numpy()] = result
    return clusters


def group_alerts_deepcase_many_eps(
    alerts: list[DeepCaseGroupableAlert],
    train_alerts: list[DeepCaseGroupableAlert],
    train_id: str,
    eps_values: list[float],
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    timeout: float = DEFAULT_TIMEOUT,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    LSTM: bool = DEFAULT_LSTM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    cluster_iterations: int = DEFAULT_CLUSTER_ITERATIONS,
    cluster_batch_size: int = DEFAULT_CLUSTER_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str | torch.device | None = None,
) -> dict[float, list[GroupingRecord]]:
    """
    Like group_alerts_deepcase, but evaluates every value in `eps_values`
    against ONE trained ContextBuilder and ONE attention-query pass, instead
    of recomputing the (expensive) attention-query optimization once per
    eps. Use this instead of calling group_alerts_deepcase in a loop over
    eps -- context_length/hidden_size/num_layers/LSTM still require
    retraining (a fresh entry in artifacts/cache/deepcase/), but eps is a
    DBSCAN-only parameter applied on top of the same precomputed vectors.

    `device` (None/"auto"/"cpu"/"cuda"/"mps"/...) is resolved once via
    thesis.grouping._device.resolve_device -- see that module for the
    mps->cuda->cpu auto-detect order.

    Returns a dict keyed by each value in eps_values (not deduplicated --
    pass unique values).
    """
    if not alerts:
        return {eps: [] for eps in eps_values}

    resolved_device = resolve_device(device)

    train_X, train_y, target_X, target_y, mapping = _build_tensors(
        train_alerts, alerts, context_length, timeout, resolved_device
    )

    context_builder = _fit_or_load_context_builder(
        train_X,
        train_y,
        mapping,
        train_id,
        context_length,
        timeout,
        hidden_size,
        num_layers,
        LSTM,
        epochs,
        batch_size,
        learning_rate,
        seed,
        resolved_device,
    )

    vectors, mask, y = _compute_attended_vectors(
        context_builder,
        len(mapping),
        target_X,
        target_y,
        threshold,
        cluster_iterations,
        cluster_batch_size,
    )

    mask_np = mask.cpu().numpy()
    results: dict[float, list[GroupingRecord]] = {}
    for eps in eps_values:
        cluster_ids = _cluster_from_vectors(vectors, mask, y, eps, min_samples)
        results[eps] = _cluster_ids_to_records(alerts, cluster_ids, mask_np)
    return results


def group_alerts_deepcase(
    alerts: list[DeepCaseGroupableAlert],
    train_alerts: list[DeepCaseGroupableAlert],
    train_id: str,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    timeout: float = DEFAULT_TIMEOUT,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    LSTM: bool = DEFAULT_LSTM,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    cluster_iterations: int = DEFAULT_CLUSTER_ITERATIONS,
    cluster_batch_size: int = DEFAULT_CLUSTER_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str | torch.device | None = None,
) -> list[GroupingRecord]:
    """
    Groups `alerts` using a DeepCASE ContextBuilder trained on `train_alerts`
    (e.g. this project's train-scenario alerts: shaw/wardbeck/wheeler/
    wilson) and an Interpreter clustering `alerts`' own attention vectors.
    Single-eps convenience wrapper around group_alerts_deepcase_many_eps --
    prefer that function directly when sweeping several eps values, since
    calling this in a loop would redundantly recompute the attention query
    once per eps.

    `train_id` names the training set for on-disk caching purposes only --
    build it with `train_id_for_scenarios()` rather than hand-writing it
    (see that function's docstring and the module docstring's "train_id
    stability" note for why). Retraining is skipped whenever a checkpoint
    for the same train_id + hyperparameters + seed already exists in
    artifacts/cache/deepcase/.

    context_length/hidden_size/num_layers/LSTM are ContextBuilder
    architecture parameters -- changing any of them requires retraining
    (a fresh cache entry). eps/min_samples/threshold are Interpreter-only
    clustering parameters applied on top of whatever ContextBuilder is
    already cached, so sweeping just those is cheap.
    """
    return group_alerts_deepcase_many_eps(
        alerts,
        train_alerts,
        train_id,
        eps_values=[eps],
        context_length=context_length,
        timeout=timeout,
        hidden_size=hidden_size,
        num_layers=num_layers,
        LSTM=LSTM,
        min_samples=min_samples,
        threshold=threshold,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        cluster_iterations=cluster_iterations,
        cluster_batch_size=cluster_batch_size,
        seed=seed,
        device=device,
    )[eps]


def group_alerts_deepcase_by_group(
    alerts: list[DeepCaseGroupableAlert],
    train_alerts: list[DeepCaseGroupableAlert],
    train_id: str,
    **kwargs,
) -> dict[str, list[DeepCaseGroupableAlert]]:
    records = group_alerts_deepcase(alerts, train_alerts, train_id, **kwargs)
    alert_by_id = {a.alert_id: a for a in alerts}
    groups: dict[str, list[DeepCaseGroupableAlert]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(alert_by_id[record.alert_id])
    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}
