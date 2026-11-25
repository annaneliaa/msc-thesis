"""
DeepCASE on CSCAS (DeepCASE comparison plan, step 5.2): the frozen DeepCASE
run our system is compared against.

    van Ede et al., DeepCASE: Semi-Supervised Contextual Analysis of Security
    Events, IEEE S&P 2022.

Setup (plan §1, §2.1)
----------------------
* One CSCAS row = one DeepCASE event. Event type = SignatureID, machine =
  IntIP. Rows without an IntIP (IntIP "-1", ~24% of CSCAS) get no context by
  default (`missing_host="none"`); `missing_host="extip"` uses ExtIP instead.
* Context = the machine's previous `context_length` events, each at most
  `timeout` seconds older (DeepCASE's own Preprocessor.sequence rule).
* Split = the CSCAS baseline split (2022-01-26 06:23:21+02:00): W0 = the
  6-day training window, the 54-day remainder is cut into horizon windows at
  `granularity` with the same compute_window_bounds convention as
  temporal_decay's "baseline_split" mode, so horizon h here is horizon h
  there.
* Vocabulary = SignatureIDs seen in the training rows only. A SignatureID
  first seen later maps to one UNKNOWN token (in context and as the event);
  an UNKNOWN event is always sent to the analyst (code -2). This replaces
  the shared train+test vocabulary of thesis.grouping.deepcase_grouping,
  which leaks the test vocabulary and crashed on santos.
* Tuning on W0 only (paper App. D): W0 split 50:50 by time, grid over
  context length x tau_confidence x eps, best by `tune_objective`.
* Simulated analyst (paper §V-B3/B4): per cluster, sample
  `samples_per_cluster` (10) members and label the cluster with the highest
  label seen (conservative rule); every outlier (low confidence or DBSCAN
  noise) is inspected individually. B0 = cluster samples + outliers.
* Test-time decisions follow Interpreter.predict: a score >= 0 is an
  automatic decision; -1 (not confident), -2 (event not in training) and -3
  (nearest cluster further than eps) go to the analyst.

Metrics (plan §2.3)
--------------------
* covered_*: precision/recall/F1 over automatically decided rows only
  (paper Table V style) + coverage.
* e2e_* (end to end): a row is escalated if DeepCASE flags it OR sends it to
  the analyst; a missed attack is an attack DeepCASE auto-labels benign.
  analyst_items = escalated rows; workload_reduction = 1 - analyst_items / n
  (the paper's "1 - (alerts + uncovered) / total").
* Each metric is also split by seen / unseen SignatureID (seen = appears in
  W0), plan §1.1.

The attention query (the expensive step) does not depend on tau or eps, so
it runs once per (context length, seed) and the tau x eps grid reuses it.
Clustering, cluster scoring and prediction mirror deepcase.interpreter
(Interpreter.cluster / score_clusters / score / predict) on those cached
vectors.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter import Interpreter
from deepcase.interpreter.cluster import Cluster
from deepcase.interpreter.utils import sp_unique
from sklearn.neighbors import KDTree

from thesis.grouping._device import resolve_device
from thesis.grouping.deepcase_grouping import _fit_or_load_context_builder
from thesis.paths import ROOT
from thesis.pipeline.pipeline import compute_window_bounds
from thesis.preprocessing.parsing import cscas_group_id

CSCAS_CSV_PATH = ROOT / "data" / "cscas" / "dataset-labeled-anon-ip.csv"
# Same constant as baselines/cscas_base.py and signature_leakage.py.
CSCAS_BASELINE_SPLIT_TIME = "2022-01-26 06:23:21+02:00"
EXPECTED_TRAIN_ROWS = 139_532
EXPECTED_TEST_ROWS = 1_255_792

DEFAULT_OUT_ROOT = ROOT / "artifacts" / "experiments" / "deepcase_comparison"

# Interpreter.predict codes.
CODE_LOW_CONFIDENCE = -1
CODE_NOT_IN_TRAINING = -2
CODE_TOO_FAR = -3
NO_SCORE = -1

GRID_0_1_TO_1 = tuple(round(0.1 * i, 1) for i in range(1, 11))
TUNE_OBJECTIVES = ("e2e_f1", "covered_f1")


@dataclass
class DeepCaseConfig:
    csv_path: str = str(CSCAS_CSV_PATH)
    split_time: str = CSCAS_BASELINE_SPLIT_TIME
    granularity: float = 0.1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # Sequencing (paper Table IX: 10 events / 1 day).
    context_length: int = 10
    timeout: float = 86400.0
    missing_host: str = "none"  # "none" | "extip"
    # ContextBuilder (paper Table IX: hidden 128, 100 epochs; delta=0.1 is
    # hardcoded in ContextBuilder.fit).
    hidden_size: int = 128
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.01
    # Interpreter (paper Table IX: tau 0.2, eps 0.1, min cluster size 5).
    threshold: float = 0.2
    eps: float = 0.1
    min_samples: int = 5
    query_iterations: int = 100
    query_batch_size: int = 1024
    # Simulated analyst.
    samples_per_cluster: int = 10
    # Tuning on W0 (paper App. D). When tune=True the tuned context_length /
    # threshold / eps replace the values above.
    tune: bool = True
    tune_context_lengths: tuple[int, ...] = (5, 10, 20)
    tune_thresholds: tuple[float, ...] = GRID_0_1_TO_1
    tune_eps: tuple[float, ...] = GRID_0_1_TO_1
    tune_objective: str = "e2e_f1"
    tune_epochs: int | None = None  # None -> epochs
    tune_seed: int = 0
    # Smoke tests: keep only the last N training rows / first N test rows.
    max_train_rows: int | None = None
    max_test_rows: int | None = None
    device: str | None = None
    out_dir: str | None = None

    def __post_init__(self) -> None:
        if self.missing_host not in ("none", "extip"):
            raise ValueError(
                f"missing_host must be none|extip, got {self.missing_host!r}"
            )
        if self.tune_objective not in TUNE_OBJECTIVES:
            raise ValueError(f"tune_objective must be one of {TUNE_OBJECTIVES}")


# --- Data -------------------------------------------------------------------


@dataclass
class CscasEvents:
    """CSCAS rows in the same chronological order as the cscas alert_groups
    (stable sort on epoch seconds over CSV file order)."""

    ts: np.ndarray  # int64 epoch seconds
    signature: np.ndarray  # int64 SignatureID
    machine: np.ndarray  # object: host key or None
    label: np.ndarray  # int8
    group_id: np.ndarray  # object: cscas_group_id, joins to AlertGroup.group_id
    split_idx: int  # rows [0, split_idx) are W0

    def __len__(self) -> int:
        return len(self.ts)

    def take(self, idx: np.ndarray, split_idx: int) -> "CscasEvents":
        return CscasEvents(
            self.ts[idx],
            self.signature[idx],
            self.machine[idx],
            self.label[idx],
            self.group_id[idx],
            split_idx,
        )


def _normalise_ip(values: pd.Series) -> pd.Series:
    """IntIP "-1"/"" -> None, exactly as IncomingSuricataGroup.from_row (and
    signature_leakage.load_cscas_signature_lookup) do."""
    s = values.map(lambda v: str(v).strip())
    return s.where(~s.isin(["-1", ""]), None)


def load_cscas_events(
    csv_path: str | Path = CSCAS_CSV_PATH,
    split_time: str = CSCAS_BASELINE_SPLIT_TIME,
    missing_host: str = "none",
) -> CscasEvents:
    df = pd.read_csv(
        csv_path, usecols=["Timestamp", "SignatureID", "ExtIP", "IntIP", "Label"]
    )
    int_ip = _normalise_ip(df["IntIP"])
    group_id = np.array(
        [
            cscas_group_id(int(s), str(e), i, str(t))
            for s, e, i, t in zip(
                df["SignatureID"], df["ExtIP"], int_ip, df["Timestamp"]
            )
        ],
        dtype=object,
    )
    ts = (
        pd.to_datetime(df["Timestamp"], utc=True)
        .dt.tz_convert(None)
        .to_numpy()
        .astype("datetime64[s]")
        .astype(np.int64)
    )
    machine = int_ip.map(lambda v: None if v is None else f"int:{v}")
    if missing_host == "extip":
        ext = df["ExtIP"].map(lambda v: f"ext:{str(v).strip()}")
        machine = machine.where(machine.notna(), ext)
    machine = machine.to_numpy(dtype=object)

    order = np.argsort(ts, kind="stable")
    cutoff = int(pd.Timestamp(split_time).timestamp())
    ts = ts[order]
    split_idx = int(np.searchsorted(ts, cutoff, side="right"))
    return CscasEvents(
        ts=ts,
        signature=df["SignatureID"].to_numpy(np.int64)[order],
        machine=machine[order],
        label=df["Label"].to_numpy(np.int8)[order],
        group_id=group_id[order],
        split_idx=split_idx,
    )


def truncate_events(
    ev: CscasEvents, max_train_rows: int | None, max_test_rows: int | None
) -> CscasEvents:
    """Smoke-test slice: the last `max_train_rows` W0 rows and the first
    `max_test_rows` test rows (kept adjacent so the split stays real)."""
    if max_train_rows is None and max_test_rows is None:
        return ev
    tr_start = max(0, ev.split_idx - max_train_rows) if max_train_rows else 0
    te_end = min(len(ev), ev.split_idx + max_test_rows) if max_test_rows else len(ev)
    idx = np.arange(tr_start, te_end)
    return ev.take(idx, ev.split_idx - tr_start)


# --- Sequencing ---------------------------------------------------------------


@dataclass
class Vocab:
    """SignatureID -> event id over the training rows; one UNKNOWN id for
    every SignatureID first seen later, then NO_EVENT (context padding)."""

    signatures: np.ndarray  # sorted known SignatureIDs

    @property
    def unknown_id(self) -> int:
        return len(self.signatures)

    @property
    def no_event_id(self) -> int:
        return len(self.signatures) + 1

    @property
    def size(self) -> int:
        return len(self.signatures) + 2

    @classmethod
    def from_signatures(cls, signatures: np.ndarray) -> "Vocab":
        return cls(np.unique(signatures))

    def encode(self, signatures: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.signatures, signatures)
        pos = np.clip(pos, 0, max(len(self.signatures) - 1, 0))
        known = (len(self.signatures) > 0) & (self.signatures[pos] == signatures)
        return np.where(known, pos, self.unknown_id).astype(np.int64)


def build_context(
    events: np.ndarray,
    machine: np.ndarray,
    ts: np.ndarray,
    length: int,
    timeout: float,
    no_event_id: int,
) -> np.ndarray:
    """(n, length) context matrix, most recent event in the last column, as
    deepcase Preprocessor.sequence builds it: position i holds the machine's
    (i+1)-th previous event if it is at most `timeout` seconds older, else
    NO_EVENT. Rows must be in chronological order; ties keep row order. Rows
    with machine None get no context."""
    n = len(events)
    codes, _ = pd.factorize(pd.Series(machine, dtype=object))  # None -> -1
    order = np.lexsort((np.arange(n), codes))
    m_s, e_s, t_s = codes[order], events[order], ts[order]
    ctx_s = np.full((n, length), no_event_id, dtype=np.int64)
    for k in range(1, min(length, n - 1) + 1):
        same = (
            (m_s[k:] == m_s[:-k]) & (m_s[k:] != -1) & ((t_s[k:] - t_s[:-k]) <= timeout)
        )
        ctx_s[k:, length - k] = np.where(same, e_s[:-k], no_event_id)
    ctx = np.empty_like(ctx_s)
    ctx[order] = ctx_s
    return ctx


# --- ContextBuilder + attention ---------------------------------------------------


def train_context_builder(
    ctx: np.ndarray,
    events: np.ndarray,
    vocab: Vocab,
    train_id: str,
    cfg: DeepCaseConfig,
    context_length: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> ContextBuilder:
    """Fit (or load from artifacts/cache/deepcase/) a ContextBuilder via the
    grouping module's cache, which also works around ContextBuilder.load()'s
    hardcoded num_layers/LSTM bug."""
    X = torch.as_tensor(ctx, dtype=torch.long, device=device)
    y = torch.as_tensor(events, dtype=torch.long, device=device)
    return _fit_or_load_context_builder(
        X,
        y,
        dict.fromkeys(range(vocab.size)),  # only len() is used
        train_id,
        context_length,
        cfg.timeout,
        cfg.hidden_size,
        1,
        False,
        epochs,
        cfg.batch_size,
        cfg.learning_rate,
        seed,
        device,
    )


@dataclass
class Attended:
    """Attention-query output for one set of rows (CPU)."""

    X: np.ndarray  # (n, L) context ids
    y: np.ndarray  # (n,) event ids
    confidence: np.ndarray  # (n,) confidence in the observed event
    attention: np.ndarray  # (n, L)


def attention_query(
    cb: ContextBuilder,
    vocab: Vocab,
    ctx: np.ndarray,
    events: np.ndarray,
    cfg: DeepCaseConfig,
    device: torch.device,
    seed: int,
) -> Attended:
    torch.manual_seed(seed)
    interp = Interpreter(
        context_builder=cb, features=vocab.size, eps=1.0, min_samples=1
    )
    X = torch.as_tensor(ctx, dtype=torch.long, device=device)
    y = torch.as_tensor(events, dtype=torch.long, device=device).reshape(-1, 1)
    # No torch.no_grad(): the query optimises the attention by gradient descent.
    confidence, attention = interp.attention_query(
        X=X,
        y=y,
        iterations=cfg.query_iterations,
        batch_size=cfg.query_batch_size,
    )
    return Attended(
        X=ctx,
        y=events,
        confidence=confidence.detach().cpu().numpy().astype(np.float64),
        attention=attention.detach().cpu().numpy().astype(np.float32),
    )


def vectorize(
    att: Attended, threshold: float, size: int
) -> tuple[sp.csr_matrix, np.ndarray]:
    """(vectors of the confident rows, mask), as Interpreter.attended_context:
    mask = confidence >= threshold, vector = total attention per event id,
    rounded to 4 decimals."""
    mask = att.confidence >= threshold
    X, A = att.X[mask], att.attention[mask]
    m, L = X.shape
    rows = np.repeat(np.arange(m), L)
    vec = sp.csr_matrix((A.ravel(), (rows, X.ravel())), shape=(m, size))
    vec.data = np.round(vec.data, 4)
    vec.eliminate_zeros()
    return vec, mask


# --- Interpreter: cluster, simulated analyst, fit, predict ---------------------


def _groups(values: np.ndarray) -> list[tuple[int, np.ndarray]]:
    order = np.argsort(values, kind="stable")
    uniq, starts = np.unique(values[order], return_index=True)
    bounds = list(starts[1:]) + [len(values)]
    return [(int(u), order[s:e]) for u, s, e in zip(uniq, starts, bounds)]


def cluster_rows(
    vec: sp.csr_matrix, mask: np.ndarray, y: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    """Cluster id per row (-1 = not confident or DBSCAN noise): DBSCAN (L1)
    per event type over the confident rows, ids offset to be globally
    unique -- Interpreter.cluster()."""
    dbscan = Cluster(p=1)
    y_m = y[mask]
    result = np.full(len(y_m), -1, dtype=np.int64)
    for _event, idx in _groups(y_m):
        c = dbscan.dbscan(X=vec[idx], eps=eps, min_samples=min_samples)
        c[c != -1] += max(0, result.max() + 1)
        result[idx] = c
    clusters = np.full(len(y), -1, dtype=np.int64)
    clusters[mask] = result
    return clusters


@dataclass
class AnalystResult:
    cluster_label: dict[int, int]
    sampled_idx: np.ndarray  # rows inspected as cluster samples
    outlier_idx: np.ndarray  # rows inspected individually
    stats: dict = field(default_factory=dict)


def simulated_analyst(
    clusters: np.ndarray,
    mask: np.ndarray,
    label: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> AnalystResult:
    """Paper §V-B3/B4 manual mode: per cluster sample min(k, size) members
    and label the cluster with the highest label among them (conservative
    rule, = attack if any sampled member is an attack); every outlier is
    inspected individually."""
    cluster_label: dict[int, int] = {}
    sampled: list[np.ndarray] = []
    n_mixed = n_mislabelled = n_attack_clusters = 0
    for cid, idx in _groups(clusters):
        if cid == -1:
            continue
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        sampled.append(np.sort(pick))
        lab = int(label[pick].max())
        cluster_label[cid] = lab
        n_attack_clusters += lab
        true = label[idx]
        n_mixed += int(true.min() != true.max())
        n_mislabelled += int((true != lab).sum())
    sampled_idx = np.concatenate(sampled) if sampled else np.array([], dtype=np.int64)
    outlier_idx = np.flatnonzero(clusters == -1)
    n_low_conf = int((~mask).sum())
    sizes = np.bincount(clusters[clusters != -1]) if cluster_label else np.array([0])
    stats = {
        "n_rows": int(len(clusters)),
        "n_clusters": len(cluster_label),
        "n_attack_clusters": n_attack_clusters,
        "n_mixed_clusters": n_mixed,
        "n_cluster_samples": int(len(sampled_idx)),
        "n_outliers": int(len(outlier_idx)),
        "n_outliers_low_confidence": n_low_conf,
        "n_outliers_dbscan_noise": int(len(outlier_idx)) - n_low_conf,
        "n_clustered_rows": int((clusters != -1).sum()),
        "n_clustered_rows_mislabelled": n_mislabelled,
        "cluster_size_median": float(np.median(sizes[sizes > 0]))
        if cluster_label
        else 0.0,
        "cluster_size_max": int(sizes.max()) if cluster_label else 0,
        # B0 (plan §2.1): cluster samples + outliers; footnote variant
        # without outliers.
        "budget_b0": int(len(sampled_idx) + len(outlier_idx)),
        "budget_clusters_only": int(len(sampled_idx)),
        "manual_workload_reduction": 1.0
        - (len(sampled_idx) + len(outlier_idx)) / max(1, len(clusters)),
    }
    return AnalystResult(cluster_label, sampled_idx, outlier_idx, stats)


@dataclass
class FittedInterpreter:
    """Per event type: KDTree (L1) over the unique clustered vectors and the
    label of each tree point (max over rows sharing that vector) --
    Interpreter.score()."""

    # event type -> (tree, label per tree point, training cluster id per tree
    # point -- the cluster with the highest label among rows sharing the
    # vector, i.e. the one that decided that label)
    trees: dict[int, tuple[KDTree, np.ndarray, np.ndarray]]
    eps: float


def fit_interpreter(
    vec: sp.csr_matrix,
    mask: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    cluster_label: dict[int, int],
    eps: float,
) -> FittedInterpreter:
    y_m, c_m = y[mask], clusters[mask]
    keep = c_m != -1
    vec_k, y_k = vec[np.flatnonzero(keep)], y_m[keep]
    cids = c_m[keep]
    scores = np.array([cluster_label[c] for c in cids], dtype=np.float64)
    trees: dict[int, tuple[KDTree, np.ndarray, np.ndarray]] = {}
    for event, idx in _groups(y_k):
        uniq, inverse, _ = sp_unique(vec_k[idx])
        inverse = np.asarray(inverse).ravel()
        labels = np.full(uniq.shape[0], -np.inf)
        np.maximum.at(labels, inverse, scores[idx])
        cluster_of = np.full(uniq.shape[0], -1, dtype=np.int64)
        order = np.argsort(scores[idx], kind="stable")  # highest label last
        cluster_of[inverse[order]] = cids[idx][order]
        trees[event] = (KDTree(uniq.toarray(), p=1), labels, cluster_of)
    return FittedInterpreter(trees, eps)


def predict_codes_and_clusters(
    fitted: FittedInterpreter,
    vec: sp.csr_matrix,
    mask: np.ndarray,
    y: np.ndarray,
    unknown_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """predict_codes plus, per row, the training cluster whose point decided
    an automatic code (-1 for every non-automatic row). Used by the monitor's
    per-cluster precision signal (DeepCASE plan B5)."""
    y_m = y[mask]
    result = np.full(len(y_m), np.nan)
    cluster_m = np.full(len(y_m), -1, dtype=np.int64)
    for event, idx in _groups(y_m):
        if event not in fitted.trees:
            result[idx] = CODE_NOT_IN_TRAINING
            continue
        tree, labels, cluster_of = fitted.trees[event]
        uniq, inverse, _ = sp_unique(vec[idx])
        inverse = np.asarray(inverse).ravel()
        dist, nb = tree.query(
            uniq.toarray(), k=1, return_distance=True, dualtree=uniq.shape[0] >= 1e3
        )
        near = dist[:, 0] <= fitted.eps
        res = np.where(near, labels[nb[:, 0]], CODE_TOO_FAR)
        result[idx] = res[inverse]
        cluster_m[idx] = np.where(near, cluster_of[nb[:, 0]], -1)[inverse]
    codes = np.full(len(y), float(CODE_LOW_CONFIDENCE))
    codes[mask] = result
    cluster = np.full(len(y), -1, dtype=np.int64)
    cluster[mask] = cluster_m
    unknown = y == unknown_id
    codes[unknown] = CODE_NOT_IN_TRAINING
    cluster[unknown] = -1
    return codes, cluster


def predict_codes(
    fitted: FittedInterpreter,
    vec: sp.csr_matrix,
    mask: np.ndarray,
    y: np.ndarray,
    unknown_id: int,
) -> np.ndarray:
    """Interpreter.predict on precomputed vectors: label of the nearest
    training cluster point if within eps, else -3; -2 if the event type has
    no clusters (or is UNKNOWN); -1 if not confident."""
    return predict_codes_and_clusters(fitted, vec, mask, y, unknown_id)[0]


# --- Metrics -----------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = (
        2 * p * r / (p + r)
        if tp and (p + r)
        else (0.0 if tp + fp + fn else float("nan"))
    )
    return p, r, f


def decision_metrics(label: np.ndarray, codes: np.ndarray, days: float) -> dict:
    label = label.astype(bool)
    auto = codes >= 0
    flag = auto & (codes > 0)
    manual = ~auto
    n = len(label)
    c_tp = int((flag & label).sum())
    c_fp = int((flag & ~label).sum())
    c_fn = int((auto & ~flag & label).sum())
    escalated = flag | manual
    e_tp = int((escalated & label).sum())
    e_fp = int((escalated & ~label).sum())
    e_fn = int((~escalated & label).sum())
    cp, cr, cf = _prf(c_tp, c_fp, c_fn)
    ep, er, ef = _prf(e_tp, e_fp, e_fn)
    items = int(escalated.sum())
    return {
        "n": n,
        "n_attack": int(label.sum()),
        "days": days,
        "coverage": float(auto.mean()) if n else float("nan"),
        "n_auto_attack": int(flag.sum()),
        "n_manual": int(manual.sum()),
        "n_manual_low_confidence": int((codes == CODE_LOW_CONFIDENCE).sum()),
        "n_manual_not_in_training": int((codes == CODE_NOT_IN_TRAINING).sum()),
        "n_manual_too_far": int((codes == CODE_TOO_FAR).sum()),
        "n_attack_manual": int((manual & label).sum()),
        "covered_precision": cp,
        "covered_recall": cr,
        "covered_f1": cf,
        "e2e_tp": e_tp,
        "e2e_fp": e_fp,
        "missed_attacks": e_fn,
        "e2e_precision": ep,
        "e2e_recall": er,
        "e2e_f1": ef,
        "analyst_items": items,
        "analyst_items_per_day": items / days if days > 0 else float("nan"),
        "fp_per_day": e_fp / days if days > 0 else float("nan"),
        "workload_reduction": 1.0 - items / n if n else float("nan"),
    }


def _days(ts: np.ndarray) -> float:
    return float(ts.max() - ts.min()) / 86400.0 if len(ts) > 1 else 0.0


def horizon_bounds(n_test: int, gran: float) -> list[tuple[int, int, int]]:
    """(h, start, end) over the test rows, h = 1..n_windows -- the same
    windows as temporal_decay's "baseline_split" WindowScheme.target_bounds
    (minus the split offset)."""
    n_windows = compute_window_bounds(n_test, gran, 0)[2]
    out = []
    for h in range(1, n_windows + 1):
        s, e, _ = compute_window_bounds(n_test, gran, h - 1)
        out.append((h, s, e))
    return out


def horizon_metric_rows(
    label: np.ndarray,
    codes: np.ndarray,
    ts: np.ndarray,
    seen: np.ndarray,
    bounds: list[tuple[int, int, int]],
) -> list[dict]:
    rows = []
    windows = [(h, s, e) for h, s, e in bounds] + [(0, 0, len(label))]
    for h, s, e in windows:
        sl = slice(s, e)
        days = _days(ts[sl])
        for subset, m in (
            ("all", np.ones(e - s, dtype=bool)),
            ("seen", seen[sl]),
            ("unseen", ~seen[sl]),
        ):
            row = decision_metrics(label[sl][m], codes[sl][m], days)
            rows.append({"horizon": h if h else "all", "subset": subset, **row})
    return rows


# --- Tuning on W0 -------------------------------------------------------------


def _interpret_cell(
    att_tr: Attended,
    att_te: Attended,
    label_tr: np.ndarray,
    vocab: Vocab,
    threshold: float,
    eps: float,
    cfg: DeepCaseConfig,
    rng: np.random.Generator,
    cached_vectors: dict | None = None,
    return_test_clusters: bool = False,
):
    """Cluster + simulated analyst on the training rows, predict the test
    rows. Returns (test codes, analyst result, training clusters), plus the
    matched training cluster per test row if return_test_clusters."""
    if cached_vectors is not None and threshold in cached_vectors:
        (vec_tr, m_tr), (vec_te, m_te) = cached_vectors[threshold]
    else:
        vec_tr, m_tr = vectorize(att_tr, threshold, vocab.size)
        vec_te, m_te = vectorize(att_te, threshold, vocab.size)
        if cached_vectors is not None:
            cached_vectors.clear()
            cached_vectors[threshold] = ((vec_tr, m_tr), (vec_te, m_te))
    clusters = cluster_rows(vec_tr, m_tr, att_tr.y, eps, cfg.min_samples)
    analyst = simulated_analyst(clusters, m_tr, label_tr, cfg.samples_per_cluster, rng)
    fitted = fit_interpreter(
        vec_tr, m_tr, att_tr.y, clusters, analyst.cluster_label, eps
    )
    codes, test_clusters = predict_codes_and_clusters(
        fitted, vec_te, m_te, att_te.y, vocab.unknown_id
    )
    if return_test_clusters:
        return codes, analyst, clusters, test_clusters
    return codes, analyst, clusters


def tune_on_w0(
    ev: CscasEvents, cfg: DeepCaseConfig, device: torch.device, out_dir: Path
) -> tuple[dict, pd.DataFrame]:
    """Paper App. D on W0 only: first half (by time) trains the
    ContextBuilder and the simulated-analyst clusters, second half is scored.
    Grid: context length x tau_confidence x eps. Every cell is written to
    tuning.csv (rewritten after each context length)."""
    w0 = ev.split_idx
    half = w0 // 2
    vocab = Vocab.from_signatures(ev.signature[:half])
    ids = vocab.encode(ev.signature[:w0])
    label_tr, label_te = ev.label[:half], ev.label[half:w0]
    days_te = _days(ev.ts[half:w0])
    epochs = cfg.tune_epochs or cfg.epochs
    rows: list[dict] = []
    for L in cfg.tune_context_lengths:
        t0 = time.time()
        ctx = build_context(
            ids, ev.machine[:w0], ev.ts[:w0], L, cfg.timeout, vocab.no_event_id
        )
        train_id = _train_id("w0tune", ev, half, vocab, cfg)
        cb = train_context_builder(
            ctx[:half],
            ids[:half],
            vocab,
            train_id,
            cfg,
            L,
            epochs,
            cfg.tune_seed,
            device,
        )
        att_tr = attention_query(
            cb, vocab, ctx[:half], ids[:half], cfg, device, cfg.tune_seed
        )
        att_te = attention_query(
            cb, vocab, ctx[half:], ids[half:], cfg, device, cfg.tune_seed
        )
        print(f"  [tune] L={L}: ContextBuilder + attention {time.time() - t0:.0f}s")
        cache: dict = {}
        for tau in cfg.tune_thresholds:
            for eps in cfg.tune_eps:
                rng = np.random.default_rng(cfg.tune_seed)
                codes, analyst, _ = _interpret_cell(
                    att_tr, att_te, label_tr, vocab, tau, eps, cfg, rng, cache
                )
                m = decision_metrics(label_te, codes, days_te)
                rows.append(
                    {
                        "context_length": L,
                        "threshold": tau,
                        "eps": eps,
                        **{f"train_{k}": v for k, v in analyst.stats.items()},
                        **m,
                    }
                )
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "tuning.csv", index=False)
        best_L = (
            table[table.context_length == L].sort_values(cfg.tune_objective).iloc[-1]
        )
        print(
            f"  [tune] L={L} best: tau={best_L.threshold} eps={best_L.eps} "
            f"{cfg.tune_objective}={best_L[cfg.tune_objective]:.3f} "
            f"coverage={best_L.coverage:.3f} ({time.time() - t0:.0f}s)"
        )
    table = pd.DataFrame(rows)
    best = pick_best(table, cfg.tune_objective)
    return best, table


def pick_best(table: pd.DataFrame, objective: str) -> dict:
    """Highest objective; ties -> higher coverage -> closest to the paper's
    defaults (L 10, tau 0.2, eps 0.1)."""
    t = table.copy()
    t["_obj"] = t[objective].fillna(-1.0).round(6)
    t["_dist"] = (
        (t.context_length - 10).abs() / 10
        + (t.threshold - 0.2).abs()
        + (t.eps - 0.1).abs()
    )
    t = t.sort_values(["_obj", "coverage", "_dist"], ascending=[False, False, True])
    r = t.iloc[0]
    return {
        "context_length": int(r.context_length),
        "threshold": float(r.threshold),
        "eps": float(r.eps),
        "objective": objective,
        "objective_value": float(r[objective]),
        "coverage": float(r.coverage),
    }


# --- Main run -------------------------------------------------------------------


def _train_id(
    tag: str, ev: CscasEvents, n_rows: int, vocab: Vocab, cfg: DeepCaseConfig
) -> str:
    """Cache name for the ContextBuilder: encodes the training slice (start
    timestamp + row count), vocabulary size and host rule, so a smoke-test
    slice never reuses a full-data model or vice versa."""
    return (
        f"cscas_{tag}_t{int(ev.ts[0])}_n{n_rows}_v{vocab.size}_host{cfg.missing_host}"
    )


def _summarise(metrics: pd.DataFrame, budget: pd.DataFrame) -> str:
    cols = [
        "e2e_f1",
        "e2e_precision",
        "e2e_recall",
        "covered_f1",
        "coverage",
        "missed_attacks",
        "analyst_items_per_day",
        "fp_per_day",
        "workload_reduction",
    ]
    lines = ["DeepCASE on CSCAS (frozen) -- mean ± std over seeds", ""]
    allrow = metrics[metrics.horizon.astype(str) == "all"]
    for subset in ("all", "seen", "unseen"):
        sub = allrow[allrow.subset == subset]
        lines.append(
            f"[test, {subset} signatures]  n_attack={int(sub.n_attack.iloc[0])}"
        )
        for c in cols:
            lines.append(f"  {c:<24}{sub[c].mean():10.4f} ± {sub[c].std(ddof=0):.4f}")
        lines.append("")
    lines.append("[training: simulated analyst]")
    for c in (
        "budget_b0",
        "budget_clusters_only",
        "n_clusters",
        "n_attack_clusters",
        "n_mixed_clusters",
        "n_outliers",
        "manual_workload_reduction",
    ):
        lines.append(f"  {c:<28}{budget[c].mean():12.1f} ± {budget[c].std(ddof=0):.1f}")
    lines.append("")
    per_h = (
        metrics[(metrics.subset == "all") & (metrics.horizon.astype(str) != "all")]
        .groupby("horizon")[
            ["e2e_f1", "e2e_recall", "coverage", "analyst_items_per_day"]
        ]
        .mean()
    )
    lines.append("[per horizon, all signatures, mean over seeds]")
    lines.append(per_h.to_string(float_format=lambda v: f"{v:.4f}"))
    return "\n".join(lines) + "\n"


def run_deepcase_comparison(cfg: DeepCaseConfig) -> Path:
    device = resolve_device(cfg.device)
    out_dir = (
        Path(cfg.out_dir)
        if cfg.out_dir
        else DEFAULT_OUT_ROOT
        / "cscas"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput → {out_dir}")

    t0 = time.time()
    ev = load_cscas_events(cfg.csv_path, cfg.split_time, cfg.missing_host)
    full = cfg.max_train_rows is None and cfg.max_test_rows is None
    if full:
        assert ev.split_idx == EXPECTED_TRAIN_ROWS, ev.split_idx
        assert len(ev) - ev.split_idx == EXPECTED_TEST_ROWS, len(ev) - ev.split_idx
    ev = truncate_events(ev, cfg.max_train_rows, cfg.max_test_rows)
    n_test = len(ev) - ev.split_idx
    print(
        f"Loaded {len(ev)} rows in {time.time() - t0:.0f}s: W0 {ev.split_idx} "
        f"({int(ev.label[: ev.split_idx].sum())} attack), test {n_test} "
        f"({int(ev.label[ev.split_idx :].sum())} attack); "
        f"{int(pd.isna(ev.machine).sum())} rows without a host"
    )

    run_cfg = asdict(cfg)
    if cfg.tune:
        print("Tuning on W0 ...")
        best, _ = tune_on_w0(ev, cfg, device, out_dir)
        print(f"Tuned: {best}")
        run_cfg["tuned"] = best
        L, tau, eps = best["context_length"], best["threshold"], best["eps"]
    else:
        L, tau, eps = cfg.context_length, cfg.threshold, cfg.eps
    run_cfg["used"] = {"context_length": L, "threshold": tau, "eps": eps}
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str))

    s = ev.split_idx
    vocab = Vocab.from_signatures(ev.signature[:s])
    ids = vocab.encode(ev.signature)
    ctx = build_context(ids, ev.machine, ev.ts, L, cfg.timeout, vocab.no_event_id)
    seen = np.isin(ev.signature[s:], vocab.signatures)
    label_tr, label_te, ts_te = ev.label[:s], ev.label[s:], ev.ts[s:]
    bounds = horizon_bounds(n_test, cfg.granularity)
    horizon_of = np.empty(n_test, dtype=np.int16)
    for h, a, b in bounds:
        horizon_of[a:b] = h
    pd.DataFrame(
        [
            {
                "horizon": h,
                "start": a,
                "end": b,
                "n": b - a,
                "t_start": ts_te[a],
                "t_end": ts_te[b - 1],
                "days": _days(ts_te[a:b]),
            }
            for h, a, b in bounds
        ]
    ).to_csv(out_dir / "windows.csv", index=False)
    print(
        f"Vocab: {vocab.size} ids ({len(vocab.signatures)} W0 signatures); "
        f"{int((~seen).sum())} test rows / {int(label_te[~seen].sum())} test attacks "
        f"on unseen signatures; {len(bounds)} horizons at gran {cfg.granularity}"
    )

    metric_rows: list[dict] = []
    budget_rows: list[dict] = []
    for seed in cfg.seeds:
        ts0 = time.time()
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        cb = train_context_builder(
            ctx[:s],
            ids[:s],
            vocab,
            _train_id("w0", ev, s, vocab, cfg),
            cfg,
            L,
            cfg.epochs,
            seed,
            device,
        )
        att_tr = attention_query(cb, vocab, ctx[:s], ids[:s], cfg, device, seed)
        att_te = attention_query(cb, vocab, ctx[s:], ids[s:], cfg, device, seed)
        t_query = time.time() - ts0
        rng = np.random.default_rng(seed)
        codes, analyst, clusters, test_clusters = _interpret_cell(
            att_tr,
            att_te,
            label_tr,
            vocab,
            tau,
            eps,
            cfg,
            rng,
            return_test_clusters=True,
        )

        pd.DataFrame(
            {
                "test_row": np.arange(n_test),
                "group_id": ev.group_id[s:],
                "ts": ts_te,
                "horizon": horizon_of,
                "label": label_te,
                "signature_id": ev.signature[s:],
                "seen": seen,
                "confidence": att_te.confidence,
                "code": codes,
                # training cluster that decided an automatic code (-1 = none)
                "cluster": test_clusters,
            }
        ).to_parquet(seed_dir / "test_predictions.parquet", index=False)
        inspected = np.concatenate([analyst.sampled_idx, analyst.outlier_idx])
        pd.DataFrame(
            {
                "train_row": inspected,
                "group_id": ev.group_id[inspected],
                "kind": ["cluster_sample"] * len(analyst.sampled_idx)
                + ["outlier"] * len(analyst.outlier_idx),
                "cluster": clusters[inspected],
                "label": label_tr[inspected],
            }
        ).to_parquet(seed_dir / "analyst_items.parquet", index=False)
        cl = pd.DataFrame(
            {"cluster": clusters, "label": label_tr, "signature_id": ev.signature[:s]}
        )
        cl = (
            cl[cl.cluster != -1]
            .groupby("cluster")
            .agg(
                signature_id=("signature_id", "first"),
                size=("label", "size"),
                attack_frac=("label", "mean"),
            )
        )
        cl["assigned_label"] = pd.Series(analyst.cluster_label)
        cl.to_csv(seed_dir / "train_clusters.csv")

        for r in horizon_metric_rows(label_te, codes, ts_te, seen, bounds):
            metric_rows.append({"seed": seed, **r})
        budget_rows.append(
            {
                "seed": seed,
                **analyst.stats,
                "seconds_query": t_query,
                "seconds_total": time.time() - ts0,
            }
        )
        pd.DataFrame(metric_rows).to_csv(out_dir / "horizon_metrics.csv", index=False)
        pd.DataFrame(budget_rows).to_csv(out_dir / "budget.csv", index=False)
        last = [
            r
            for r in metric_rows
            if r["seed"] == seed and r["horizon"] == "all" and r["subset"] == "all"
        ][0]
        print(
            f"  seed {seed}: B0={analyst.stats['budget_b0']} "
            f"({analyst.stats['n_clusters']} clusters, {analyst.stats['n_outliers']} outliers) | "
            f"test e2e F1={last['e2e_f1']:.3f} R={last['e2e_recall']:.3f} "
            f"coverage={last['coverage']:.3f} items/day={last['analyst_items_per_day']:.0f} "
            f"({time.time() - ts0:.0f}s)"
        )

    summary = _summarise(pd.DataFrame(metric_rows), pd.DataFrame(budget_rows))
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"Saved → {out_dir}")
    return out_dir


__all__ = [
    "DeepCaseConfig",
    "CscasEvents",
    "Vocab",
    "build_context",
    "load_cscas_events",
    "run_deepcase_comparison",
    "simulated_analyst",
    "decision_metrics",
    "horizon_bounds",
]
