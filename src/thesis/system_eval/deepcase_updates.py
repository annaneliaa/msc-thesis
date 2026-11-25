"""
DeepCASE kept up to date over the CSCAS test windows (DeepCASE implementation
plan B6 / comparison plan step 5.6): the same three policies and label
accounting as our system's monitor_attached budget mode.

Starting point = the frozen step-5.2 setup for each seed (same tuned
context length / tau / eps, read from that run's config.json; same cached
W0 ContextBuilder; same W0 manual mode), so `policy="never"` reproduces the
5.2 numbers.

Policies (per test window h, after scoring it):
* never: frozen;
* schedule: update every window with `schedule_action` (default
  REMINE_AND_RETRAIN = DeepCASE's own Table V update);
* combined: our monitor (thesis.system_eval.deepcase_monitor signals, now
  acted on): cluster precision or coverage drop -> REMINE_AND_RETRAIN, else
  feedback precision drop -> RETRAIN_ONLY. The reference is the first window
  after the last update (warm-up), as for our system.

Update actions (DeepCASE's own update procedure, plan §4.1):
* RETRAIN_ONLY ("add clusters"): keep the ContextBuilder; cluster the events
  DeepCASE could not decide since the last update (confident, known event
  type -- code -3 / DBSCAN noise) and label the new clusters with the
  simulated analyst (10 samples per cluster, highest label). Those events
  were in the manual queue, so their labels are already known: no extra
  labels unless a sample was never reviewed.
* REMINE_AND_RETRAIN ("retrain + recluster"): retrain the ContextBuilder on
  W0 + every window so far (self-supervised, no labels; vocabulary = every
  SignatureID seen so far), recompute attention, and redo manual mode over
  that data. Inspected rows (cluster samples + outliers) without a known
  label are extra update labels.

Label accounting (same rule as label_budget for our system):
* budget = B0 (W0 manual mode) + audit labels + extra update labels;
* workload = analyst items per window (auto-attack + manual queue), whose
  labels are free;
* the audit (`audit_rate` of the auto-benign events, same group_id hash as
  ours) runs under `audit="auto"` only for the combined policy, or is
  forced on/off.

The per-window loop works on an `AttentionProvider` (ContextBuilder +
attention query per training slice) so tests can swap in a fake one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from thesis.monitor.feedback import feedback_signal
from thesis.monitor.pattern_validity import AUDIT_RATE, audit_mask_ids
from thesis.system_eval.deepcase_comparison import (
    Attended,
    CscasEvents,
    DeepCaseConfig,
    Vocab,
    _days,
    cluster_rows,
    decision_metrics,
    fit_interpreter,
    horizon_bounds,
    predict_codes_and_clusters,
    simulated_analyst,
    vectorize,
)
from thesis.system_eval.deepcase_monitor import (
    MIN_SAMPLES,
    _feedback,
    compare_windows,
    label_weights,
    window_stats,
)

POLICIES = ("never", "schedule", "combined")
ACTIONS = ("RETRAIN_ONLY", "REMINE_AND_RETRAIN")


@dataclass
class UpdateConfig:
    base_run: str  # a deepcase_comparison run dir (tuned params, seeds)
    policy: str = "never"
    schedule_action: str = "REMINE_AND_RETRAIN"
    audit: str = "auto"  # auto | on | off
    audit_rate: float = AUDIT_RATE
    seeds: tuple[int, ...] | None = None  # None = the base run's seeds
    update_epochs: int | None = None  # ContextBuilder epochs per retrain; None = base
    monitor_min_samples: int = MIN_SAMPLES
    out_dir: str | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.schedule_action not in ACTIONS:
            raise ValueError(f"schedule_action must be one of {ACTIONS}")
        if self.audit not in ("auto", "on", "off"):
            raise ValueError("audit must be auto|on|off")

    @property
    def audit_on(self) -> bool:
        return self.audit == "on" or (
            self.audit == "auto" and self.policy == "combined"
        )


# Builds a ContextBuilder on rows [0, n_train) of the events with `vocab`,
# then returns the attention query over ALL rows. (n_train, vocab, tag) ->
# Attended over every row.
AttentionProvider = Callable[[int, Vocab, str], Attended]


@dataclass
class Knowledge:
    """DeepCASE's labelled clusters, in the current attention space: the
    clustered rows (global indices), their cluster ids and cluster labels."""

    rows: np.ndarray
    clusters: np.ndarray
    cluster_label: dict[int, int]


@dataclass
class Space:
    """One ContextBuilder's view of all rows."""

    vocab: Vocab
    ids: np.ndarray  # event id per row
    vec: object  # csr over the confident rows
    mask: np.ndarray  # confident rows
    row_to_vec: np.ndarray  # row -> index in vec (-1 if not confident)

    @classmethod
    def build(cls, att: Attended, vocab: Vocab, tau: float) -> "Space":
        vec, mask = vectorize(att, tau, vocab.size)
        r2v = np.full(len(mask), -1, dtype=np.int64)
        r2v[mask] = np.arange(int(mask.sum()))
        return cls(vocab, att.y, vec, mask, r2v)

    def vec_of(self, rows: np.ndarray):
        return self.vec[self.row_to_vec[rows[self.mask[rows]]]]


@dataclass
class SeedState:
    space: Space
    knowledge: Knowledge
    fitted: object = None
    labelled: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def refit(self, eps: float) -> None:
        r = self.knowledge.rows
        self.fitted = fit_interpreter(
            self.space.vec[self.space.row_to_vec[r]],
            np.ones(len(r), dtype=bool),
            self.space.ids[r],
            self.knowledge.clusters,
            self.knowledge.cluster_label,
            eps,
        )

    def predict(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp_ = self.space
        return predict_codes_and_clusters(
            self.fitted,
            sp_.vec_of(rows),
            sp_.mask[rows],
            sp_.ids[rows],
            sp_.vocab.unknown_id,
        )


def manual_mode(
    space: Space,
    rows: np.ndarray,
    label: np.ndarray,
    eps: float,
    min_samples: int,
    k: int,
    rng: np.random.Generator,
    id_offset: int = 0,
) -> tuple[Knowledge, np.ndarray]:
    """DeepCASE manual mode on `rows`: cluster the confident ones, the
    simulated analyst labels each cluster from k samples and inspects every
    outlier. Returns the labelled clusters and the inspected rows."""
    m = space.mask[rows]
    clusters = cluster_rows(space.vec_of(rows), m, space.ids[rows], eps, min_samples)
    analyst = simulated_analyst(clusters, m, label[rows], k, rng)
    keep = clusters != -1
    ids = clusters[keep] + id_offset
    kn = Knowledge(
        rows=rows[keep],
        clusters=ids,
        cluster_label={c + id_offset: lab for c, lab in analyst.cluster_label.items()},
    )
    inspected = rows[np.concatenate([analyst.sampled_idx, analyst.outlier_idx])]
    return kn, inspected


def _subset_metrics(label, codes, seen, days) -> dict:
    out = {}
    for subset, m in (
        ("all", np.ones(len(label), bool)),
        ("seen", seen),
        ("unseen", ~seen),
    ):
        d = decision_metrics(label[m], codes[m], days)
        out.update({f"{subset}_{k}": v for k, v in d.items()})
    return out


def run_seed(
    ev: CscasEvents,
    provider: AttentionProvider,
    params: dict,
    cfg: UpdateConfig,
    k: int,
    min_cluster: int,
    gran: float,
    seed: int,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Walk the test windows for one seed. Returns (per-window rows, update
    events, per-row decisions)."""
    s = ev.split_idx
    tau, eps = params["threshold"], params["eps"]
    label = ev.label.astype(int)
    n_test = len(ev) - s
    bounds = [(h, s + a, s + b) for h, a, b in horizon_bounds(n_test, gran)]
    rng = np.random.default_rng(seed)
    audited_all = audit_mask_ids(ev.group_id, cfg.audit_rate) > 0

    vocab = Vocab.from_signatures(ev.signature[:s])
    space = Space.build(provider(s, vocab, "w0"), vocab, tau)
    kn, inspected = manual_mode(space, np.arange(s), label, eps, min_cluster, k, rng)
    st = SeedState(space, kn, labelled=np.zeros(len(ev), dtype=bool))
    st.labelled[inspected] = True
    st.refit(eps)
    seen_sig = vocab.signatures  # seen = in W0, always (as in B4)
    n_b0 = len(inspected)
    budget = {"b0": n_b0, "audit": 0, "update": 0, "workload": 0}

    rows_out, events, per_row = [], [], []
    ref_stats, ref_fb = None, None
    pv_streak = fb_streak = 0
    since = bounds[0][1]  # first test row not yet covered by an update
    for h, a, b in bounds:
        rows = np.arange(a, b)
        codes, clus = st.predict(rows)
        lab = label[rows]
        seen = np.isin(ev.signature[rows], seen_sig)
        days = _days(ev.ts[rows])
        analyst = codes != 0
        audited = (
            audited_all[rows] & (codes == 0)
            if cfg.audit_on
            else np.zeros(len(rows), bool)
        )
        st.labelled[rows[analyst | audited]] = True
        n_audit = int(audited.sum())
        budget["audit"] += n_audit
        budget["workload"] += int(analyst.sum())

        # Monitor (always logged; acted on only by the combined policy).
        w = label_weights(codes, audited, "audit", cfg.audit_rate)
        stats = window_stats(lab, codes, clus, w)
        fb = _feedback(lab, analyst)
        is_ref = ref_stats is None
        if is_ref:
            ref_stats, ref_fb = stats, fb
        sig = compare_windows(stats, ref_stats, cfg.monitor_min_samples)
        sig.update(feedback_signal(fb, ref_fb))
        pv_streak = pv_streak + 1 if (sig["elevated"] and not is_ref) else 0
        fb_streak = (
            fb_streak + 1 if (sig.get("fb_precision_elevated") and not is_ref) else 0
        )

        if cfg.policy == "schedule":
            action = cfg.schedule_action
        elif cfg.policy == "combined" and pv_streak:
            action = "REMINE_AND_RETRAIN"
        elif cfg.policy == "combined" and fb_streak:
            action = "RETRAIN_ONLY"
        else:
            action = "NO_ACTION"

        row = {
            "seed": seed,
            "policy": cfg.policy,
            "horizon": h,
            "signals_reference_window": is_ref,
            "n_clusters": len(st.knowledge.cluster_label),
            "action": action,
            "n_audit_labels": n_audit,
            "budget_b0": budget["b0"],
            "budget_audit": budget["audit"],
            "budget_update": budget["update"],
            "budget_labels": budget["b0"] + budget["audit"] + budget["update"],
            "workload_labels": budget["workload"],
            **{f"mon_{kk}": v for kk, v in sig.items()},
            **_subset_metrics(lab, codes, seen, days),
        }
        rows_out.append(row)
        per_row.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "horizon": h,
                    "group_id": ev.group_id[rows],
                    "code": codes.astype(np.int8),
                    "cluster": clus.astype(np.int32),
                }
            )
        )

        if action == "NO_ACTION":
            continue
        t0 = time.time()
        upd_rows = np.arange(since, b)
        if action == "RETRAIN_ONLY":
            sp_ = st.space
            cand = upd_rows[
                sp_.mask[upd_rows]
                & (sp_.ids[upd_rows] != sp_.vocab.unknown_id)
                & ~np.isin(upd_rows, st.knowledge.rows)
            ]
            # Undecided events only: those not matched by an existing cluster.
            if len(cand):
                c_codes, _ = st.predict(cand)
                cand = cand[c_codes < 0]
            offset = (
                (max(st.knowledge.cluster_label) + 1)
                if st.knowledge.cluster_label
                else 0
            )
            new_kn, insp = manual_mode(
                st.space, cand, label, eps, min_cluster, k, rng, id_offset=offset
            )
            extra = int((~st.labelled[insp]).sum())
            st.labelled[insp] = True
            st.knowledge = Knowledge(
                np.concatenate([st.knowledge.rows, new_kn.rows]),
                np.concatenate([st.knowledge.clusters, new_kn.clusters]),
                {**st.knowledge.cluster_label, **new_kn.cluster_label},
            )
            n_new = len(new_kn.cluster_label)
        else:
            data = np.arange(0, b)
            vocab = Vocab.from_signatures(ev.signature[:b])
            space = Space.build(provider(b, vocab, f"upd_h{h}"), vocab, tau)
            new_kn, insp = manual_mode(space, data, label, eps, min_cluster, k, rng)
            extra = int((~st.labelled[insp]).sum())
            st.labelled[insp] = True
            n_new = len(new_kn.cluster_label) - len(st.knowledge.cluster_label)
            st.space, st.knowledge = space, new_kn
        st.refit(eps)
        budget["update"] += extra
        events.append(
            {
                "seed": seed,
                "policy": cfg.policy,
                "horizon": h,
                "action": action,
                "n_update_rows": len(upd_rows),
                "n_inspected": len(insp),
                "extra_labels": extra,
                "n_clusters_after": len(st.knowledge.cluster_label),
                "n_new_clusters": n_new,
                "seconds": time.time() - t0,
            }
        )
        rows_out[-1]["budget_update"] = budget["update"]
        rows_out[-1]["budget_labels"] = (
            budget["b0"] + budget["audit"] + budget["update"]
        )
        since = b
        ref_stats = ref_fb = None  # next window = new reference (warm-up)
        pv_streak = fb_streak = 0
    return rows_out, events, pd.concat(per_row, ignore_index=True)


def cb_attention_provider(
    ev: CscasEvents, dc: DeepCaseConfig, L: int, epochs: int, seed: int, device
):
    """The real provider: build contexts with the vocabulary, train (or load
    from the cache) a ContextBuilder on rows [0, n_train), query attention
    for every row."""
    from thesis.system_eval.deepcase_comparison import (
        _train_id,
        attention_query,
        build_context,
        train_context_builder,
    )

    def provide(n_train: int, vocab: Vocab, tag: str) -> Attended:
        ids = vocab.encode(ev.signature)
        ctx = build_context(ids, ev.machine, ev.ts, L, dc.timeout, vocab.no_event_id)
        ep = dc.epochs if tag == "w0" else epochs
        cb = train_context_builder(
            ctx[:n_train],
            ids[:n_train],
            vocab,
            _train_id(tag, ev, n_train, vocab, dc),
            dc,
            L,
            ep,
            seed,
            device,
        )
        # Two queries (training rows, then the rest), exactly as step 5.2
        # does: the attention query is a seeded per-batch optimisation, so
        # this keeps policy="never" identical to the 5.2 run.
        a = attention_query(cb, vocab, ctx[:n_train], ids[:n_train], dc, device, seed)
        b = attention_query(cb, vocab, ctx[n_train:], ids[n_train:], dc, device, seed)
        return Attended(
            X=np.concatenate([a.X, b.X]),
            y=np.concatenate([a.y, b.y]),
            confidence=np.concatenate([a.confidence, b.confidence]),
            attention=np.concatenate([a.attention, b.attention]),
        )

    return provide


def run_deepcase_updates(cfg: UpdateConfig) -> Path:
    from thesis.grouping._device import resolve_device
    from thesis.system_eval.deepcase_comparison import (
        DEFAULT_OUT_ROOT,
        load_cscas_events,
        truncate_events,
    )

    base = Path(cfg.base_run)
    run_cfg = json.loads((base / "config.json").read_text())
    used = run_cfg["used"]
    dc = DeepCaseConfig(
        **{
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in run_cfg.items()
            if k in DeepCaseConfig.__dataclass_fields__
        }
    )
    seeds = cfg.seeds or dc.seeds
    out_dir = (
        Path(cfg.out_dir)
        if cfg.out_dir
        else (
            DEFAULT_OUT_ROOT.parent
            / "deepcase_updates"
            / "cscas"
            / f"{cfg.policy}_audit-{cfg.audit}"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"update": asdict(cfg), "base_used": used}, indent=2, default=str)
    )
    device = resolve_device(dc.device)
    ev = load_cscas_events(dc.csv_path, dc.split_time, dc.missing_host)
    ev = truncate_events(ev, dc.max_train_rows, dc.max_test_rows)
    print(
        f"DeepCASE updates: policy={cfg.policy} audit={cfg.audit} seeds={seeds} "
        f"used={used} → {out_dir}"
    )
    all_rows, all_events = [], []
    for seed in seeds:
        t0 = time.time()
        provider = cb_attention_provider(
            ev, dc, used["context_length"], cfg.update_epochs or dc.epochs, seed, device
        )
        rows, events, per_row = run_seed(
            ev,
            provider,
            used,
            cfg,
            dc.samples_per_cluster,
            dc.min_samples,
            dc.granularity,
            seed,
        )
        all_rows += rows
        all_events += events
        per_row.to_parquet(out_dir / f"decisions_seed{seed}.parquet", index=False)
        pd.DataFrame(all_rows).to_csv(out_dir / "per_window.csv", index=False)
        pd.DataFrame(all_events).to_csv(out_dir / "events.csv", index=False)
        last = rows[-1]
        print(
            f"  seed {seed}: {len(events)} updates, budget {last['budget_labels']}, "
            f"mean e2e F1 {np.mean([r['all_e2e_f1'] for r in rows]):.3f} "
            f"({time.time() - t0:.0f}s)"
        )
    print(f"Saved → {out_dir}")
    return out_dir
