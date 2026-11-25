"""
Synthetic drift test for the monitor on DeepCASE (DeepCASE implementation
plan B5, follow-up): can the monitor's signals fire on DeepCASE at all?

On real CSCAS frozen DeepCASE does not decay, and most of our signals stay
quiet on it. That is only informative if they *could* fire. So, as for our
system (system_eval.synthetic_drift), we inject controlled drift into the
test events from window `inject_from_h` on, keep every label unchanged, run
frozen DeepCASE (cached W0 ContextBuilder, the 5.2 run's tuned settings) and
log every monitor signal per window: the mapped ones (feedback precision,
per-cluster precision, coverage; escalated + audit labels) and the
DeepCASE-native ones (manual-queue reason shares). False alarms = windows
2..inject_from_h-1 (window 1 is the reference); detection = windows
inject_from_h..N.

DeepCASE only sees an event's signature and its host's earlier events, so
the injections change those (labels never change):

* attack_mimics_benign: an attack event takes the signature AND host of a
  random W0 benign event, so its context looks benign too (harmful: attacks
  auto-labelled benign);
* benign_mimics_attack: a benign event takes an attack event's signature and
  host (harmful: false alarms);
* new_signature: an event gets one of a few never-seen signature ids
  (unknown event type -> manual queue);
* context_shift: an event gets a random host from the W0 pool, so its context
  sequence is unfamiliar (low confidence -> manual queue);
* benign_mix_shift: a benign event takes the signature and host of a
  minority benign signature's event (meant to be harmless).
Changing an event's host also changes the context of that host's later
events -- a side effect DeepCASE's sequencing makes unavoidable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.system_eval.deepcase_comparison import (
    CscasEvents,
    DeepCaseConfig,
    horizon_bounds,
)
from thesis.system_eval.deepcase_monitor import monitor_seed
from thesis.system_eval.deepcase_updates import UpdateConfig, run_seed

KINDS = (
    "none",
    "attack_mimics_benign",
    "benign_mimics_attack",
    "new_signature",
    "context_shift",
    "benign_mix_shift",
)
DEFAULT_FRACTIONS = {
    "attack_mimics_benign": 0.5,
    "benign_mimics_attack": 0.02,
    "new_signature": 0.3,
    "context_shift": 0.3,
    "benign_mix_shift": 0.5,
}
_TARGET = {
    "attack_mimics_benign": "attack",
    "benign_mimics_attack": "benign",
    "new_signature": "all",
    "context_shift": "all",
    "benign_mix_shift": "benign",
}
N_NEW_SIGNATURES = 5
SIGNAL_FLAGS = {
    "feedback_precision": "fb_precision_elevated",
    "cluster_precision": "pv_audit_precision_elevated",
    "coverage": "pv_audit_coverage_elevated",
    "pattern_validity": "pv_audit_elevated",
    "native_low_confidence": "native_low_confidence_elevated",
    "native_unknown_event": "native_unknown_event_elevated",
    "native_too_far": "native_too_far_elevated",
    "native_any": "native_elevated",
}


@dataclass(frozen=True, slots=True)
class Injection:
    kind: str
    fraction: float

    @property
    def name(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}@{self.fraction:g}"


def default_injections() -> list[Injection]:
    return [Injection("none", 0.0)] + [
        Injection(k, f) for k, f in DEFAULT_FRACTIONS.items()
    ]


def _donor_rows(ev: CscasEvents, cls: int, minority: bool = False) -> np.ndarray:
    """W0 rows of class `cls` with a host (so a donor brings a context). With
    minority=True: only the rows of the least frequent such signature with at
    least 20 events."""
    s = ev.split_idx
    has_host = np.array([m is not None for m in ev.machine[:s]])
    rows = np.flatnonzero((ev.label[:s] == cls) & has_host)
    if minority and len(rows):
        sig, cnt = np.unique(ev.signature[rows], return_counts=True)
        ok = cnt >= 20
        pick = sig[ok][np.argmin(cnt[ok])] if ok.any() else sig[np.argmin(cnt)]
        rows = rows[ev.signature[rows] == pick]
    return rows


def inject(
    ev: CscasEvents, injection: Injection, start_row: int, seed: int = 0
) -> CscasEvents:
    """Copy of `ev` with rows >= start_row perturbed per `injection`; labels,
    timestamps, group ids and every earlier row unchanged."""
    if injection.kind == "none":
        return ev
    if injection.kind not in DEFAULT_FRACTIONS:
        raise ValueError(f"unknown injection kind {injection.kind!r}")
    rng = np.random.default_rng(seed)
    sig = ev.signature.copy()
    mach = ev.machine.copy()
    rows = np.arange(start_row, len(ev))
    target = _TARGET[injection.kind]
    if target != "all":
        rows = rows[ev.label[rows] == (1 if target == "attack" else 0)]
    hit = rows[rng.random(len(rows)) < injection.fraction]
    if len(hit) == 0:
        return replace(ev, signature=sig, machine=mach)

    if injection.kind in (
        "attack_mimics_benign",
        "benign_mimics_attack",
        "benign_mix_shift",
    ):
        donors = _donor_rows(
            ev,
            cls=1 if injection.kind == "benign_mimics_attack" else 0,
            minority=injection.kind == "benign_mix_shift",
        )
        d = rng.choice(donors, size=len(hit))
        sig[hit] = ev.signature[d]
        mach[hit] = ev.machine[d]
    elif injection.kind == "new_signature":
        base = int(ev.signature.max()) + 1_000_000
        sig[hit] = base + rng.integers(0, N_NEW_SIGNATURES, len(hit))
    elif injection.kind == "context_shift":
        pool = np.array(
            sorted({m for m in ev.machine[: ev.split_idx] if m is not None}),
            dtype=object,
        )
        mach[hit] = rng.choice(pool, size=len(hit))
    return replace(ev, signature=sig, machine=mach)


def inject_start_row(ev: CscasEvents, gran: float, inject_from_h: int) -> int:
    n_test = len(ev) - ev.split_idx
    bounds = {h: a for h, a, _ in horizon_bounds(n_test, gran)}
    return ev.split_idx + bounds[inject_from_h]


def run_injection(
    ev: CscasEvents,
    provider_factory,
    params: dict,
    injection: Injection,
    inject_from_h: int,
    gran: float,
    k: int,
    min_cluster: int,
    seed: int,
    audit_rate: float,
) -> pd.DataFrame:
    """Frozen DeepCASE on the injected events; per-window metrics + every
    monitor signal. `provider_factory(ev) -> AttentionProvider`."""
    start = inject_start_row(ev, gran, inject_from_h)
    ev_i = inject(ev, injection, start, seed)
    cfg = UpdateConfig(
        base_run="unused", policy="never", audit="on", audit_rate=audit_rate
    )
    rows, _, per_row = run_seed(
        ev_i, provider_factory(ev_i), params, cfg, k, min_cluster, gran, seed
    )
    s = ev.split_idx
    test = pd.DataFrame(
        {
            "group_id": ev_i.group_id[s:],
            "ts": ev_i.ts[s:],
            "label": ev_i.label[s:].astype(int),
        }
    )
    preds = per_row.drop(columns="seed").merge(test, on="group_id", how="left")
    signals = monitor_seed(preds, audit_rate=audit_rate)
    metrics = pd.DataFrame(rows)[
        ["horizon", "all_e2e_f1", "all_e2e_recall", "all_e2e_precision", "all_coverage"]
    ]
    out = signals.merge(metrics, on="horizon", how="left")
    out.insert(0, "seed", seed)
    out.insert(1, "injection", injection.name)
    out.insert(2, "kind", injection.kind)
    out.insert(3, "fraction", injection.fraction)
    out["inject_from_h"] = inject_from_h
    out["injected"] = (injection.kind != "none") & (out.horizon >= inject_from_h)
    return out


def detection_summary(per_window: pd.DataFrame) -> pd.DataFrame:
    """Per (seed, injection): F1 change vs the clean run on the injected
    windows, and per signal the firing rate on clean windows (2..k-1, false
    alarm) and on injected windows (k..N, detection)."""
    walk = per_window[~per_window.is_reference]
    rows = []
    for (seed, name), g in walk.groupby(["seed", "injection"], sort=False):
        k = int(g.inject_from_h.iloc[0])
        pre, post = g[g.horizon < k], g[g.horizon >= k]
        clean = walk[(walk.seed == seed) & (walk.injection == "none")]
        clean_post = clean[clean.horizon >= k]
        r = {
            "seed": seed,
            "injection": name,
            "kind": g.kind.iloc[0],
            "d_f1_vs_clean": post.all_e2e_f1.mean() - clean_post.all_e2e_f1.mean(),
            "d_recall_vs_clean": post.all_e2e_recall.mean()
            - clean_post.all_e2e_recall.mean(),
            "d_coverage_vs_clean": post.all_coverage.mean()
            - clean_post.all_coverage.mean(),
        }
        for label, col in SIGNAL_FLAGS.items():
            if col in g:
                r[f"{label}_false_alarm"] = pre[col].astype(float).mean()
                r[f"{label}_detect"] = post[col].astype(float).mean()
        rows.append(r)
    return pd.DataFrame(rows)


@dataclass
class DeepCaseDriftConfig:
    base_run: str
    seeds: tuple[int, ...] = (0,)
    inject_from_h: int = 6
    audit_rate: float = 0.05
    kinds: tuple[str, ...] | None = None  # None = every default injection
    out_dir: str | None = None


def run_deepcase_synthetic_drift(cfg: DeepCaseDriftConfig) -> Path:
    from thesis.grouping._device import resolve_device
    from thesis.system_eval.deepcase_comparison import (
        DEFAULT_OUT_ROOT,
        load_cscas_events,
        truncate_events,
    )
    from thesis.system_eval.deepcase_updates import cb_attention_provider

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
    out_dir = (
        Path(cfg.out_dir)
        if cfg.out_dir
        else (
            DEFAULT_OUT_ROOT.parent
            / "deepcase_synthetic_drift"
            / "cscas"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"drift": asdict(cfg), "base_used": used}, indent=2, default=str)
    )
    injections = [
        i
        for i in default_injections()
        if cfg.kinds is None or i.kind in cfg.kinds or i.kind == "none"
    ]
    device = resolve_device(dc.device)
    ev = load_cscas_events(dc.csv_path, dc.split_time, dc.missing_host)
    ev = truncate_events(ev, dc.max_train_rows, dc.max_test_rows)
    frames = []
    for seed in cfg.seeds:
        for inj in injections:
            t0 = time.time()
            frames.append(
                run_injection(
                    ev,
                    lambda e: cb_attention_provider(
                        e, dc, used["context_length"], dc.epochs, seed, device
                    ),
                    used,
                    inj,
                    cfg.inject_from_h,
                    dc.granularity,
                    dc.samples_per_cluster,
                    dc.min_samples,
                    seed,
                    cfg.audit_rate,
                )
            )
            per_window = pd.concat(frames, ignore_index=True)
            per_window.to_csv(out_dir / "per_window.csv", index=False)
            print(f"  seed {seed} {inj.name}: done ({time.time() - t0:.0f}s)")
    summ = detection_summary(per_window)
    summ.to_csv(out_dir / "detection_summary.csv", index=False)
    text = format_summary(summ)
    (out_dir / "summary.txt").write_text(text)
    print(text)
    print(f"Saved → {out_dir}")
    return out_dir


def format_summary(summ: pd.DataFrame) -> str:
    cols = ["d_f1_vs_clean", "d_recall_vs_clean", "d_coverage_vs_clean"]
    for label in SIGNAL_FLAGS:
        cols += [f"{label}_false_alarm", f"{label}_detect"]
    cols = [c for c in cols if c in summ]
    mean = summ.groupby("injection", sort=False)[cols].mean()
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        return (
            "Synthetic drift on frozen DeepCASE (mean over seeds)\n"
            "false_alarm = fires on clean windows 2..k-1; detect = fires on injected windows k..N\n\n"
            + mean.round(2).T.to_string()
        )
