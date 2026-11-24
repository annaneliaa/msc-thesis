"""
Synthetic drift test for the monitor (positive control).

On real CSCAS the label-free monitor signals don't track model decay. That
leaves two explanations: the signals are blind, or CSCAS has little harmful
drift for them to catch. This experiment separates the two by injecting
drift we control into the second half of the test period and checking which
signals fire.

Setup: the same observe-only walk as monitor_drift (mine+fit once on W_src,
frozen model/schema/threshold, every signal logged per window), run once on
the clean timeline and once per injection. W_src and the first half of the
walk are never touched, so their windows measure each signal's false-alarm
rate; windows h >= inject_from measure detection. Donor rows come from
W_src's train split (traffic the deployment has already seen).

Injections (fields refer to AlertGroup attributes):

* attack_mimics_benign -- harmful, evasion: a fraction of attack groups copy
  their whole profile (PROFILE_FIELDS: the mined fields + the model's base
  features) from random benign donors. Expect recall to drop. Attacks are ~1% of traffic, so input
  statistics barely move; precision-based feedback can't see missed attacks.
* benign_mimics_attack -- harmful, false positives: a fraction of benign
  groups copy PROFILE_FIELDS from attack donors. Expect precision to drop and
  input statistics to move.
* benign_mix_shift -- harmless covariate shift: a fraction of benign groups
  copy PROFILE_FIELDS from benign donors of one minority benign category, so
  the benign mix changes but stays benign. Input signals should fire, F1
  should not drop.
* unseen_field_shift -- control: fields neither the monitor nor the model
  reads (ports, port-multiplicity flags, qualifiers) are changed. Nothing
  should fire and F1 should not move.
* volume_shift -- a model-only base feature (signature_matches_per_day) is
  scaled x10. The mined predicates don't read it, so Signal 1 and novelty are
  blind by construction; only score-based signals can see it.
* new_values -- unseen category/ruleset values appear (a new product or
  rule feed).

Outputs (artifacts/experiments/synthetic_drift/<scenario>/<ts>/):
per_horizon_results.csv (monitor_drift's columns + injection columns),
monitor_signals.csv, detection_summary.csv, summary.txt, config.json.
"""

from __future__ import annotations

import dataclasses
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from thesis.experiments._shared import load_scenario_context
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.monitor.noise_aware import STATISTICS
from thesis.monitor.pattern_validity import LABEL_MODES as PV_LABEL_MODES
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_train_end
from thesis.schemas.experiments import MonitorDriftConfig
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.monitor_drift import _run_one_monitor_config
from thesis.system_eval.temporal_decay import WindowScheme, _build_window_scheme

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "synthetic_drift"

# AlertGroup attributes behind the mined single predicates / compound rules
# on CSCAS (category, ruleset, proto, scas, cve_present, multi_target,
# attr_populated:*, applicable_layer:*, signature_id_similarity).
MINED_FIELDS = (
    "category",
    "ruleset",
    "proto",
    "scas",
    "cve_refs",
    "int_ip_is_multiple",
    "attr_similarities",
    "signature_id_similarity",
)

# The whole profile the model reads: mined fields + the base features.
PROFILE_FIELDS = MINED_FIELDS + (
    "signature_matches_per_day",
    "similarity",
    "n_alerts",
)

# Per-window statistic values compared with the clean run (post-injection
# mean minus the clean run's mean on the same windows).
SIGNAL_VALUES = (
    *(f"na_{s}" for s in STATISTICS),
    "novelty_pattern_novelty_rate",
    "fb_precision",
    "fpsi_max",
    *(
        f"pv_{m}_{v}"
        for m in PV_LABEL_MODES
        for v in ("n_invalid", "n_prec_drop", "cov_attack", "cov_benign")
    ),
)

KINDS = (
    "none",
    "attack_mimics_benign",
    "benign_mimics_attack",
    "benign_mix_shift",
    "unseen_field_shift",
    "volume_shift",
    "new_values",
)
# Which groups an injection may touch.
_TARGET = {
    "attack_mimics_benign": "attack",
    "benign_mimics_attack": "benign",
    "benign_mix_shift": "benign",
    "unseen_field_shift": "all",
    "volume_shift": "all",
    "new_values": "all",
}
# Two strengths per kind: moderate and strong. benign_mimics_attack is tiny
# because benign groups outnumber attacks ~75:1.
DEFAULT_FRACTIONS = {
    "attack_mimics_benign": (0.25, 0.5),
    "benign_mimics_attack": (0.005, 0.02),
    "benign_mix_shift": (0.2, 0.5),
    "unseen_field_shift": (0.5, 1.0),
    "volume_shift": (0.5, 1.0),
    "new_values": (0.1, 0.3),
}

# Logged boolean signal columns summarised in detection_summary.csv.
SIGNAL_FLAGS = (
    ("signal_1", "signal_1_elevated"),
    ("signal_2", "signal_2_elevated"),
    *((f"na_{s}", f"na_{s}_elevated") for s in STATISTICS),
    ("feedback_precision", "fb_precision_elevated"),
    ("feature_psi", "fpsi_elevated"),
    ("feature_psi_fixed", "fpsi_fixed_elevated"),
    *(
        (f"pattern_validity_{m}{part}", f"pv_{m}{part}_elevated")
        for m in PV_LABEL_MODES
        for part in ("", "_contrast", "_precision", "_coverage")
    ),
)


@dataclass(frozen=True, slots=True)
class Injection:
    kind: str
    fraction: float

    @property
    def name(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}@{self.fraction:g}"


def default_injections() -> list[Injection]:
    out = [Injection("none", 0.0)]
    for kind, fracs in DEFAULT_FRACTIONS.items():
        out.extend(Injection(kind, f) for f in fracs)
    return out


@dataclass(slots=True)
class Donors:
    attack: list[AlertGroup]
    benign: list[AlertGroup]
    benign_minority: list[AlertGroup]
    minority_category: str


def build_donors(train_rows: Sequence[AlertGroup]) -> Donors:
    """Donor pools from W_src's train split. The minority category is the
    rarest benign category that still covers >= 1% of benign rows (fallback:
    the least common one), so benign_mix_shift raises its share sharply."""
    attack = [g for g in train_rows if g.group_label == "attack"]
    benign = [g for g in train_rows if g.group_label == "benign"]
    counts = Counter(g.category or "" for g in benign)
    if not attack or not benign or not counts:
        raise ValueError("W_src train split needs both attack and benign rows")
    n = sum(counts.values())
    eligible = [c for c, k in counts.items() if k / n >= 0.01]
    pool = eligible or list(counts)
    minority = min(pool, key=lambda c: (counts[c], c))
    return Donors(
        attack=attack,
        benign=benign,
        benign_minority=[g for g in benign if (g.category or "") == minority],
        minority_category=minority,
    )


def _copy_fields(g: AlertGroup, donor: AlertGroup, fields: Sequence[str]) -> AlertGroup:
    return dataclasses.replace(g, **{f: getattr(donor, f) for f in fields})


def perturb_group(
    g: AlertGroup, kind: str, donors: Donors, rng: np.random.Generator
) -> AlertGroup:
    if kind == "attack_mimics_benign":
        return _copy_fields(
            g, donors.benign[rng.integers(len(donors.benign))], PROFILE_FIELDS
        )
    if kind == "benign_mimics_attack":
        return _copy_fields(
            g, donors.attack[rng.integers(len(donors.attack))], PROFILE_FIELDS
        )
    if kind == "benign_mix_shift":
        pool = donors.benign_minority
        return _copy_fields(g, pool[rng.integers(len(pool))], PROFILE_FIELDS)
    if kind == "unseen_field_shift":
        return dataclasses.replace(
            g,
            int_port=int(rng.integers(49152, 65536)),
            ext_port=int(rng.integers(49152, 65536)),
            int_port_is_multiple=not bool(g.int_port_is_multiple),
            ext_port_is_multiple=not bool(g.ext_port_is_multiple),
            qualifiers=set(),
        )
    if kind == "volume_shift":
        return dataclasses.replace(
            g, signature_matches_per_day=(g.signature_matches_per_day or 0.0) * 10.0
        )
    if kind == "new_values":
        return dataclasses.replace(g, category="SYNTHETIC_NEW", ruleset="SYNTHETIC")
    raise ValueError(f"unknown injection kind {kind!r}")


def apply_injection(
    alert_groups: Sequence[AlertGroup],
    start_idx: int,
    injection: Injection,
    donors: Donors,
    seed: int,
) -> list[AlertGroup]:
    """A new timeline: rows before start_idx untouched; from start_idx on, each
    targeted group is replaced (copy, never mutated) with probability
    injection.fraction. Deterministic for a given seed."""
    if injection.kind == "none" or injection.fraction <= 0:
        return list(alert_groups)
    target = _TARGET[injection.kind]
    rng = np.random.default_rng(seed)
    out = list(alert_groups[:start_idx])
    for g in alert_groups[start_idx:]:
        hit = target == "all" or g.group_label == target
        if hit and rng.random() < injection.fraction:
            out.append(perturb_group(g, injection.kind, donors, rng))
        else:
            out.append(g)
    return out


def _rate(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.astype(bool).mean()) if len(s) else float("nan")


def build_detection_summary(per_horizon: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, injection): F1/precision/recall before and after
    the injection point, the post-injection change vs the clean run on the
    same windows (removes natural decay), and per signal its firing rate on
    clean pre-injection windows (false alarms) and on injected windows
    (detection)."""
    rows = []
    walk = per_horizon[per_horizon.horizon_window_index > 0]
    for (model, gran), grp in walk.groupby(["model", "granularity"]):
        clean = grp[grp.injection == "none"].set_index("horizon_window_index")
        for name, run in grp.groupby("injection", sort=False):
            pre = run[~run.injected]
            post = run[run.injected]
            post_h = post.horizon_window_index
            row = {
                "model": model,
                "granularity": gran,
                "injection": name,
                "kind": run.kind.iloc[0],
                "fraction": run.fraction.iloc[0],
                "inject_from_h": run.inject_from_h.iloc[0],
                "n_pre": len(pre),
                "n_post": len(post),
            }
            for m in ("f1", "precision", "recall"):
                row[f"{m}_pre"] = pre[m].mean()
                row[f"{m}_post"] = post[m].mean()
                ref = clean.loc[clean.index.intersection(post_h), m]
                row[f"d_{m}_vs_clean"] = (
                    post[m].mean() - ref.mean() if len(ref) else np.nan
                )
            for col in SIGNAL_VALUES:
                if col not in run.columns:
                    continue
                ref = clean.loc[clean.index.intersection(post_h), col]
                row[f"d_{col}_vs_clean"] = (
                    post[col].mean() - ref.mean() if len(ref) else np.nan
                )
            for label, col in SIGNAL_FLAGS:
                if col not in run.columns:
                    continue
                row[f"{label}_false_alarm"] = _rate(pre[col])
                row[f"{label}_detect"] = _rate(post[col])
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class SyntheticDriftConfig:
    scenario: str
    mining_setting: str
    mining_settings_path: Path
    models: tuple[str, ...] = ("xgboost", "rf")
    granularity: float = 0.05
    inject_from_fraction: float = 0.5
    injections: list[Injection] = field(default_factory=default_injections)
    source_split_mode: str = "baseline_split"
    source_split_time: str | None = None
    train_frac_within_window: float = 0.7
    threshold_mode: str = "fixed"
    calibrated_recall_target: float = 0.90
    cache_dir: Path | None = None
    grouping: object | None = None
    alerts_json_path: Path | None = None
    results_dir: Path | None = None
    random_seed: int = 42
    n_jobs: int = 4
    monitor_consecutive_windows: int = 3
    monitor_min_samples_signal_2: int = 30


def run_synthetic_drift_experiment(config: SyntheticDriftConfig) -> Path:
    ensure_artifact_dirs()
    md_kwargs = dict(
        scenario=config.scenario,
        shortlist_path=Path("unused.csv"),
        train_frac_within_window=config.train_frac_within_window,
        source_split_mode=config.source_split_mode,
        source_split_time=config.source_split_time,
        mining_settings_path=config.mining_settings_path,
        threshold_mode=config.threshold_mode,
        calibrated_recall_target=config.calibrated_recall_target,
        alerts_json_path=config.alerts_json_path,
        random_seed=config.random_seed,
        n_jobs=1,
        monitor_consecutive_windows=config.monitor_consecutive_windows,
        monitor_min_samples_signal_2=config.monitor_min_samples_signal_2,
    )
    if config.cache_dir is not None:
        md_kwargs["cache_dir"] = config.cache_dir
    if config.grouping is not None:
        md_kwargs["grouping"] = config.grouping
    md_config = MonitorDriftConfig(**md_kwargs)

    ctx = load_scenario_context(
        scenario=config.scenario,
        cache_dir=md_config.cache_dir,
        grouping=md_config.grouping,
        alerts_json_path=config.alerts_json_path,
        mining_settings_path=config.mining_settings_path,
    )
    alert_groups = ctx.alert_groups
    n_total = ctx.n_total
    scheme: WindowScheme = _build_window_scheme(md_config, alert_groups, n_total)

    gran = config.granularity
    n_windows = scheme.n_windows(gran)
    inject_from_h = max(1, int(round(config.inject_from_fraction * (n_windows - 1))))
    start_idx = scheme.target_bounds(gran, inject_from_h)[0]
    src_start, src_end = scheme.source_bounds(gran)
    train_end = compute_window_train_end(
        src_start, src_end, config.train_frac_within_window
    )
    donors = build_donors(alert_groups[src_start:train_end])
    print(
        f"  [synthetic] {n_windows} windows at gran={gran:g}; injecting from "
        f"h={inject_from_h} (row {start_idx}); donors: {len(donors.attack)} attack, "
        f"{len(donors.benign)} benign, minority benign category "
        f"{donors.minority_category!r} ({len(donors.benign_minority)} rows)"
    )

    jobs = [
        (model, inj, seed)
        for model in config.models
        for seed, inj in enumerate(config.injections)
    ]
    horizon_rows: list[dict] = []
    signal_rows: list[dict] = []

    def _one(model: str, inj: Injection, seed: int) -> tuple[list[dict], list[dict]]:
        groups = apply_injection(
            alert_groups, start_idx, inj, donors, config.random_seed + seed
        )
        cfg = ShortlistedConfig(
            feature_set="symbolic",
            mining_setting=config.mining_setting,
            granularity=gran,
            model=model,
        )
        print(f"\n[synthetic {model} {inj.name}]")
        h_rows, s_rows = _run_one_monitor_config(
            cfg=cfg,
            config=md_config,
            scenario=config.scenario,
            alert_groups=groups,
            n_total=n_total,
            base_schema=ctx.base_schema,
            mining_settings_by_name=ctx.mining_settings_by_name,
            mining_settings_path=ctx.mining_settings_path,
            scheme=scheme,
        )
        extra = {
            "injection": inj.name,
            "kind": inj.kind,
            "fraction": inj.fraction,
            "inject_from_h": inject_from_h,
            "minority_category": donors.minority_category,
        }
        for r in h_rows:
            r.update(extra)
            r["injected"] = (
                inj.kind != "none" and r["horizon_window_index"] >= inject_from_h
            )
        for r in s_rows:
            r.update(extra)
        return h_rows, s_rows

    with ThreadPoolExecutor(max_workers=config.n_jobs) as pool:
        futures = {pool.submit(_one, *job): job for job in jobs}
        for fut in as_completed(futures):
            model, inj, _ = futures[fut]
            try:
                h_rows, s_rows = fut.result()
                horizon_rows.extend(h_rows)
                signal_rows.extend(s_rows)
                print(f"  done: {model} {inj.name}")
            except Exception as exc:
                print(f"  [warn] {model} {inj.name} failed: {exc}")
                traceback.print_exc()

    results_dir = config.results_dir or (_EXPERIMENTS_DIR / config.scenario)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    per_horizon = pd.DataFrame(horizon_rows)
    per_horizon.to_csv(out_dir / "per_horizon_results.csv", index=False)
    print(f"\n  Saved → {out_dir / 'per_horizon_results.csv'}")
    pd.DataFrame(signal_rows).to_csv(out_dir / "monitor_signals.csv", index=False)
    print(f"  Saved → {out_dir / 'monitor_signals.csv'}")
    summary = (
        build_detection_summary(per_horizon)
        if not per_horizon.empty
        else pd.DataFrame()
    )
    summary.to_csv(out_dir / "detection_summary.csv", index=False)
    print(f"  Saved → {out_dir / 'detection_summary.csv'}")

    lines = [
        f"Synthetic drift test — {ts}",
        f"Scenario: {config.scenario}, mining setting: {config.mining_setting}, "
        f"gran={gran:g}, models={list(config.models)}",
        f"Injected from h={inject_from_h} of {n_windows - 1}; minority benign "
        f"category for benign_mix_shift: {donors.minority_category!r}",
        "",
        summary.to_string(index=False) if not summary.empty else "(no data)",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"  Saved → {out_dir / 'summary.txt'}")
    cfg_dump = {
        **{k: v for k, v in asdict(config).items() if k != "injections"},
        "injections": [i.name for i in config.injections],
    }
    (out_dir / "config.json").write_text(
        pd.Series(cfg_dump).apply(str).to_json(indent=2)
    )
    return out_dir
