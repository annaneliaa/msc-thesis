"""
Ours vs DeepCASE end to end on AIT-ADS, frozen, same label budget: a
secondary robustness check for the CSCAS comparison (deepcase_comparison /
deepcase_frozen_comparison). Is the CSCAS result a dataset artefact?

Protocol
--------
* Split (both systems): the AIT-ADS mining comparison's within-scenario
  chronological 70/30 split (mining_comparison.scenario_split over each
  scenario's labelled cscas_grouping groups in time order). Test groups =
  the labelled test 30% plus every mixed / unlabelled group that starts at
  or after the cut (scored, never dropped). Test alerts = the alerts of the
  test groups; every other alert of the scenario is a training alert.
  Both systems see exactly these test alerts.
* No hour_of_day for either side (testbed shortcut, FINDINGS.md). DeepCASE
  never sees a clock feature: its input is event type + machine + order,
  and the time is only used for the context timeout.
* DeepCASE (as deepcase_comparison on CSCAS): one AIT-ADS alert = one
  event, event type = the alert name (`event_field="signature"`, the same
  identity our seen/unseen split uses; "short" = the 75 short codes),
  machine = "<scenario>:<host>". One ContextBuilder + Interpreter trained
  on the training alerts of ALL `scenarios` pooled (it is self-supervised,
  so the 3 scenarios without training attacks still contribute), with a
  train-only vocabulary and an UNKNOWN token sent to the manual queue.
  Tuning (optional, paper App. D): each scenario's training alerts split
  50:50 by time, grid context length x tau x eps, best by e2e F1.
  Simulated analyst: 10 samples per cluster, max label; every outlier is
  inspected. B0 = cluster samples + outliers.
* Ours: base features without hour_of_day, cscas_grouping groups, thesis
  contrast-tree miner under the shared treatment (mining_comparison.run_step:
  item table, gate, dedupes, K), class-weighted RF at 0.5, plus the "none"
  arm (no mining). Trained per scenario on its labelled training groups,
  restricted to the label budget; scenarios with no training attack are
  not run (no attack class to learn), as in the mining comparison.
* Label budget, per seed and scenario: B0_s = DeepCASE's analyst items that
  are training alerts of scenario s. Variants (`budget_variants`):
    stratified_b0   stratified random B0_s labelled training groups (all if
                    B0_s >= the scenario's labelled training groups);
    deepcase_items  the training groups holding an alert DeepCASE's analyst
                    inspected (<= B0_s groups);
    full            every labelled training group (= the mining comparison).
  One label = one inspected item (an alert for DeepCASE, a group for ours).

Unit of evaluation
------------------
* Alert level (primary): every test alert counts once. Ours: an alert is
  escalated iff its group is (group score >= the RF threshold); its score is
  the group's score. DeepCASE: escalated iff auto-flagged attack OR sent to
  the manual queue (decision_metrics' e2e_* columns, as on CSCAS).
  Alert label = the alert's own time-window label, so mixed groups are
  scored alert by alert.
* Group level (secondary): every test cscas_grouping group counts once, as
  attack if it holds any attack alert. Ours: the group decision; DeepCASE:
  the group is escalated if any of its alerts is. AIT-ADS attack floods
  (fox's dirb: ~410k alerts in a handful of groups) dominate the alert
  level, so the group level is the one comparable to the mining comparison.
* Subsets: all / seen / unseen, seen = the alert's event type is in
  DeepCASE's training vocabulary (what decides its "not in training" manual
  queue), applied to both systems' alerts.
* PR-AUC: ours only (group score per alert / per group). DeepCASE has no
  ranking score -- its decision is a code -- so its PR-AUC is not defined;
  the flag-everything PR-AUC (= prevalence) is reported next to ours.

Workload (workload.csv), identical rule for both systems
--------------------------------------------------------
total = B0 labels (training) + escalated test items, where the manual queue
is part of escalated for DeepCASE. In items (what the analyst opens: an
alert for DeepCASE, a group for ours) and in alerts (ours: every alert of
each labelled / escalated group -- the conservative count). The
flag-everything workload (every test item) is given next to it.

Bootstrap (bootstrap.csv)
-------------------------
Paired, cluster-resampled over the test groups of a scenario (stratified by
whether the group holds an attack alert), alert-level metrics recomputed on
each resample from per-group TP/FP/FN counts, each system's metric = mean
over its seeds, delta = ours - DeepCASE. Groups, not alerts, are resampled:
alerts in one group are not independent (one flood = one group). "all"
averages the per-scenario delta inside each resample.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.paths import CACHE_DIR, ROOT
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.deepcase_comparison import (
    DeepCaseConfig,
    Vocab,
    _days,
    _interpret_cell,
    _prf,
    attention_query,
    build_context,
    decision_metrics,
    pick_best,
    train_context_builder,
)
from thesis.system_eval.mining_comparison import (
    AIT_ADS_SCENARIOS,
    COMPARISON_K,
    ComparisonConfig,
    Step,
    _ait_ads_spec,
    _window_days,
    run_step,
    scenario_split,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig, labelled
from thesis.system_eval.score_bootstrap import average_precision

DEFAULT_OUT_ROOT = ROOT / "artifacts" / "experiments" / "deepcase_ait_ads"
BENIGN_LABEL = "false_positive"
EVENT_FIELDS = ("signature", "short")
BUDGET_VARIANTS = ("stratified_b0", "deepcase_items", "full")
OUR_ARMS = ("contrast_tree", "none")
SUBSETS = ("all", "seen", "unseen")


@dataclass
class AitAdsH2HConfig:
    # DeepCASE's training pool (every scenario's training alerts).
    scenarios: tuple[str, ...] = AIT_ADS_SCENARIOS
    # Scenarios scored (None = `scenarios`); must be in `scenarios`.
    eval_scenarios: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    grouping: str = "cscas_grouping"
    event_field: str = "signature"  # "signature" (alert name) | "short"
    # DeepCASE settings (sequencing, ContextBuilder, Interpreter, analyst,
    # tuning grid). csv_path / split_time / granularity / max_*_rows unused.
    deepcase: DeepCaseConfig = field(default_factory=DeepCaseConfig)
    # Ours.
    arms: tuple[str, ...] = OUR_ARMS
    budget_variants: tuple[str, ...] = BUDGET_VARIANTS
    k: int = COMPARISON_K
    settings_variant: str = "default"
    no_hour: bool = True
    mining_cache_dir: Path | None = None
    n_boot: int = 2000
    boot_seed: int = 0
    out_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.event_field not in EVENT_FIELDS:
            raise ValueError(f"event_field must be one of {EVENT_FIELDS}")
        unknown = set(self.budget_variants) - set(BUDGET_VARIANTS)
        if unknown:
            raise ValueError(f"unknown budget variants {sorted(unknown)}")
        bad_arms = set(self.arms) - set(OUR_ARMS)
        if bad_arms:
            raise ValueError(f"arms must be in {OUR_ARMS}, got {sorted(bad_arms)}")
        missing = set(self.eval_scenarios or ()) - set(self.scenarios)
        if missing:
            raise ValueError(
                f"eval scenarios must be in the DeepCASE pool: {sorted(missing)}"
            )

    @property
    def evaluated(self) -> tuple[str, ...]:
        return self.eval_scenarios or self.scenarios


# --- Data -------------------------------------------------------------------


@dataclass
class ScenarioData:
    """One scenario under the shared split."""

    scenario: str
    train_groups: list[AlertGroup]  # labelled training groups (time order)
    test_groups: list[AlertGroup]  # every test group (time order)
    # One row per alert, time order: alert_id, ts, host, event, label (0/1),
    # is_test, test_group (index into test_groups, -1 = training alert),
    # train_group (index into train_groups, -1 = not in a labelled training
    # group).
    alerts: pd.DataFrame
    cut_ts: int

    @property
    def has_training_attack(self) -> bool:
        return any(g.group_label == "attack" for g in self.train_groups)


def split_groups(
    groups: list[AlertGroup],
) -> tuple[list[AlertGroup], list[AlertGroup], int]:
    """(labelled training groups, test groups, cut ts). Training/test of the
    labelled groups is exactly scenario_split; mixed / unlabelled groups
    starting at or after the cut join the test side, earlier ones are
    training-side context only (no group label to train on)."""
    groups = sorted(groups, key=lambda g: g.start_ts)
    tr, te = scenario_split(labelled(groups))
    if not te:
        raise ValueError("empty test split")
    cut = int(te[0].start_ts)
    te_ids = {id(g) for g in te}
    extra = [
        g
        for g in groups
        if g.group_label not in ("attack", "benign") and g.start_ts >= cut
    ]
    test = sorted(te + extra, key=lambda g: g.start_ts)
    assert len({id(g) for g in test}) == len(te_ids) + len(extra)
    return tr, test, cut


def _read_tokenized(scenario: str) -> pd.DataFrame:
    """alert_id, ts, host, short, signature, label of every alert, from the
    tokenized-alert cache (built from data/alerts_csv if missing)."""
    from thesis.grouping.window_sweep import _tok_cache_path, load_and_tokenize

    path = _tok_cache_path(scenario, None)
    if not path.exists():
        load_and_tokenize(scenario)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(
        {
            "alert_id": [r["alert_id"] for r in rows],
            "ts": np.array([int(r["ts"]) for r in rows], dtype=np.int64),
            "host": [r.get("host") or "" for r in rows],
            "short": [r.get("short") or "" for r in rows],
            "signature": [r.get("signature") or "" for r in rows],
            "label_str": [r.get("label") for r in rows],
        }
    )
    del rows
    return df


def alert_labels(label_str: pd.Series) -> np.ndarray:
    """1 = attack: any label other than benign/missing (the group rule of
    grouping.window_sweep._group_label_from_labels)."""
    return (label_str.notna() & (label_str != BENIGN_LABEL)).to_numpy(np.int8)


def attach_alerts(
    alerts: pd.DataFrame,
    train_groups: list[AlertGroup],
    test_groups: list[AlertGroup],
    event_field: str,
) -> pd.DataFrame:
    """Alert table in time order with the split columns (see ScenarioData).

    AIT-ADS has many exact duplicate alerts (same time, host and name), which
    share one alert_id; a group lists such an id once per copy. So alerts and
    group members are matched as (alert_id, k-th copy) -- the copies are
    identical, so which copy goes where doesn't matter."""
    df = alerts.sort_values("ts", kind="stable").reset_index(drop=True)
    members = [
        (aid, side, i)
        for side, groups in ((0, train_groups), (1, test_groups))
        for i, g in enumerate(groups)
        for aid in (g.alert_ids or [])
    ]
    mem = pd.DataFrame(members, columns=["alert_id", "side", "group"])
    mem["copy"] = mem.groupby("alert_id").cumcount()
    left = pd.DataFrame(
        {"alert_id": df["alert_id"], "copy": df.groupby("alert_id").cumcount()}
    )
    joined = left.merge(mem, on=["alert_id", "copy"], how="left", validate="one_to_one")
    n_missing = len(mem) - int(joined["group"].notna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} group members not in the alert table")
    side = joined["side"].fillna(-1).to_numpy(np.int64)
    grp = joined["group"].fillna(-1).to_numpy(np.int64)
    test_group = np.where(side == 1, grp, -1)
    train_group = np.where(side == 0, grp, -1)
    df["event"] = df[event_field]
    df["label"] = alert_labels(df["label_str"])
    df["is_test"] = test_group != -1
    df["test_group"] = test_group
    df["train_group"] = train_group
    return df[
        [
            "alert_id",
            "ts",
            "host",
            "event",
            "label",
            "is_test",
            "test_group",
            "train_group",
        ]
    ]


def load_scenario(scenario: str, grouping: str, event_field: str) -> ScenarioData:
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    groups = load_or_build_alert_groups(
        scenario, CACHE_DIR / scenario / "groups" / grouping
    )
    tr, te, cut = split_groups(groups)
    alerts = attach_alerts(_read_tokenized(scenario), tr, te, event_field)
    return ScenarioData(scenario, tr, te, alerts, cut)


def split_table(data: dict[str, ScenarioData]) -> pd.DataFrame:
    rows = []
    for s, d in data.items():
        a = d.alerts
        te = a[a.is_test]
        rows.append(
            {
                "scenario": s,
                "cut_ts": d.cut_ts,
                "n_train_groups": len(d.train_groups),
                "n_train_groups_attack": sum(
                    g.group_label == "attack" for g in d.train_groups
                ),
                "n_test_groups": len(d.test_groups),
                "n_test_groups_attack_any": int(
                    te.groupby("test_group")["label"].max().sum()
                ),
                "n_test_groups_mixed": sum(
                    g.group_label not in ("attack", "benign") for g in d.test_groups
                ),
                "n_train_alerts": int((~a.is_test).sum()),
                "n_train_alerts_attack": int(a.label[~a.is_test].sum()),
                "n_test_alerts": len(te),
                "n_test_alerts_attack": int(te.label.sum()),
                "test_alert_prevalence": float(te.label.mean()) if len(te) else np.nan,
                "test_days": _days(te.ts.to_numpy()),
                "ours_runs": d.has_training_attack,
            }
        )
    return pd.DataFrame(rows)


# --- DeepCASE on the pooled scenarios -----------------------------------------


@dataclass
class Pool:
    """Every alert of every DeepCASE-pool scenario, concatenated per scenario
    in time order (so each machine's rows are chronological)."""

    scenario: np.ndarray  # object
    row: np.ndarray  # index into that scenario's alerts table
    ts: np.ndarray
    event: np.ndarray  # int64 event-type code (global)
    event_names: np.ndarray  # code -> event type string
    machine: np.ndarray  # object "<scenario>:<host>"
    label: np.ndarray  # int8
    is_test: np.ndarray  # bool

    @property
    def train_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.is_test)

    @property
    def test_idx(self) -> np.ndarray:
        return np.flatnonzero(self.is_test)

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for arr in (self.ts, self.event, self.is_test.astype(np.int8)):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:12]


def build_pool(data: dict[str, ScenarioData]) -> Pool:
    parts = [(s, d.alerts) for s, d in data.items()]
    events = pd.concat([a["event"] for _, a in parts], ignore_index=True)
    codes, names = pd.factorize(events, sort=True)
    machine = np.concatenate(
        [(s + ":" + a["host"].astype(str)).to_numpy(dtype=object) for s, a in parts]
    )
    return Pool(
        scenario=np.concatenate([np.full(len(a), s, dtype=object) for s, a in parts]),
        row=np.concatenate([np.arange(len(a)) for _, a in parts]),
        ts=np.concatenate([a["ts"].to_numpy(np.int64) for _, a in parts]),
        event=codes.astype(np.int64),
        event_names=np.asarray(names, dtype=object),
        machine=machine,
        label=np.concatenate([a["label"].to_numpy(np.int8) for _, a in parts]),
        is_test=np.concatenate([a["is_test"].to_numpy(bool) for _, a in parts]),
    )


def tune_halves(pool: Pool, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each scenario's training alerts split 50:50 by time (rows are already
    in time order per scenario): (first halves, second halves), as pool
    indices in pool order."""
    first, second = [], []
    sc = pool.scenario[train_idx]
    for s in pd.unique(sc):
        idx = train_idx[sc == s]
        h = len(idx) // 2
        first.append(idx[:h])
        second.append(idx[h:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def _cb_train_id(tag: str, pool: Pool, n_rows: int, vocab: Vocab, cfg) -> str:
    return (
        f"aitads_{tag}_{'+'.join(sorted(set(pool.scenario)))}_{cfg.event_field}"
        f"_n{n_rows}_v{vocab.size}_{pool.fingerprint()}"
    )


def tune_deepcase(cfg: AitAdsH2HConfig, pool: Pool, device, out_dir: Path) -> dict:
    """Paper App. D on the training alerts only (see tune_halves)."""
    dc = cfg.deepcase
    a_idx, b_idx = tune_halves(pool, pool.train_idx)
    sub = np.sort(np.concatenate([a_idx, b_idx]))  # all training rows, pool order
    in_a = np.isin(sub, a_idx)
    vocab = Vocab.from_signatures(pool.event[a_idx])
    ids = vocab.encode(pool.event[sub])
    label_a, label_b = pool.label[a_idx], pool.label[b_idx]
    days_b = _days(np.sort(pool.ts[b_idx]))
    epochs = dc.tune_epochs or dc.epochs
    rows: list[dict] = []
    for L in dc.tune_context_lengths:
        t0 = time.time()
        ctx = build_context(
            ids, pool.machine[sub], pool.ts[sub], L, dc.timeout, vocab.no_event_id
        )
        cb = train_context_builder(
            ctx[in_a],
            ids[in_a],
            vocab,
            _cb_train_id("tune", pool, int(in_a.sum()), vocab, cfg),
            dc,
            L,
            epochs,
            dc.tune_seed,
            device,
        )
        att_a = attention_query(
            cb, vocab, ctx[in_a], ids[in_a], dc, device, dc.tune_seed
        )
        att_b = attention_query(
            cb, vocab, ctx[~in_a], ids[~in_a], dc, device, dc.tune_seed
        )
        print(
            f"  [tune] L={L}: ContextBuilder + attention {time.time() - t0:.0f}s",
            flush=True,
        )
        cache: dict = {}
        for tau in dc.tune_thresholds:
            for eps in dc.tune_eps:
                rng = np.random.default_rng(dc.tune_seed)
                codes, analyst, _ = _interpret_cell(
                    att_a, att_b, label_a, vocab, tau, eps, dc, rng, cache
                )
                m = decision_metrics(label_b, codes, days_b)
                rows.append(
                    {
                        "context_length": L,
                        "threshold": tau,
                        "eps": eps,
                        **{f"train_{k}": v for k, v in analyst.stats.items()},
                        **m,
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / "tuning.csv", index=False)
        print(f"  [tune] L={L} grid done ({time.time() - t0:.0f}s)", flush=True)
    return pick_best(pd.DataFrame(rows), dc.tune_objective)


@dataclass
class DeepCaseSeedRun:
    codes: np.ndarray  # per pool test row
    seen: np.ndarray  # per pool test row: event type in the training vocab
    inspected: np.ndarray  # pool indices of analyst items
    inspected_kind: np.ndarray  # "cluster_sample" | "outlier"
    stats: dict


def run_deepcase_seed(
    cfg: AitAdsH2HConfig,
    pool: Pool,
    L: int,
    tau: float,
    eps: float,
    seed: int,
    device,
    ctx_cache: dict,
) -> DeepCaseSeedRun:
    dc = cfg.deepcase
    tr, te = pool.train_idx, pool.test_idx
    vocab = Vocab.from_signatures(pool.event[tr])
    ids = vocab.encode(pool.event)
    if L not in ctx_cache:
        ctx_cache.clear()
        ctx_cache[L] = build_context(
            ids, pool.machine, pool.ts, L, dc.timeout, vocab.no_event_id
        )
    ctx = ctx_cache[L]
    cb = train_context_builder(
        ctx[tr],
        ids[tr],
        vocab,
        _cb_train_id("train", pool, len(tr), vocab, cfg),
        dc,
        L,
        dc.epochs,
        seed,
        device,
    )
    att_tr = attention_query(cb, vocab, ctx[tr], ids[tr], dc, device, seed)
    att_te = attention_query(cb, vocab, ctx[te], ids[te], dc, device, seed)
    rng = np.random.default_rng(seed)
    codes, analyst, _clusters = _interpret_cell(
        att_tr, att_te, pool.label[tr], vocab, tau, eps, dc, rng
    )
    inspected_local = np.concatenate([analyst.sampled_idx, analyst.outlier_idx])
    kind = np.array(
        ["cluster_sample"] * len(analyst.sampled_idx)
        + ["outlier"] * len(analyst.outlier_idx),
        dtype=object,
    )
    return DeepCaseSeedRun(
        codes=codes,
        seen=np.isin(pool.event[te], vocab.signatures),
        inspected=tr[inspected_local],
        inspected_kind=kind,
        stats=analyst.stats,
    )


# --- Ours -----------------------------------------------------------------------


def budget_mask(
    variant: str,
    train_groups: list[AlertGroup],
    b0: int,
    inspected_groups: set[int],
    seed: int,
) -> np.ndarray:
    """Which labelled training groups our system gets labels for."""
    from thesis.system_eval.label_budget import draw_budget

    n = len(train_groups)
    if variant == "full":
        return np.ones(n, dtype=bool)
    if variant == "stratified_b0":
        if b0 >= n:
            return np.ones(n, dtype=bool)
        return draw_budget(train_groups, "stratified", size=b0, seed=seed)
    if variant == "deepcase_items":
        m = np.zeros(n, dtype=bool)
        m[sorted(inspected_groups)] = True
        return m
    raise ValueError(f"unknown budget variant {variant!r}")


def comparison_config(cfg: AitAdsH2HConfig, seed: int) -> ComparisonConfig:
    from thesis.system_eval.mining_settings_sweep import apply_variant

    return ComparisonConfig(
        dataset="ait_ads",
        seeds=(seed,),
        methods=tuple(a for a in cfg.arms if a != "none"),
        include_no_mining="none" in cfg.arms,
        mining=apply_variant(replace(MiniConfig(), k=cfg.k), cfg.settings_variant),
        base_exclude_features=frozenset({"hour_of_day"})
        if cfg.no_hour
        else frozenset(),
        grouping=cfg.grouping,
        save_test_scores=True,
        mining_cache_dir=cfg.mining_cache_dir,
        stage="all",
    )


def run_ours(
    cfg: AitAdsH2HConfig, d: ScenarioData, mask: np.ndarray, seed: int
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit our arms on the budgeted training groups, score every test group.
    Returns (scores: arm, test_idx, proba, pred; run_step's rows)."""
    ccfg = comparison_config(cfg, seed)
    train = [g for g, k in zip(d.train_groups, mask) if k]
    step = Step(
        0, train, d.test_groups, _window_days(d.test_groups), scenario=d.scenario
    )
    spec = _ait_ads_spec(ccfg, [d.scenario], lambda groups: None)
    scores: list[pd.DataFrame] = []
    rows, _ = run_step(ccfg, step, spec, scores)
    sc = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    return sc, rows


# --- Metrics -----------------------------------------------------------------


def _subset_masks(seen: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(seen), dtype=bool), "seen": seen, "unseen": ~seen}


def flag_all_row(label: np.ndarray) -> dict:
    n, n_pos = len(label), int(label.sum())
    p = n_pos / n if n else np.nan
    return {
        "flag_all_precision": p,
        "flag_all_recall": 1.0 if n_pos else np.nan,
        "flag_all_f1": 2 * p / (1 + p) if n_pos else np.nan,
        "flag_all_pr_auc": p,
    }


def alert_metric_rows(
    label: np.ndarray,
    codes: np.ndarray,
    seen: np.ndarray,
    days: float,
    score: np.ndarray | None = None,
) -> list[dict]:
    """decision_metrics per subset (codes: >=0 auto decision, <0 manual),
    plus PR-AUC from `score` and the flag-everything reference."""
    out = []
    for subset, m in _subset_masks(seen).items():
        r = decision_metrics(label[m], codes[m], days)
        r["pr_auc"] = (
            average_precision(label[m].astype(int), score[m])
            if score is not None and label[m].any()
            else np.nan
        )
        out.append({"subset": subset, **r, **flag_all_row(label[m])})
    return out


def group_level(
    test_group: np.ndarray, label: np.ndarray, escalated: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """(group label = any attack alert, group escalated = any alert escalated)."""
    g_label = np.zeros(n_groups, dtype=bool)
    g_esc = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(g_label, test_group, label.astype(bool))
    np.logical_or.at(g_esc, test_group, escalated.astype(bool))
    return g_label, g_esc


def group_metric_row(
    g_label: np.ndarray, g_esc: np.ndarray, g_score: np.ndarray | None = None
) -> dict:
    tp = int((g_esc & g_label).sum())
    fp = int((g_esc & ~g_label).sum())
    fn = int((~g_esc & g_label).sum())
    p, r, f = _prf(tp, fp, fn)
    return {
        "n": len(g_label),
        "n_attack": int(g_label.sum()),
        "tp": tp,
        "fp": fp,
        "missed_attacks": fn,
        "precision": p,
        "recall": r,
        "f1": f,
        "escalated": int(g_esc.sum()),
        "pr_auc": (
            average_precision(g_label.astype(int), g_score)
            if g_score is not None and g_label.any()
            else np.nan
        ),
        **flag_all_row(g_label),
    }


def per_group_counts(
    test_group: np.ndarray, label: np.ndarray, escalated: np.ndarray, n_groups: int
) -> np.ndarray:
    """(n_groups, 3) alert-level TP / FP / FN per test group."""
    lab, esc = label.astype(bool), escalated.astype(bool)
    out = np.zeros((n_groups, 3), dtype=np.int64)
    for j, m in enumerate((esc & lab, esc & ~lab, ~esc & lab)):
        out[:, j] = np.bincount(test_group[m], minlength=n_groups)
    return out


def _metric_from_counts(tp, fp, fn, metric: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "f1":
            d = 2 * tp + fp + fn
            return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
        if metric == "recall":
            d = tp + fn
            return np.where(d > 0, tp / np.maximum(d, 1), np.nan)
        if metric == "precision":
            d = tp + fp
            return np.where(d > 0, tp / np.maximum(d, 1), np.nan)
    raise ValueError(metric)


BOOT_METRICS = ("f1", "recall", "precision")


def cluster_bootstrap(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    strata: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, np.ndarray]]:
    """counts_*: (seeds, groups, 3) TP/FP/FN per group. Resample groups with
    replacement within each stratum; metric per seed from the summed counts,
    mean over seeds; returns metric -> (point delta a-b, resampled deltas)."""
    G = counts_a.shape[1]
    W = np.zeros((n_boot, G), dtype=np.float64)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        for b in range(n_boot):
            W[b] += np.bincount(idx[draws[b]], minlength=G)
    out = {}
    for metric in BOOT_METRICS:

        def stat(c, w):
            s = np.einsum("bg,kgj->bkj", w, c.astype(np.float64))
            return np.nanmean(
                _metric_from_counts(s[..., 0], s[..., 1], s[..., 2], metric), axis=1
            )

        one = np.ones((1, G))
        point = float(stat(counts_a, one)[0] - stat(counts_b, one)[0])
        out[metric] = (point, stat(counts_a, W) - stat(counts_b, W))
    return out


def _boot_row(scenario, arm, variant, metric, point, d) -> dict:
    d = d[~np.isnan(d)]
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
    return {
        "scenario": scenario,
        "arm": arm,
        "variant": variant,
        "metric": metric,
        "delta": point,
        "ci_low": float(np.percentile(d, 2.5)) if len(d) else np.nan,
        "ci_high": float(np.percentile(d, 97.5)) if len(d) else np.nan,
        "p_boot": p,
        "n_boot": len(d),
    }


# --- Main run -----------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def run_deepcase_ait_ads(cfg: AitAdsH2HConfig) -> Path:
    from thesis.grouping._device import resolve_device
    from thesis.system_eval.mining_mini_experiment import _repo_commit

    device = resolve_device(cfg.deepcase.device)
    out_dir = (
        Path(cfg.out_dir)
        if cfg.out_dir
        else DEFAULT_OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput → {out_dir}", flush=True)

    t0 = time.time()
    data: dict[str, ScenarioData] = {}
    for s in cfg.scenarios:
        data[s] = load_scenario(s, cfg.grouping, cfg.event_field)
        print(f"  loaded {s}: {len(data[s].alerts)} alerts", flush=True)
    split = split_table(data)
    split.to_csv(out_dir / "split.csv", index=False)
    print(split.to_string(index=False), flush=True)
    pool = build_pool(data)
    print(
        f"Pool: {len(pool.ts)} alerts ({len(pool.train_idx)} train, "
        f"{len(pool.test_idx)} test), {len(pool.event_names)} event types, "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )

    run_cfg = {**asdict(cfg), "repo_commit": _repo_commit()}
    dc = cfg.deepcase
    if dc.tune:
        print("Tuning DeepCASE on the training alerts ...", flush=True)
        best = tune_deepcase(cfg, pool, device, out_dir)
        print(f"Tuned: {best}", flush=True)
        run_cfg["tuned"] = best
        L, tau, eps = best["context_length"], best["threshold"], best["eps"]
    else:
        L, tau, eps = dc.context_length, dc.threshold, dc.eps
    run_cfg["used"] = {"context_length": L, "threshold": tau, "eps": eps}
    (out_dir / "config.json").write_text(
        json.dumps(run_cfg, indent=2, default=_json_default)
    )

    te = pool.test_idx
    te_pos = {s: np.flatnonzero(pool.scenario[te] == s) for s in cfg.evaluated}
    ctx_cache: dict = {}
    alert_rows: list[dict] = []
    group_rows: list[dict] = []
    work_rows: list[dict] = []
    budget_rows: list[dict] = []
    ours_rows: list[dict] = []
    # (scenario, system/arm, variant) -> per-seed (groups, 3) counts
    counts: dict[tuple, list[np.ndarray]] = {}

    def flush():
        for name, rows in (
            ("alert_metrics", alert_rows),
            ("group_metrics", group_rows),
            ("workload", work_rows),
            ("deepcase_budget", budget_rows),
            ("ours_step_rows", ours_rows),
        ):
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)

    for seed in cfg.seeds:
        ts0 = time.time()
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        run = run_deepcase_seed(cfg, pool, L, tau, eps, seed, device, ctx_cache)
        pd.DataFrame(
            {
                "scenario": pool.scenario[te],
                "alert_row": pool.row[te],
                "label": pool.label[te],
                "event": pool.event_names[pool.event[te]],
                "seen": run.seen,
                "code": run.codes,
            }
        ).to_parquet(seed_dir / "deepcase_test_predictions.parquet", index=False)
        insp = pd.DataFrame(
            {
                "scenario": pool.scenario[run.inspected],
                "alert_row": pool.row[run.inspected],
                "kind": run.inspected_kind,
                "label": pool.label[run.inspected],
            }
        )
        insp.to_parquet(seed_dir / "deepcase_analyst_items.parquet", index=False)
        budget_rows.append({"seed": seed, "scenario": "all", **run.stats})
        print(
            f"  seed {seed}: DeepCASE B0={run.stats['budget_b0']} "
            f"({run.stats['n_clusters']} clusters, {run.stats['n_outliers']} outliers), "
            f"{time.time() - ts0:.0f}s",
            flush=True,
        )

        for s in cfg.evaluated:
            d = data[s]
            a = d.alerts
            pos = te_pos[s]
            rows_s = pool.row[te][pos]  # scenario alert rows, test only
            label = a["label"].to_numpy()[rows_s]
            tg = a["test_group"].to_numpy()[rows_s]
            ts_s = a["ts"].to_numpy()[rows_s]
            seen = run.seen[pos]
            codes = run.codes[pos]
            days = _days(np.sort(ts_s))
            nG = len(d.test_groups)
            strata = np.zeros(nG, dtype=bool)
            np.logical_or.at(strata, tg, label.astype(bool))

            si = insp[insp.scenario == s]
            b0_s = len(si)
            b0_clusters_s = int((si.kind == "cluster_sample").sum())
            tr_grp = a["train_group"].to_numpy()[si.alert_row.to_numpy()]
            inspected_groups = set(int(g) for g in tr_grp if g >= 0)
            budget_rows.append(
                {
                    "seed": seed,
                    "scenario": s,
                    "budget_b0": b0_s,
                    "budget_clusters_only": b0_clusters_s,
                    "n_inspected_attack": int(si.label.sum()),
                    "n_train_groups_with_inspected_alert": len(inspected_groups),
                }
            )

            # DeepCASE
            dc_esc = codes < 0
            dc_esc |= codes > 0
            key = dict(
                scenario=s,
                system="deepcase",
                arm="deepcase",
                variant="native",
                seed=seed,
            )
            for r in alert_metric_rows(label, codes, seen, days):
                alert_rows.append({**key, **r})
            for subset, m in _subset_masks(seen).items():
                gl, ge = group_level(tg[m], label[m], dc_esc[m], nG)
                present = np.zeros(nG, dtype=bool)
                present[tg[m]] = True
                group_rows.append(
                    {
                        **key,
                        "subset": subset,
                        **group_metric_row(gl[present], ge[present]),
                    }
                )
            n_manual = int((codes < 0).sum())
            work_rows.append(
                {
                    **key,
                    "unit": "alert",
                    "labels_b0_items": b0_s,
                    "labels_b0_alerts": b0_s,
                    "test_escalated_items": int(dc_esc.sum()),
                    "test_escalated_alerts": int(dc_esc.sum()),
                    "test_auto_flagged": int((codes > 0).sum()),
                    "test_manual_queue": n_manual,
                    "test_manual_unseen": int((codes[~seen] < 0).sum()),
                    "test_manual_low_confidence": int((codes == -1).sum()),
                    "test_manual_not_in_training": int((codes == -2).sum()),
                    "test_manual_too_far": int((codes == -3).sum()),
                    "total_items": b0_s + int(dc_esc.sum()),
                    "total_alerts": b0_s + int(dc_esc.sum()),
                    "flag_all_items": len(label),
                    "flag_all_alerts": len(label),
                    "n_test_alerts": len(label),
                    "n_test_groups": nG,
                }
            )
            counts.setdefault((s, "deepcase", "native"), []).append(
                per_group_counts(tg, label, dc_esc, nG)
            )

            # Ours
            if not d.has_training_attack:
                continue
            n_alerts_tr = np.array([g.n_alerts for g in d.train_groups])
            n_alerts_te = np.array([g.n_alerts for g in d.test_groups])
            for variant in cfg.budget_variants:
                mask = budget_mask(
                    variant, d.train_groups, b0_s, inspected_groups, seed
                )
                n_lab = int(mask.sum())
                lab_attack = sum(
                    g.group_label == "attack" for g, k in zip(d.train_groups, mask) if k
                )
                if lab_attack == 0 or lab_attack == n_lab:
                    print(
                        f"  [skip] ours {s} {variant} seed {seed}: "
                        f"{lab_attack}/{n_lab} labelled training groups attack",
                        flush=True,
                    )
                    ours_rows.append(
                        {
                            "scenario": s,
                            "variant": variant,
                            "seed": seed,
                            "status": "single-class budget",
                            "n_labelled": n_lab,
                        }
                    )
                    continue
                print(
                    f"  ours {s} {variant} seed {seed}: {n_lab}/{len(mask)} labelled "
                    f"groups ({lab_attack} attack)",
                    flush=True,
                )
                sc, rows = run_ours(cfg, d, mask, seed)
                for r in rows:
                    ours_rows.append({"variant": variant, "n_labelled": n_lab, **r})
                for arm, g in sc.groupby("arm"):
                    g = g.sort_values("test_idx")
                    proba = g["proba"].to_numpy(np.float64)
                    pred = g["pred"].to_numpy(bool)
                    esc = pred[tg]
                    score = proba[tg]
                    key = dict(
                        scenario=s, system="ours", arm=arm, variant=variant, seed=seed
                    )
                    for r in alert_metric_rows(
                        label, esc.astype(int), seen, days, score
                    ):
                        alert_rows.append({**key, **r})
                    for subset, m in _subset_masks(seen).items():
                        gl, _ = group_level(tg[m], label[m], esc[m], nG)
                        present = np.zeros(nG, dtype=bool)
                        present[tg[m]] = True
                        group_rows.append(
                            {
                                **key,
                                "subset": subset,
                                **group_metric_row(
                                    gl[present], pred[present], proba[present]
                                ),
                            }
                        )
                    work_rows.append(
                        {
                            **key,
                            "unit": "group",
                            "labels_b0_items": n_lab,
                            "labels_b0_alerts": int(n_alerts_tr[mask].sum()),
                            "test_escalated_items": int(pred.sum()),
                            "test_escalated_alerts": int(n_alerts_te[pred].sum()),
                            "test_auto_flagged": int(pred.sum()),
                            "test_manual_queue": 0,
                            "total_items": n_lab + int(pred.sum()),
                            "total_alerts": int(n_alerts_tr[mask].sum())
                            + int(n_alerts_te[pred].sum()),
                            "flag_all_items": nG,
                            "flag_all_alerts": len(label),
                            "n_test_alerts": len(label),
                            "n_test_groups": nG,
                        }
                    )
                    counts.setdefault((s, arm, variant), []).append(
                        per_group_counts(tg, label, esc, nG)
                    )
            counts.setdefault(("_strata", s), [strata])
        flush()
        print(f"  seed {seed} done ({time.time() - ts0:.0f}s)", flush=True)

    boot = bootstrap_table(cfg, counts)
    boot.to_csv(out_dir / "bootstrap.csv", index=False)
    summary = summarise(
        pd.DataFrame(alert_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(work_rows),
        boot,
        split,
    )
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"Saved → {out_dir}")
    return out_dir


def bootstrap_table(cfg: AitAdsH2HConfig, counts: dict) -> pd.DataFrame:
    """Paired cluster bootstrap, every ours (arm, variant) vs DeepCASE, per
    scenario and averaged over the scenarios both systems ran on."""
    rng = np.random.default_rng(cfg.boot_seed)
    rows = []
    pooled: dict[tuple, list] = {}
    for key, ours in counts.items():
        if key[0] == "_strata" or key[1] == "deepcase":
            continue
        s, arm, variant = key
        dc = counts.get((s, "deepcase", "native"))
        if not dc or len(dc) != len(ours):
            continue
        strata = counts[("_strata", s)][0]
        res = cluster_bootstrap(
            np.stack(ours),
            np.stack(dc),
            strata,
            cfg.n_boot,
            np.random.default_rng(rng.integers(2**32)),
        )
        for metric, (point, d) in res.items():
            rows.append(_boot_row(s, arm, variant, metric, point, d))
            pooled.setdefault((arm, variant, metric), []).append((point, d))
    for (arm, variant, metric), items in pooled.items():
        if len(items) < 2:
            continue
        point = float(np.mean([p for p, _ in items]))
        d = np.nanmean(np.stack([x for _, x in items]), axis=0)
        rows.append(_boot_row("all", arm, variant, metric, point, d))
    return pd.DataFrame(rows)


def summarise(
    alerts: pd.DataFrame,
    groups: pd.DataFrame,
    work: pd.DataFrame,
    boot: pd.DataFrame,
    split: pd.DataFrame,
) -> str:
    keys = ["scenario", "system", "arm", "variant"]
    lines = ["Ours vs DeepCASE on AIT-ADS (frozen, 70/30 split) -- mean over seeds", ""]
    lines.append("[split]")
    lines.append(
        split[
            [
                "scenario",
                "n_train_alerts",
                "n_test_alerts",
                "n_test_alerts_attack",
                "test_alert_prevalence",
                "n_test_groups",
                "n_test_groups_attack_any",
                "ours_runs",
            ]
        ].to_string(index=False)
    )

    def block(df, cols, title):
        sub = df[df.subset == "all"]
        if sub.empty:
            return
        t = sub.groupby(keys, sort=False)[cols].mean().reset_index()
        lines.append("")
        lines.append(title)
        lines.append(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    block(
        alerts,
        ["e2e_recall", "e2e_precision", "e2e_f1", "pr_auc", "flag_all_f1", "coverage"],
        "[alert level, all signatures]",
    )
    block(
        groups,
        ["recall", "precision", "f1", "pr_auc", "flag_all_f1"],
        "[group level (cscas_grouping groups), all signatures]",
    )
    unseen = alerts[alerts.subset == "unseen"]
    if not unseen.empty:
        t = (
            unseen.groupby(keys, sort=False)[
                ["n", "n_attack", "e2e_recall", "n_manual"]
            ]
            .mean()
            .reset_index()
        )
        lines += [
            "",
            "[alert level, unseen signatures: n, attacks, recall, manual queue]",
        ]
        lines.append(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not work.empty:
        t = (
            work.groupby(keys, sort=False)[
                [
                    "labels_b0_items",
                    "test_escalated_items",
                    "test_manual_queue",
                    "total_items",
                    "total_alerts",
                    "flag_all_items",
                ]
            ]
            .mean()
            .reset_index()
        )
        lines += [
            "",
            "[workload: B0 labels + escalated test items (manual queue included)]",
        ]
        lines.append(t.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    if not boot.empty:
        lines += [
            "",
            "[paired cluster bootstrap over test groups, alert level, ours - DeepCASE]",
        ]
        lines.append(boot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return "\n".join(lines) + "\n"


__all__ = [
    "AitAdsH2HConfig",
    "ScenarioData",
    "split_groups",
    "attach_alerts",
    "build_pool",
    "tune_halves",
    "budget_mask",
    "alert_metric_rows",
    "group_level",
    "group_metric_row",
    "per_group_counts",
    "cluster_bootstrap",
    "run_deepcase_ait_ads",
]
