"""
Anomaly-detection arm on DEDALE Sigma, on the dataset's recommended split:
weeks 1-2 (23/12-05/01, no attack) train, weeks 3-4 (06/01-19/01, the whole
APT) test. This is the use the split was made for: learn normal behaviour
from clean weeks, flag what deviates. Companion to dedale_comparison.py (the
supervised mining comparison on a cut inside the attack week).

No attack labels in training, so the label-contrast miners of the comparison
(contrast tree, RIPPER, MCTSExtent, and Eclat / PrefixSpan with their
attack-vs-benign gate) have nothing to mine. The symbolic side is instead the
group's items themselves, unmined. Feature sets:
  - base:          the 5 AIT-ADS base features (hour_of_day, n_alerts,
                   n_sigs, n_hosts, n_shorts), as in the comparison.
  - base_items:    base + one binary column per item seen in training
                   (host:, short:, sig: raw items; see dedale_comparison's
                   token mapping). Items only seen at test time get no column.
  - base_items_novel: base_items + n_novel_items, the number of the group's
                   items never seen in training (a new rule, program, parent
                   process, ...). Computed from the training vocabulary only.

Models: IsolationForest (one fit per seed) and OneClassSVM (RBF, nu = 0.05,
standardised; deterministic, fit once), both on the training groups (all
benign). Score = anomaly score (higher = more anomalous).

Threshold: chosen without labels, at the `quantile` of the training groups'
scores (default 0.99: about 1% of normal groups flagged). Reported: F1,
precision, recall and FP/day at that threshold, plus the threshold-free
PR-AUC and ROC-AUC, the flag-all F1, and the paired bootstrap over test
groups between every pair of arms (score_bootstrap.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.dedale_comparison import (
    DATASET,
    DedaleConfig,
    build_dedale_groups,
    flag_all_f1,
    split_groups,
)
from thesis.system_eval.mining_comparison import (
    SCORES_FILE,
    _window_days,
    ait_ads_base_schema,
    metric_row,
    write_bootstrap_summary,
)
from thesis.system_eval.mining_mini_experiment import _labels, _repo_commit, labelled

FEATURE_SETS = ("base", "base_items", "base_items_novel")
MODELS = ("iforest", "ocsvm")


@dataclass
class AnomalyConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    feature_sets: tuple[str, ...] = FEATURE_SETS
    models: tuple[str, ...] = MODELS
    # Threshold = this quantile of the training (benign) scores.
    quantile: float = 0.99
    # Leave host: items out of the item columns and the novelty count.
    no_host: bool = False
    # Leave hour_of_day out of the base features.
    no_hour: bool = False
    # Label groups mixing benign and attack alerts as attack (else dropped).
    mixed_as_attack: bool = False
    out_dir: Path | None = None
    dedale: DedaleConfig = field(default_factory=lambda: DedaleConfig(cut="weeks"))


def _items(g: AlertGroup, no_host: bool) -> set[str]:
    items = set(g.raw_items or ())
    if no_host:
        items = {i for i in items if not i.startswith("host:")}
    return items


def feature_matrices(
    train: list[AlertGroup],
    test: list[AlertGroup],
    feature_set: str,
    cfg: AnomalyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test matrices for one feature set; every column is
    defined from the training groups only."""
    schema = ait_ads_base_schema(
        frozenset({"hour_of_day"}) if cfg.no_hour else frozenset()
    )
    X_tr = encode_alert_groups_for_schema(train, schema).reset_index(drop=True)
    X_te = encode_alert_groups_for_schema(test, schema).reset_index(drop=True)
    if feature_set == "base":
        return X_tr.astype(float), X_te.astype(float)
    vocab = sorted(set().union(*(_items(g, cfg.no_host) for g in train)))
    col = {item: j for j, item in enumerate(vocab)}

    def onehot(groups):
        M = np.zeros((len(groups), len(vocab)), dtype=np.int8)
        novel = np.zeros(len(groups), dtype=float)
        for r, g in enumerate(groups):
            for item in _items(g, cfg.no_host):
                j = col.get(item)
                if j is None:
                    novel[r] += 1
                else:
                    M[r, j] = 1
        return pd.DataFrame(M, columns=[f"item:{v}" for v in vocab]), novel

    I_tr, n_tr = onehot(train)
    I_te, n_te = onehot(test)
    X_tr, X_te = pd.concat([X_tr, I_tr], axis=1), pd.concat([X_te, I_te], axis=1)
    if feature_set == "base_items_novel":
        X_tr["n_novel_items"], X_te["n_novel_items"] = n_tr, n_te
    elif feature_set != "base_items":
        raise ValueError(f"unknown feature set {feature_set!r}")
    return X_tr.astype(float), X_te.astype(float)


def anomaly_scores(
    model: str, X_tr: pd.DataFrame, X_te: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(training scores, test scores), higher = more anomalous."""
    if model == "iforest":
        est = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    elif model == "ocsvm":
        est = Pipeline(
            [("scaler", StandardScaler()), ("clf", OneClassSVM(kernel="rbf", nu=0.05))]
        )
    else:
        raise ValueError(f"unknown model {model!r}")
    est.fit(X_tr.to_numpy())
    return -est.score_samples(X_tr.to_numpy()), -est.score_samples(X_te.to_numpy())


def run_dedale_anomaly(cfg: AnomalyConfig) -> Path:
    from thesis.system_eval.mining_comparison import default_out_dir

    exp = (
        "anomaly_dedale"
        + ("_no_host" if cfg.no_host else "")
        + ("_no_hour" if cfg.no_hour else "")
    )
    out_dir = Path(cfg.out_dir) if cfg.out_dir else default_out_dir(exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {**asdict(cfg), "repo_commit": _repo_commit()}, indent=2, default=str
        )
    )

    groups = build_dedale_groups(cfg.dedale)
    n_mixed = sum(g.group_label == "mixed" for g in groups)
    if cfg.mixed_as_attack:
        for g in groups:
            if g.group_label == "mixed":
                g.group_label = "attack"
    groups = labelled(groups)
    train, test, cut_desc = split_groups(groups, cfg.dedale.cut)
    y_tr, y_te = _labels(train).astype(int), _labels(test).astype(int)
    train_days, test_days = _window_days(train), _window_days(test)
    window = {
        "scenario": DATASET,
        "grouping": cfg.dedale.grouping,
        "grouping_seconds": cfg.dedale.grouping_seconds,
        "cut": cut_desc,
        "n_train": len(train),
        "n_train_attack": int(y_tr.sum()),
        "n_test": len(test),
        "n_test_attack": int(y_te.sum()),
        "test_days": test_days,
        "n_mixed": n_mixed,
        "flag_all_f1": flag_all_f1(test),
    }
    pd.DataFrame([window]).to_csv(out_dir / "windows.csv", index=False)
    print(
        f"[{DATASET} anomaly {cfg.dedale.grouping}] train {len(train)} "
        f"({window['n_train_attack']} attack, {train_days:.1f} days), test "
        f"{len(test)} ({window['n_test_attack']} attack), flag-all F1 "
        f"{window['flag_all_f1']:.4f} → {out_dir}"
    )
    if window["n_train_attack"]:
        print("  [note] the training side holds attack groups; fit on all of them")
    if not y_te.any():
        print("  [skip] no test attacks")
        return out_dir

    rows, scores = [], []
    no_seen = np.zeros(len(test), dtype=bool)
    for fs in cfg.feature_sets:
        X_tr, X_te = feature_matrices(train, test, fs, cfg)
        for model in cfg.models:
            seeds = cfg.seeds if model == "iforest" else cfg.seeds[:1]
            for seed in seeds:
                s_tr, s_te = anomaly_scores(model, X_tr, X_te, seed)
                thr = float(np.quantile(s_tr, cfg.quantile))
                arm = f"{fs}_{model}"
                row = {
                    "arm": arm,
                    "feature_set": fs,
                    "model": model,
                    "seed": seed,
                    "n_features": X_tr.shape[1],
                    **{k: window[k] for k in ("n_train", "n_test", "n_test_attack")},
                    "quantile": cfg.quantile,
                    "threshold": thr,
                    **metric_row(y_te, s_te, thr, no_seen, test_days),
                    "roc_auc": float(roc_auc_score(y_te, s_te)),
                    "flag_all_f1": window["flag_all_f1"],
                }
                for k in [k for k in row if k.endswith("_seen")]:
                    row.pop(k)
                rows.append(row)
                scores.append(
                    pd.DataFrame(
                        {
                            "step_i": 0,
                            "scenario": DATASET,
                            "arm": arm,
                            "seed": seed,
                            "test_idx": np.arange(len(y_te)),
                            "y": y_te,
                            "proba": s_te.astype(np.float32),
                            "pred": s_te >= thr,
                        }
                    )
                )
            last = rows[-1]
            print(
                f"  {fs:<17} {model:<8} f1={last['f1']:.3f} recall={last['recall']:.3f} "
                f"fp/day={last['fp_per_day']:.2f} pr_auc={last['pr_auc']:.3f} "
                f"roc_auc={last['roc_auc']:.3f} ({X_tr.shape[1]} features)",
                flush=True,
            )

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "per_run_results.csv", index=False)
    metrics = ["f1", "precision", "recall", "fp_per_day", "pr_auc", "roc_auc"]
    summary = per_run.groupby(["arm", "feature_set", "model"], sort=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        text = summary.round(4).to_string(index=False)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)
    all_scores = pd.concat(scores, ignore_index=True)
    all_scores.to_csv(out_dir / SCORES_FILE, index=False)
    write_bootstrap_summary(out_dir, all_scores)
    caught_test_attacks(test, all_scores).to_csv(
        out_dir / "test_attacks.csv", index=False
    )
    return out_dir


def caught_test_attacks(test: list[AlertGroup], scores: pd.DataFrame) -> pd.DataFrame:
    """Per test attack group: when, its rules, and the share of seeds each
    arm flags it at the threshold."""
    att = [i for i, g in enumerate(test) if g.group_label == "attack"]
    flagged = (
        scores[scores["test_idx"].isin(att)]
        .groupby(["test_idx", "arm"])["pred"]
        .mean()
        .unstack()
    )
    info = pd.DataFrame(
        {
            "test_idx": att,
            "start": [pd.Timestamp(test[i].start_ts, unit="s") for i in att],
            "n_alerts": [test[i].n_alerts for i in att],
            "rules": [
                " ".join(
                    sorted(x[6:] for x in test[i].raw_items if x.startswith("short:"))
                )
                for i in att
            ],
        }
    ).set_index("test_idx")
    return info.join(flagged).reset_index()
