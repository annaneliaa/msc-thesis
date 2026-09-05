"""
IsolationForest counterpart to baselines/ait_ads_mining_anomaly.py: the
same 5-column AIT-ADS base schema extended with attribute-mined symbolic
features (contrast-set + decision-tree rules) on the train split, but fit
as a one-class IsolationForest on benign-only rows instead of a
OneClassSVM.

Stands to ait_ads_mining_anomaly.py as ait_ads_anomaly_iforest.py stands to
ait_ads_anomaly.py -- isolates the anomaly-detector model choice within the
mining scenario. Uses the same group-returning loader and the same mining
step (same run_name / cache namespace), so the mined predicates are shared
with the OneClassSVM sibling.

Guard is TEST-SIDE ONLY (test["Label"].nunique() < 2), same as
ait_ads_anomaly.py -- a 0-attack train split is fine (the model is fit on
benign rows only regardless), and mining degrades to 0 predicates rather
than raising when train has no attack rows to contrast against.

Unlike the OneClassSVM sibling (a deterministic convex fit), IsolationForest
tree bootstrapping is stochastic, so this runs the 5-seed protocol the
trainable baselines use -- IsolationForest(random_state=seed) for seed in
range(5), fit on the identical (base + mined) benign rows every seed -- and
stores the seed mean plus the per-seed breakdown. Each result also stores
`workload_at_recall` (seed-averaged): precision / FP / analyst-workload-
reduction at the threshold that hits each target recall.

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists *and is in the current format* (has workload_at_recall, and
for the IsolationForest scripts a 5-seed breakdown) -- a stale-format
result left over from an older run is recomputed automatically, no need to
hand-delete it. Set AIT_ADS_FORCE=1 to force a full re-run.

Run:
    cd src/thesis/baselines
    python ait_ads_mining_anomaly_iforest.py
"""

import os
import traceback

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split_with_groups,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import anomaly_results_current, save_anomaly_results
from thesis.configs import load_scenarios
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import save_alert_groups_json
from thesis.schemas.mining import AttributeMiningConfig
from thesis.training.workload import (
    average_workload_at_recall,
    compute_workload_at_recall,
)

FEATURE_COLS = ["hour_of_day", "n_alerts", "n_sigs", "n_hosts", "n_shorts"]

FORCE = os.environ.get("AIT_ADS_FORCE", "0") == "1"

print("Using device: cpu")

_scenarios_env = os.environ.get("AIT_ADS_SCENARIOS")
SCENARIOS = (
    [s.strip() for s in _scenarios_env.split(",") if s.strip()]
    if _scenarios_env
    else load_scenarios("ait-ads")
)

_grouping_methods_env = os.environ.get("AIT_ADS_GROUPING_METHODS")
GROUPING_METHODS = (
    [g.strip() for g in _grouping_methods_env.split(",") if g.strip()]
    if _grouping_methods_env
    else ALL_GROUPING_METHODS
)


def run_scenario(scenario: str, grouping_method: str) -> None:
    run_tag = f"{grouping_method}_{scenario}"
    result_name = f"ait_ads_mining_anomaly_{run_tag}_iforest"
    print(
        f"\n{'=' * 70}\n  SCENARIO: {scenario} / GROUPING: {grouping_method}\n{'=' * 70}"
    )
    if not FORCE and anomaly_results_current(result_name, require_seeds=True):
        print(
            f"  [skip] {run_tag}: {result_name}.json already exists (set AIT_ADS_FORCE=1 to re-run)."
        )
        return
    if grouping_method in LEARNED_METHODS and scenario in LEAKAGE_SCENARIOS:
        print(
            f"  [skip] {scenario}/{grouping_method}: would be self-training "
            f"leakage -- {grouping_method}'s model was trained on this scenario. "
            "See _ait_ads_grouping.py's module docstring."
        )
        return

    train, test, train_groups, test_groups = load_ait_ads_baseline_split_with_groups(
        scenario, grouping_method=grouping_method
    )
    print(
        f"  {len(train)} train / {len(test)} test alert_groups, "
        f"{int(train['Label'].sum())} train positive, "
        f"{int(test['Label'].sum())} test positive"
    )
    if test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class test split.")
        return

    train_benign = train[train["Label"] == 0]
    if len(train_benign) == 0:
        print(f"  [skip] {run_tag}: no benign rows in train split.")
        return

    # train.index equals positional row order; train_groups is aligned 1:1
    # by position -- train_benign.index gives positions into symbolic_train_df.
    assert list(train.index) == list(range(len(train)))

    train_alert_groups_path = (
        CACHE_DIR
        / scenario
        / "groups"
        / grouping_method
        / "mining_anomaly"
        / "train_alert_groups.json"
    )
    train_alert_groups_path.parent.mkdir(parents=True, exist_ok=True)
    save_alert_groups_json(train_groups, train_alert_groups_path)

    print(f"  Mining attribute schema on {run_tag} train split...")
    mining_result = run_alert_group_attribute_mining_job(
        alert_groups_path=train_alert_groups_path,
        scenario_name=scenario,
        run_name=f"ait_ads_mining_anomaly_{run_tag}",
        config=AttributeMiningConfig(),
    )
    print(f"    Mined {len(mining_result.predicates)} predicates from train split.")

    symbolic_schema = build_symbolic_feature_schema(
        df=mining_result.mined_df,
        source_label="attack",
        schema_name=f"ait_ads_mining_anomaly_symbolic_{run_tag}",
        schema_version="0.1.0",
        predicates=mining_result.predicates,
    )
    print(f"    Built {len(symbolic_schema.features)} symbolic features.")

    encoder = SymbolicFeatureEncoder(feature_schema=symbolic_schema)
    symbolic_train_df = encoder.transform(train_groups)
    symbolic_test_df = encoder.transform(test_groups)

    print(f"  Training on {len(train_benign)} benign-only rows (natural count)")
    X_train = pd.concat(
        [
            train_benign[FEATURE_COLS].reset_index(drop=True),
            symbolic_train_df.iloc[train_benign.index].reset_index(drop=True),
        ],
        axis=1,
    ).values
    X_test = pd.concat(
        [
            test[FEATURE_COLS].reset_index(drop=True),
            symbolic_test_df.reset_index(drop=True),
        ],
        axis=1,
    ).values
    y_test = test["Label"].values

    # 5 seeds -- IsolationForest(random_state=seed), identical (base + mined)
    # benign rows every seed; default-cut metrics + tuned operating point per
    # seed, seed-averaged before saving.
    seed_metrics: list[dict[str, float]] = []
    seed_workloads: list[dict] = []
    for seed in range(5):
        model = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=seed, n_jobs=-1
        )
        model.fit(X_train)

        scores = -model.decision_function(X_test)  # higher = more anomalous
        y_pred = (model.predict(X_test) == -1).astype(int)  # 1 = anomaly = attack

        seed_metrics.append(
            {
                "auc": roc_auc_score(y_test, scores),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )
        seed_workloads.append(compute_workload_at_recall(y_test, scores))

    workload = average_workload_at_recall(seed_workloads)
    mean = {
        k: sum(m[k] for m in seed_metrics) / len(seed_metrics)
        for k in ("auc", "precision", "recall", "f1")
    }
    w90 = workload.get("0.90")
    print(
        f"  mean of 5 seeds: AUC={mean['auc']:.3f} P={mean['precision']:.3f} "
        f"R={mean['recall']:.3f} F1={mean['f1']:.3f} (default cut)"
        + (f"  |  @recall>=0.90: P={w90['precision']:.3f}" if w90 else "")
    )

    save_anomaly_results(
        name=result_name,
        description=(
            f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
            "5-column base schema + attribute-mined symbolic features "
            "(contrast-set + decision-tree rules, mined on the same train "
            "split), IsolationForest(n_estimators=100, contamination=0.05) "
            "fit on benign-only train rows, evaluated on the scenario's full "
            "test split. Test-side-only single-class guard -- IsolationForest "
            "sibling of ait_ads_mining_anomaly.py's OneClassSVM. Mean over 5 "
            "seeds (random_state=0..4); precision/recall/f1 at the default "
            "contamination=0.05 cut; workload_at_recall is the seed-averaged "
            "tuned-threshold view."
        ),
        seeds=seed_metrics,
        workload=workload,
    )


for _grouping_method in GROUPING_METHODS:
    for _scenario in SCENARIOS:
        try:
            run_scenario(_scenario, _grouping_method)
        except Exception:
            print(
                f"\n[ERROR] {_grouping_method}_{_scenario}: unhandled exception -- "
                "continuing to the next combo."
            )
            traceback.print_exc()
