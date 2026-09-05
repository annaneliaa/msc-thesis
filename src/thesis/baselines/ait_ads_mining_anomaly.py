"""
Anomaly-detection counterpart to baselines/ait_ads_mining.py, and the
mining sibling of baselines/ait_ads_anomaly.py: the same 5-column AIT-ADS
base schema extended with symbolic features mined by the attribute-mining
pipeline (contrast-set + decision-tree rules) on the train split, but fit
as a one-class OneClassSVM on benign-only rows instead of a
RandomForestClassifier on a class-balanced pool.

Uses _ait_ads_data.load_ait_ads_baseline_split_with_groups -- the same
group-returning loader ait_ads_mining.py uses, so this scores the
identical rows every other ait_ads_*.py script does for a given
(scenario, grouping_method).

Guard is TEST-SIDE ONLY (test["Label"].nunique() < 2), same as
ait_ads_anomaly.py -- not the train-side guard ait_ads_mining.py (and the
other five classifier/LLM scripts) use. This is the only mining-based
script that produces a result for harrison/santos/russellmitchell.

Mining still runs on the full train split (both classes, when present):
for the 3 scenarios above, train is 100% benign, so
run_alert_group_attribute_mining_job's contrast-set step has nothing to
contrast against and mines 0 predicates (verified empirically -- it
degrades gracefully rather than raising), so those 3 scenarios fall back
to the plain 5-column base schema here, same numbers ait_ads_anomaly.py
would produce. Every other scenario gets whatever mined features the
attack-vs-benign contrast in its train split actually supports.

No pool conditions, no seeds -- OneClassSVM is a deterministic convex fit
(per-seed sd exactly 0), same "single deterministic run" precedent as
ait_ads_anomaly.py/cscas_mining_anomaly.py. The IsolationForest sibling
(ait_ads_mining_anomaly_iforest.py) runs the 5-seed protocol. Each result
also stores `workload_at_recall` -- the tuned-threshold view (precision /
FP / analyst-workload-reduction at each target recall).

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists *and is in the current format* (has workload_at_recall, and
for the IsolationForest scripts a 5-seed breakdown) -- a stale-format
result left over from an older run is recomputed automatically, no need to
hand-delete it. Set AIT_ADS_FORCE=1 to force a full re-run.

Run:
    cd src/thesis/baselines
    python ait_ads_mining_anomaly.py
"""

import os
import traceback

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
from thesis.training.workload import compute_workload_at_recall

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
    result_name = f"ait_ads_mining_anomaly_{run_tag}_ocsvm"
    print(
        f"\n{'=' * 70}\n  SCENARIO: {scenario} / GROUPING: {grouping_method}\n{'=' * 70}"
    )
    if not FORCE and anomaly_results_current(result_name):
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
    # Test-side only -- see module docstring: unlike ait_ads_mining.py, a
    # 0-attack train split is fine here (the model is fit on benign rows
    # only regardless), and mining degrades to 0 predicates rather than
    # raising when train has no attack rows to contrast against.
    if test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class test split.")
        return

    train_benign = train[train["Label"] == 0]
    if len(train_benign) == 0:
        print(f"  [skip] {run_tag}: no benign rows in train split.")
        return

    # train.index equals positional row order (fresh reset_index(drop=True)
    # from load_ait_ads_baseline_split_with_groups), and train_groups is
    # aligned 1:1 by position with train -- train_benign.index below gives
    # positions into symbolic_train_df.
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

    # Same estimator as model_factory's "ocsvm", built inline to avoid the
    # classifier factory's heavier deps. Deterministic convex fit, no seed.
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", OneClassSVM(kernel="rbf", nu=0.05))]
    )
    model.fit(X_train)

    scores = -model.decision_function(X_test)  # higher = more anomalous
    y_pred = (model.predict(X_test) == -1).astype(int)  # 1 = anomaly = attack

    auc = roc_auc_score(y_test, scores)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    workload = compute_workload_at_recall(y_test, scores)
    w90 = workload.get("0.90")
    print(
        f"  AUC={auc:.3f} P={p:.3f} R={r:.3f} F1={f:.3f} (default nu=0.05 cut)"
        + (
            f"  |  @recall>=0.90: P={w90['precision']:.3f} FP={int(w90['fp'])}"
            if w90
            else ""
        )
    )

    save_anomaly_results(
        name=result_name,
        description=(
            f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
            "5-column base schema + attribute-mined symbolic features "
            "(contrast-set + decision-tree rules, mined on the same train "
            "split), OneClassSVM(kernel='rbf', nu=0.05) fit on benign-only "
            "train rows, evaluated on the scenario's full test split. "
            "Test-side-only single-class guard -- recovers scenarios "
            "excluded from ait_ads_mining.py by its train-side guard. "
            "precision/recall/f1 at the default nu=0.05 cut; "
            "workload_at_recall is the tuned-threshold view."
        ),
        metrics={"auc": auc, "precision": p, "recall": r, "f1": f},
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
