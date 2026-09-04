"""
AIT-ADS counterpart to baselines/cscas_anomaly.py, and the anomaly-detection
sibling of baselines/ait_ads_rf.py: OneClassSVM (via
training/model_factory.get_model_factory("ocsvm")) fit on benign-only rows
of the 5-column AIT-ADS base schema, evaluated per (grouping_method,
scenario).

Uses _ait_ads_data.load_ait_ads_baseline_split for both features and the
train/test split -- the same function ait_ads_rf.py/ait_ads_logreg.py/
ait_ads_xgboost.py/ait_ads_bert.py/ait_ads_securebert.py/ait_ads_zeroshot.py
all call, so this scores the identical rows those do for a given
(scenario, grouping_method).

Guard is TEST-SIDE ONLY (test["Label"].nunique() < 2), unlike every other
ait_ads_*.py script, which also requires train to contain both classes.
That train-side check is exactly what excludes harrison/santos (all
grouping methods) and russellmitchell (all but deepcase) from the
classifier baselines -- see _ait_ads_data.py's "Known, accepted exclusion"
docstring paragraph. Fitting a one-class model only ever needs benign rows
in train, so it was never blocked by that in the first place; this script
is the reason that exclusion doesn't need to be "fixed" elsewhere -- these
3 scenarios still get a baseline result here, just under this one method.

No pool conditions, no seeds -- same "single deterministic run" precedent
as cscas_anomaly.py/ait_ads_zeroshot.py.

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists, so a partial run followed by a restart doesn't redo
everything from scratch. Set AIT_ADS_FORCE=1 to force a full re-run.

Run:
    cd src/thesis/baselines
    python ait_ads_anomaly.py
"""

import os
import traceback

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import results_exist, save_anomaly_results
from thesis.configs import load_scenarios
from thesis.training.model_factory import get_model_factory

FEATURE_COLS = ["hour_of_day", "n_alerts", "n_sigs", "n_hosts", "n_shorts"]

# Skip a (grouping_method, scenario) combo whose results/*.json already
# exists -- set AIT_ADS_FORCE=1 to redo everything from scratch (matches
# scripts/system_eval/run_model_comparison_attribute.py's --force).
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
    result_name = f"ait_ads_anomaly_{run_tag}_ocsvm"
    print(
        f"\n{'=' * 70}\n  SCENARIO: {scenario} / GROUPING: {grouping_method}\n{'=' * 70}"
    )
    if not FORCE and results_exist(result_name):
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

    train, test = load_ait_ads_baseline_split(scenario, grouping_method=grouping_method)
    print(
        f"  {len(train)} train / {len(test)} test alert_groups, "
        f"{int(train['Label'].sum())} train positive, "
        f"{int(test['Label'].sum())} test positive"
    )
    # Test-side only: unlike the classifier baselines, this model is fit on
    # benign rows only, so a single-class *train* split (0 attack rows) is
    # completely fine -- see module docstring. AUC/F1/P/R still need both
    # classes present in test to be defined.
    if test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class test split.")
        return

    train_benign = train[train["Label"] == 0]
    if len(train_benign) == 0:
        print(f"  [skip] {run_tag}: no benign rows in train split.")
        return

    X_train = train_benign[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    y_test = test["Label"].values

    print(f"\n=== {run_tag} / anomaly (OneClassSVM, base schema) ===")
    print(f"  Training on {len(train_benign)} benign-only rows (natural count)")

    model = get_model_factory("ocsvm")()
    model.fit(X_train)

    scores = -model.decision_function(X_test)  # higher = more anomalous
    y_pred = (model.predict(X_test) == -1).astype(int)  # 1 = anomaly = attack

    auc = roc_auc_score(y_test, scores)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    print(f"  AUC={auc:.3f} P={p:.3f} R={r:.3f} F1={f:.3f}")

    save_anomaly_results(
        name=result_name,
        description=(
            f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
            "5-column base schema, OneClassSVM(kernel='rbf', nu=0.05) fit on "
            "benign-only train rows, evaluated on the scenario's full test "
            "split. Test-side-only single-class guard -- recovers scenarios "
            "excluded from the classifier baselines by the train-side guard "
            "(see _ait_ads_data.py's 'Known, accepted exclusion' paragraph)."
        ),
        metrics={"auc": auc, "precision": p, "recall": r, "f1": f},
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
