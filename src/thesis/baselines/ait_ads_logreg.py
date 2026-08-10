"""
Same experimental setup as ait_ads_rf.py (data loading, split, per-
(grouping_method, scenario) loop, two training-pool conditions) -- swaps
RandomForestClassifier for LogisticRegression, matching cscas_logreg.py's
relationship to cscas_base.py.

Unlike cscas_logreg.py, no missingness-flag handling is needed here: CSCAS's
base schema has a `-1` "not applicable" sentinel in 3 of its 5 columns (see
cscas_logreg.py's module docstring); AIT-ADS's base schema
(hour_of_day/n_alerts/n_sigs/n_hosts/n_shorts, from
encoders/baseline.py's compute_ait_ads_baseline_features) has no such
sentinel -- every value is a real count or hour, so a plain StandardScaler
on all 5 columns is sufficient.

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists, so a partial run followed by a restart doesn't redo
everything from scratch. Set AIT_ADS_FORCE=1 to force a full re-run.

Run:
    cd src/thesis/baselines
    python ait_ads_logreg.py
"""

import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import class_weighted_pool, random_undersample_pool
from thesis.configs import load_scenarios

FEATURE_COLS = ["hour_of_day", "n_alerts", "n_sigs", "n_hosts", "n_shorts"]
N_SEEDS = 5

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
    result_name = f"ait_ads_logreg_{run_tag}"
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
    if train["Label"].nunique() < 2 or test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class train or test split.")
        return

    important = train[train["Label"] == 1]
    if len(important) == 0:
        print(f"  [skip] {run_tag}: no positive (attack) rows in train split.")
        return

    X_test = test[FEATURE_COLS].values
    y_test = test["Label"].values

    pool_builders = {
        "random": lambda seed: random_undersample_pool(train, important, seed),
        "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    }

    results: dict[str, list[dict[str, float]]] = {name: [] for name in pool_builders}

    for condition, build_pool in pool_builders.items():
        print(f"\n=== {run_tag} / {condition} (LogisticRegression, base schema) ===")
        for seed in range(N_SEEDS):
            pool, extra_kwargs = build_pool(seed)
            X_tr = pool[FEATURE_COLS].values
            y_tr = pool["Label"].values

            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=seed,
                            class_weight=extra_kwargs.get("class_weight"),
                        ),
                    ),
                ]
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_test)

            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f = f1_score(y_test, y_pred, zero_division=0)
            results[condition].append({"precision": p, "recall": r, "f1": f})
            print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

        avg = pd.DataFrame(results[condition]).mean()
        print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")

    save_baseline_results(
        name=result_name,
        description=(
            f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
            "5-column base schema, StandardScaler + LogisticRegression, "
            "random + class-weighted training-pool conditions (no 'guided' -- "
            "CSCAS-only, no SCAS-equivalent signal for AIT-ADS), evaluated on "
            "the scenario's full test split"
        ),
        results=results,
    )


for _grouping_method in GROUPING_METHODS:
    for _scenario in SCENARIOS:
        run_scenario(_scenario, _grouping_method)
