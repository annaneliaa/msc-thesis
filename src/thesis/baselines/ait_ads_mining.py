"""
AIT-ADS counterpart to baselines/cscas_mining.py, and the mining sibling of
baselines/ait_ads_rf.py: the same 5-column AIT-ADS base schema, extended
with symbolic features mined by the attribute-mining pipeline (contrast-set
+ decision-tree rules, see thesis.mining.attribute_mining_job) on the SAME
train split, evaluated per (grouping_method, scenario).

Uses _ait_ads_data.load_ait_ads_baseline_split_with_groups -- the
group-returning variant of the same load_ait_ads_baseline_split every other
ait_ads_*.py script calls, so this scores the identical rows those do for a
given (scenario, grouping_method); the extra `train_groups`/`test_groups`
it returns are only needed here, to hand actual AlertGroup objects to
run_alert_group_attribute_mining_job (mining works over AlertGroup
attributes, not the encoded base-schema DataFrame columns).

No `exclude_fields` -- unlike cscas_mining.py, which excludes SCAS/
Similarity-derived candidate fields because CSCAS's raw dataset carries
those as label-adjacent, offline-computed columns. AIT-ADS's AlertGroup
attributes have no such leakage-prone fields (same reasoning
run_model_comparison_attribute.py already applies for the real AIT-ADS
mining experiments -- no exclude_fields there either), so mining draws on
everything mineable from the train-split AlertGroups directly.

The mined symbolic schema is built and used purely in-memory here (via
build_symbolic_feature_schema + SymbolicFeatureEncoder), same as
cscas_mining.py -- not through mine_or_reuse_attribute_schema's on-disk
registry, which is shared with real experiments on these scenarios.

The mined feature matrix is fit with all three tabular classifiers -- the
mining counterparts of ait_ads_rf.py / ait_ads_logreg.py / ait_ads_xgboost.py
-- and saved as ait_ads_mining_<run_tag> (RF, name unchanged),
ait_ads_mining_logreg_<run_tag> and ait_ads_mining_xgboost_<run_tag>.
AIT-ADS's base schema has no -1 sentinel, so LogReg just needs a plain
StandardScaler (unlike cscas_mining.py) -- see ait_ads_logreg.py's docstring.

Two training-pool conditions, not three -- same as ait_ads_rf.py: no
"guided" (CSCAS-only, no SCAS-equivalent outlier signal for AIT-ADS).

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: a (grouping_method, scenario) combo is skipped entirely (no
mining pass) only when all three model results already exist; if some are
missing the mining pass runs once and only the missing models are fit and
saved -- so adding LogReg/XGBoost to a tree already carrying RF results does
not recompute or overwrite the RF JSON. Set AIT_ADS_FORCE=1 to force a full
re-run.

Run:
    cd src/thesis/baselines
    python ait_ads_mining.py
"""

import os
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split_with_groups,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import class_weighted_pool, random_undersample_pool
from thesis.configs import load_scenarios
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import save_alert_groups_json
from thesis.schemas.mining import AttributeMiningConfig

FEATURE_COLS = ["hour_of_day", "n_alerts", "n_sigs", "n_hosts", "n_shorts"]
N_SEEDS = 5  # same seed count as ait_ads_rf.py

# The mined feature matrix is fit with all three tabular classifiers -- the
# mining counterparts of ait_ads_rf.py / ait_ads_logreg.py / ait_ads_xgboost.py.
# "ait_ads_mining_<run_tag>" stays the RF result (name unchanged); the two
# new ones get a model suffix. AIT-ADS's base schema has no -1 sentinel (see
# ait_ads_logreg.py's docstring), so LogReg just needs a plain StandardScaler
# -- the mined symbolic columns are binary and pass through unchanged.
MINING_MODELS = {
    "rf": "RandomForestClassifier(n_estimators=100)",
    "logreg": "StandardScaler + LogisticRegression",
    "xgboost": "XGBClassifier(n_estimators=100)",
}


def build_classifier(model: str, seed: int, extra_kwargs: dict):
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=extra_kwargs.get("scale_pos_weight"),
        )
    return Pipeline(
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


def _result_name(model: str, run_tag: str) -> str:
    return (
        f"ait_ads_mining_{run_tag}"
        if model == "rf"
        else f"ait_ads_mining_{model}_{run_tag}"
    )


def run_scenario(scenario: str, grouping_method: str) -> None:
    run_tag = f"{grouping_method}_{scenario}"
    result_names = {m: _result_name(m, run_tag) for m in MINING_MODELS}
    print(
        f"\n{'=' * 70}\n  SCENARIO: {scenario} / GROUPING: {grouping_method}\n{'=' * 70}"
    )
    if not FORCE and all(results_exist(n) for n in result_names.values()):
        print(
            f"  [skip] {run_tag}: all mining-model results already exist "
            "(set AIT_ADS_FORCE=1 to re-run)."
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
    if train["Label"].nunique() < 2 or test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class train or test split.")
        return

    important = train[train["Label"] == 1]
    if len(important) == 0:
        print(f"  [skip] {run_tag}: no positive (attack) rows in train split.")
        return

    # train.index equals positional row order (fresh reset_index(drop=True)
    # from load_ait_ads_baseline_split_with_groups), and train_groups is
    # aligned 1:1 by position with train -- pool.index below gives positions
    # into symbolic_train_df, same invariant cscas_mining.py relies on.
    assert list(train.index) == list(range(len(train)))

    train_alert_groups_path = (
        CACHE_DIR
        / scenario
        / "groups"
        / grouping_method
        / "mining"
        / "train_alert_groups.json"
    )
    train_alert_groups_path.parent.mkdir(parents=True, exist_ok=True)
    save_alert_groups_json(train_groups, train_alert_groups_path)

    print(f"  Mining attribute schema on {run_tag} train split...")
    mining_result = run_alert_group_attribute_mining_job(
        alert_groups_path=train_alert_groups_path,
        scenario_name=scenario,
        run_name=f"ait_ads_mining_{run_tag}",
        config=AttributeMiningConfig(),
    )
    print(f"    Mined {len(mining_result.predicates)} predicates from train split.")

    symbolic_schema = build_symbolic_feature_schema(
        df=mining_result.mined_df,
        source_label="attack",
        schema_name=f"ait_ads_mining_symbolic_{run_tag}",
        schema_version="0.1.0",
        predicates=mining_result.predicates,
    )
    print(f"    Built {len(symbolic_schema.features)} symbolic features.")

    encoder = SymbolicFeatureEncoder(feature_schema=symbolic_schema)
    symbolic_train_df = encoder.transform(train_groups)
    symbolic_test_df = encoder.transform(test_groups)

    X_test = pd.concat(
        [
            test[FEATURE_COLS].reset_index(drop=True),
            symbolic_test_df.reset_index(drop=True),
        ],
        axis=1,
    ).values
    y_test = test["Label"].values

    pool_builders = {
        "random": lambda seed: random_undersample_pool(train, important, seed),
        "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    }

    # One mining pass above feeds all three classifiers; the base + mined
    # feature matrix is model-independent (LogReg's StandardScaler lives
    # inside its Pipeline), so X_test/X_tr are built once.
    for model, model_desc in MINING_MODELS.items():
        result_name = result_names[model]
        if not FORCE and results_exist(result_name):
            print(f"\n  [skip] {result_name}.json already exists.")
            continue

        results: dict[str, list[dict[str, float]]] = {
            name: [] for name in pool_builders
        }

        for condition, build_pool in pool_builders.items():
            print(f"\n=== {run_tag} / {model} / {condition} (base schema + mining) ===")
            for seed in range(N_SEEDS):
                pool, extra_kwargs = build_pool(seed)
                X_tr = np.hstack(
                    [
                        pool[FEATURE_COLS].values,
                        symbolic_train_df.iloc[pool.index].values,
                    ]
                )
                y_tr = pool["Label"].values

                clf = build_classifier(model, seed, extra_kwargs)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_test)

                p = precision_score(y_test, y_pred, zero_division=0)
                r = recall_score(y_test, y_pred, zero_division=0)
                f = f1_score(y_test, y_pred, zero_division=0)
                results[condition].append({"precision": p, "recall": r, "f1": f})
                print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

            avg = pd.DataFrame(results[condition]).mean()
            print(
                f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
            )

        save_baseline_results(
            name=result_name,
            description=(
                f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
                "5-column base schema + attribute-mined symbolic features "
                "(contrast-set + decision-tree rules, mined on the same train "
                f"split), {model_desc}, random + class-weighted training-pool "
                "conditions (no 'guided' -- CSCAS-only), evaluated on the "
                "scenario's full test split"
            ),
            results=results,
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
