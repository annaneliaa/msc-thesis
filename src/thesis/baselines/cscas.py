"""
Reproduces the CSCAS paper's own two baselines (Table IV) -- random
undersampling and guided by CSCAS's SCAS outlier clusters -- using the
paper's own 42 raw feature columns and a RandomForestClassifier, averaged
over 5 seeds. Also runs a third, non-paper condition (class-weighted,
natural-ratio) alongside them, per the project's extended baseline design
(see Docs/Baselines.md).

All three conditions evaluate on the FULL test set, for all three -- this
script is the anchor replication target, and matching the paper's published
F1=0.908 (guided) requires exactly that protocol. The new class-weighted
condition has no published target to match; it just needs to run cleanly.

In addition to the paper's own 42-feature schema, this script also runs the
exact same replication protocol (same POOL_BUILDERS, same RF params, same
5 seeds, same eval sets) restricted to the reduced 5-feature base schema, so
the comparison table has a "Paper (reproduced), Base (5)" row alongside the
"Paper (reproduced), Full (42)" one -- both under the paper's own RF +
pool-sampling method, differing only in which features it sees. Note this
duplicates cscas_base.py's own computation (same code path, same numbers up
to seed order): that script produces the same base-schema/full-test cell
under the "Internal system" label; this one produces it again under the
"Paper (reproduced)" label, for scripts/notebook cells that want the full
replication story self-contained in this one file.

Run:
    cd src/thesis/baselines
    python cscas.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from thesis.baselines._cscas_schema import cscas_feature_cols, force_recompute
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)

# RandomForestClassifier here is CPU-only -- no GPU/device selection in this
# script -- printed for parity with the torch-based baselines' device line.
print("Using device: cpu")

# 1) Load and sort dataset

df = pd.read_csv("../../../data/cscas/dataset-labeled-anon-ip.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2) Verify dataset against papers numbers
assert len(df) == 1_395_324, f"got {len(df)}"
assert df["Label"].sum() == 20_952, f"got {df['Label'].sum()}"
assert df["SCAS"].sum() == 72_672, f"got {df['SCAS'].sum()}"

# 3) Split into train and test sets based on timestamp
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# 4) Define feature columns
DROP_COLS = ["Timestamp", "SignatureText", "Label", "ExtIP", "IntIP"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

# Sanity check: should be 42 columns
# SignatureID, SignatureMatchesPerDay, AlertCount, Proto,
# ExtPort, IntPort, Similarity, SCAS,
# + 34 AttrSimilarity columns
print(f"Feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

# 4b) The same replication protocol, restricted to the reduced 5-feature base
# schema (see module docstring). cscas_feature_cols is the same helper every
# other CSCAS baseline uses for this, so the column set is identical to
# cscas_base.py's.
FEATURE_COLS_BASE = cscas_feature_cols(df, schema="base")
assert len(FEATURE_COLS_BASE) == 5, f"got {len(FEATURE_COLS_BASE)}"
print(f"Base-schema feature count: {len(FEATURE_COLS_BASE)}")
print(FEATURE_COLS_BASE)

_ALL_OUTPUTS = (
    "cscas",
    "cscas_subsample",
    "cscas_repro_base",
    "cscas_repro_base_subsample",
)
if not force_recompute() and all(results_exist(n) for n in _ALL_OUTPUTS):
    print(f"[skip] {_ALL_OUTPUTS} already exist (CSCAS_FORCE=1 to re-run).")
    raise SystemExit(0)

# 5) Verify training pools against Table IV (pool construction itself now
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare test sets. The full test set is the replication protocol (see
# docstring). We ALSO score every fitted model on the shared 20k eval
# subsample -- same fitted models, a second .predict() -- so the 42-feature
# RF has a cell in the shared-subsample comparison grid the other baselines
# live in (-> results/cscas_subsample.json). The primary results/cscas.json
# output is unchanged.
X_test = test[FEATURE_COLS].values
y_test = test["Label"].values

eval_sub = get_cscas_eval_subsample(test)
X_sub = eval_sub[FEATURE_COLS].values
y_sub = eval_sub["Label"].values

# Same two eval sets, base-schema columns.
X_test_base = test[FEATURE_COLS_BASE].values
X_sub_base = eval_sub[FEATURE_COLS_BASE].values

# 7) Three training-pool conditions
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

TARGETS = {
    "random": "P=0.669, R=0.963, F1=0.789",
    "class_weighted": None,
    "guided": "P=0.868, R=0.952, F1=0.908",
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}
results_sub: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}
results_base: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}
results_base_sub: dict[str, list[dict[str, float]]] = {
    name: [] for name in POOL_BUILDERS
}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


for condition, build_pool in POOL_BUILDERS.items():
    target = TARGETS[condition]
    print(f"\n=== {condition} ===")
    if target:
        print(f"    Paper target: {target}")

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)

        X_tr = pool[FEATURE_COLS].values
        y_tr = pool["Label"].values

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
        clf.fit(X_tr, y_tr)

        m_full = _metrics(y_test, clf.predict(X_test))
        m_sub = _metrics(y_sub, clf.predict(X_sub))
        results[condition].append(m_full)
        results_sub[condition].append(m_sub)
        print(
            f"  seed={seed}: full  P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}"
            f"   |  subsample  P={m_sub['precision']:.3f} R={m_sub['recall']:.3f} F1={m_sub['f1']:.3f}"
        )

    avg = pd.DataFrame(results[condition]).mean()
    print(
        f"  AVERAGE (full test): P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )


# 7b) Same protocol again, base-schema columns -- a separate RF fit per
# (condition, seed), since a model fit on 42 columns can't predict from 5.
# build_pool(seed) is deterministic (sample(random_state=seed)), so this
# reuses the identical pools already built above, just re-sampled.
for condition, build_pool in POOL_BUILDERS.items():
    print(f"\n=== {condition} (base schema) ===")

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)

        X_tr_base = pool[FEATURE_COLS_BASE].values
        y_tr = pool["Label"].values

        clf_base = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
        clf_base.fit(X_tr_base, y_tr)

        m_full = _metrics(y_test, clf_base.predict(X_test_base))
        m_sub = _metrics(y_sub, clf_base.predict(X_sub_base))
        results_base[condition].append(m_full)
        results_base_sub[condition].append(m_sub)
        print(
            f"  seed={seed}: full  P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}"
            f"   |  subsample  P={m_sub['precision']:.3f} R={m_sub['recall']:.3f} F1={m_sub['f1']:.3f}"
        )

    avg = pd.DataFrame(results_base[condition]).mean()
    print(
        f"  AVERAGE (full test, base schema): P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )


# Scenario                          Expected P      Expected R  Expected F1
# random (undersampling)            0.669           0.963       0.789
# guided (by CSCAS)                 0.868           0.952       0.908
# class_weighted                    -- no published target --

save_baseline_results(
    name="cscas",
    description="Paper's own 42 raw features, RandomForestClassifier(n_estimators=100)",
    results=results,
)
save_baseline_results(
    name="cscas_subsample",
    description=(
        "Paper's own 42 raw features, RandomForestClassifier(n_estimators=100), "
        "scored on the shared frozen 20k eval subsample (same fitted models as "
        "cscas.json -- this is the 42-feature / subsample cell of the comparison grid)"
    ),
    results=results_sub,
)
save_baseline_results(
    name="cscas_repro_base",
    description=(
        "The paper's own replication protocol (same POOL_BUILDERS, RF params, "
        "5 seeds), restricted to the reduced 5-feature base schema instead of "
        "the paper's 42, scored on the full test set. Numerically identical to "
        "cscas_base_fulltest.json (same code path via cscas_base.py) -- kept "
        "here too so this script's own output covers both schemas."
    ),
    results=results_base,
)
save_baseline_results(
    name="cscas_repro_base_subsample",
    description=(
        "Same as cscas_repro_base, scored on the shared frozen 20k eval "
        "subsample instead of the full test set. Numerically identical to "
        "cscas_base.json."
    ),
    results=results_base_sub,
)
