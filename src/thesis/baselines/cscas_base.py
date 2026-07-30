"""
Same experimental setup as baselines/cscas.py (split, training-pool
sampling, classifier, seeds) -- the only things that change are (a)
FEATURE_COLS, swapped from the paper's own 42 raw columns to this
project's "base" schema (see encoders/baseline.py:
compute_cscas_baseline_features), which is the paper's feature set minus
the raw SignatureID column (kept out deliberately -- see docstring there),
and (b) the eval set: unlike cscas.py (the paper-replication anchor, which
stays on the full test set forever), this is an internal-system baseline,
so it scores all three conditions on the shared, frozen evaluation
subsample (see _sampling.get_cscas_eval_subsample) for a fair head-to-head
against the other non-replication baselines.

Run:
    cd src/thesis/baselines
    python cscas_base.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from thesis.baselines._results import save_baseline_results
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
# Dropped vs. the paper's own DROP_COLS (Timestamp, SignatureText, Label,
# ExtIP, IntIP): also drop SignatureID, since this project's base schema
# excludes it as a raw nominal identifier.
DROP_COLS = ["Timestamp", "SignatureText", "Label", "ExtIP", "IntIP", "SignatureID"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

# Sanity check: should be 41 columns (paper's 42 minus SignatureID) --
# SignatureMatchesPerDay, AlertCount, Proto, ExtPort, IntPort, Similarity,
# SCAS, SignatureIDSimilarity + 33 AttrSimilarity columns
assert len(FEATURE_COLS) == 41, f"got {len(FEATURE_COLS)}"
print(f"Feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

# 5) Verify training pools against Table IV (pool construction itself now
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval set -- shared, frozen subsample (not the full test set --
# that's reserved for the paper-replication script only).
eval_df = get_cscas_eval_subsample(test)
X_test = eval_df[FEATURE_COLS].values
y_test = eval_df["Label"].values
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
)

# 7) Three training-pool conditions
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

REFERENCE = {
    "random": "P=0.669, R=0.963, F1=0.789",
    "class_weighted": None,
    "guided": "P=0.868, R=0.952, F1=0.908",
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}

for condition, build_pool in POOL_BUILDERS.items():
    reference = REFERENCE[condition]
    print(f"\n=== {condition} (my 41-feature base schema) ===")
    if reference:
        print(f"    Paper reference (their 42 features incl. SignatureID): {reference}")

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
        y_pred = clf.predict(X_test)

        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        results[condition].append({"precision": p, "recall": r, "f1": f})
        print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

    avg = pd.DataFrame(results[condition]).mean()
    print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")


print(
    "\n=== Summary: paper (42 features, full test set) vs mine (41 features, no SignatureID, shared eval subsample) ==="
)
for condition, reference in REFERENCE.items():
    avg = pd.DataFrame(results[condition]).mean()
    ref_str = f"paper {reference}  |  " if reference else ""
    print(
        f"{condition:<16}"
        f"{ref_str}"
        f"mine P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )

save_baseline_results(
    name="cscas_base",
    description="This project's base schema (41 features, no SignatureID), RandomForestClassifier(n_estimators=100), evaluated on the shared eval subsample",
    results=results,
)
