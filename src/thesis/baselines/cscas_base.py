"""
Same experimental setup as baselines/cscas.py (split, training-pool
sampling, classifier, seeds) -- the only thing that changes is
FEATURE_COLS, swapped from the paper's own 42 raw columns to this
project's "base" schema (see encoders/baseline.py:
compute_cscas_baseline_features), which is the paper's feature set minus
the raw SignatureID column (kept out deliberately -- see docstring there).

Run:
    cd src/thesis/baselines
    python cscas_base.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

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

# 5) Build training pools
# Important points (Label=1) — all of them
important = train[train["Label"] == 1]

# Irrelevant inliers (Label=0, SCAS=0)
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]

# Irrelevant outliers (Label=0, SCAS=1)
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

# Verify against Table IV
assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare test set
X_test = test[FEATURE_COLS].values
y_test = test["Label"].values

# 7) Baseline 1 = random undersampling
print("=== Baseline 1: Random undersampling (my 41-feature base schema) ===")
print(
    "    Paper reference (their 42 features incl. SignatureID): P=0.669, R=0.963, F1=0.789"
)

results_b1 = []
irrelevant = train[train["Label"] == 0]

for seed in range(5):
    irr_sample = irrelevant.sample(n=1_765, random_state=seed)
    sample = pd.concat([important, irr_sample])

    X_tr = sample[FEATURE_COLS].values
    y_tr = sample["Label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test)

    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    results_b1.append({"precision": p, "recall": r, "f1": f})
    print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

avg_b1 = pd.DataFrame(results_b1).mean()
print(
    f"  AVERAGE: P={avg_b1.precision:.3f} " f"R={avg_b1.recall:.3f} F1={avg_b1.f1:.3f}"
)


# 8) Run baseline 2 = guided by CSCAS
print("\n=== Baseline 2: Guided by CSCAS (my 41-feature base schema) ===")
print(
    "    Paper reference (their 42 features incl. SignatureID): P=0.868, R=0.952, F1=0.908"
)

results_b2 = []

for seed in range(5):
    irr_inl_sample = irr_inliers.sample(n=882, random_state=seed)
    irr_out_sample = irr_outliers.sample(n=883, random_state=seed)
    sample = pd.concat([important, irr_inl_sample, irr_out_sample])

    X_tr = sample[FEATURE_COLS].values
    y_tr = sample["Label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test)

    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    results_b2.append({"precision": p, "recall": r, "f1": f})
    print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

avg_b2 = pd.DataFrame(results_b2).mean()
print(
    f"  AVERAGE: P={avg_b2.precision:.3f} " f"R={avg_b2.recall:.3f} F1={avg_b2.f1:.3f}"
)


print("\n=== Summary: paper (42 features) vs mine (41 features, no SignatureID) ===")
print(
    f"{'Baseline 1 (random undersampling)':<40}"
    f"paper P=0.669 R=0.963 F1=0.789  |  "
    f"mine P={avg_b1.precision:.3f} R={avg_b1.recall:.3f} F1={avg_b1.f1:.3f}"
)
print(
    f"{'Baseline 2 (guided by CSCAS)':<40}"
    f"paper P=0.868 R=0.952 F1=0.908  |  "
    f"mine P={avg_b2.precision:.3f} R={avg_b2.recall:.3f} F1={avg_b2.f1:.3f}"
)
