"""
Same experimental setup as cscas_base.py (base schema, training-pool
sampling, shared eval subsample, 5 seeds) -- swaps RandomForestClassifier
for LogisticRegression, the "standard interpretable linear floor" in the
project's baseline design (see Docs/Baselines.md).

Unlike RF/XGBoost, LogisticRegression is NOT scale-invariant, so this
script needs two things neither of the tree-based scripts does:

  1. StandardScaler, fit on each seed's training pool only (never on the
     eval subsample -- that would leak eval statistics into training) and
     then applied to both that pool and the eval subsample.

  2. A decision on how to treat CSCAS's `-1` "not applicable" sentinel.
     It appears in 3 of the reduced base schema's 5 feature columns (Proto,
     ExtPort, IntPort -- e.g. ExtPort is -1 whenever a protocol has no
     notion of a port). Scaling -1 in place would conflate "structurally
     not applicable" with "very dissimilar" on the same continuous axis,
     which distorts StandardScaler's fitted mean/std for columns where -1
     is a sizable share of rows (RF/XGBoost don't have this problem -- tree
     splits just treat -1 as a very low value, no distortion). Decision: add one binary
     `{col}_missing` indicator column per sentinel-bearing column,
     impute the sentinel to 0 in the original column, then scale
     everything (imputed values + flags) together. This lets the linear
     model separate "this alert type never has this field" (the flag)
     from the actual similarity signal on rows where it IS applicable.
     Which columns carry the sentinel is determined once from the full
     training set (not per-pool), so the augmented feature schema is
     fixed across every seed/condition and matches the eval subsample.

Run:
    cd src/thesis/baselines
    python cscas_logreg.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from thesis.baselines._results import save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)

# LogisticRegression here is CPU-only -- no GPU/device selection in this
# script -- printed for parity with the torch-based baselines' device line.
print("Using device: cpu")

SENTINEL_VALUE = -1

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

# 4) Reduced base schema -- same 5 columns as cscas_base.py (see that
# module's docstring for what's dropped and why: SignatureID, SCAS, and
# every *Similarity column, all unrealistic for a real deployment).
DROP_COLS = [
    "Timestamp",
    "SignatureText",
    "Label",
    "ExtIP",
    "IntIP",
    "SignatureID",
    "SCAS",
]
FEATURE_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]
assert len(FEATURE_COLS) == 5, f"got {len(FEATURE_COLS)}"
print(f"Feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

# 4b) Columns carrying the -1 sentinel, determined once from the full
# training set -- see module docstring.
SENTINEL_COLS = [c for c in FEATURE_COLS if (train[c] == SENTINEL_VALUE).any()]
print(f"Columns with -1 sentinel in train: {len(SENTINEL_COLS)} of {len(FEATURE_COLS)}")

MODEL_FEATURE_COLS = FEATURE_COLS + [f"{c}_missing" for c in SENTINEL_COLS]


def add_missingness_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a {col}_missing indicator per SENTINEL_COLS column and impute
    the sentinel to 0 in place. Deterministic elementwise transform (no
    statistics learned from data), so safe to apply identically to every
    pool and the eval subsample with no leakage risk."""
    frame = frame.copy()
    for col in SENTINEL_COLS:
        is_missing = frame[col] == SENTINEL_VALUE
        frame[f"{col}_missing"] = is_missing.astype(int)
        frame.loc[is_missing, col] = 0.0
    return frame


# 5) Verify training pools against Table IV (pool construction itself
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval set -- shared, frozen subsample, missingness flags
# applied once outside the seed loop (deterministic transform).
eval_df = add_missingness_flags(get_cscas_eval_subsample(test))
X_test = eval_df[MODEL_FEATURE_COLS].values
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
    print(
        f"\n=== {condition} (LogisticRegression, reduced base schema + missingness flags) ==="
    )
    if reference:
        print(
            f"    Paper reference (RF, 42 numeric features, full test set): {reference}"
        )

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)
        pool_enc = add_missingness_flags(pool)

        X_tr = pool_enc[MODEL_FEATURE_COLS].values
        y_tr = pool_enc["Label"].values

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

        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        results[condition].append({"precision": p, "recall": r, "f1": f})
        print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

    avg = pd.DataFrame(results[condition]).mean()
    print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")


print(
    "\n=== Summary: paper (RF, 42 features, full test set) vs "
    "LogReg (reduced base schema + missingness flags, shared eval subsample) ==="
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
    name="cscas_logreg",
    description=(
        "Reduced base schema (5 features -- SignatureID, SCAS, and all "
        "Similarity columns removed as unrealistic for a real deployment -- "
        "plus missingness flags for -1 sentinel columns), StandardScaler, "
        "LogisticRegression, evaluated on the shared eval subsample"
    ),
    results=results,
)
