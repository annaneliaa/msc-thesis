"""
Anomaly-detection counterpart to baselines/cscas_base.py: same raw CSV load,
paper-parity asserts, fixed timestamp split, 5-column reduced FEATURE_COLS,
and shared eval subsample -- but trains a one-class model (OneClassSVM, via
training/model_factory.get_model_factory("ocsvm")) on benign-only rows
instead of a RandomForestClassifier on a class-balanced pool.

This deliberately does not route through experiments/anomaly.py's
AlertGroup/FeatureSchemaRegistry pipeline (a different feature
representation entirely, and CSCAS isn't wired into it) -- it matches this
project's own cscas_*.py baseline convention instead, the same way
cscas_base.py/cscas_bert.py/cscas_zeroshot.py all do.

No pool-condition loop, no seeds: anomaly detection doesn't need class
balance (it's fit on benign rows only, natural count), and OneClassSVM
(kernel='rbf', nu=0.05) is a deterministic convex fit with no random_state
-- there is genuinely nothing to average over, so its per-seed sd is
exactly 0. The IsolationForest siblings (cscas_anomaly_iforest.py /
cscas_mining_anomaly_iforest.py) DO run the 5-seed protocol the trainable
baselines use, since tree bootstrapping is stochastic.

Scoring convention (matches experiments/anomaly.py::_compute_anomaly_metrics):
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack
AUC is this method's headline metric (score-based, threshold-free); F1/P/R
use the model's own -1/+1 decision boundary.

Run:
    cd src/thesis/baselines
    python cscas_anomaly.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from thesis.baselines._results import save_anomaly_results
from thesis.baselines._sampling import get_cscas_eval_subsample
from thesis.training.workload import compute_workload_at_recall

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

# 4) Define feature columns -- same reduced set as cscas_base.py.
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

# 5) Benign-only training data -- no pool conditions, no undersampling.
train_benign = train[train["Label"] == 0]
print(f"Training on {len(train_benign)} benign-only rows (natural count)")

# 6) Shared, frozen eval subsample (same as cscas_base.py).
eval_df = get_cscas_eval_subsample(test)
X_train = train_benign[FEATURE_COLS].values
X_test = eval_df[FEATURE_COLS].values
y_test = eval_df["Label"].values
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
)

# 7) Fit + score. Same estimator as model_factory's "ocsvm" (StandardScaler
# + OneClassSVM(kernel='rbf', nu=0.05)) -- built inline so this script
# doesn't pull in the classifier factory's heavier deps. Deterministic
# convex fit, no random_state: a genuine single run.
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

# Tuned-operating-point view: precision / FP / analyst-workload-reduction at
# the threshold that hits each target recall (default 0.90/0.95/0.99). The
# default nu=0.05 cut caps precision because it flags ~3x the true 1.5%
# attack prevalence -- this reports the model at a sensible cut instead.
workload = compute_workload_at_recall(y_test, scores)

print("\n=== cscas_anomaly_ocsvm ===")
print(f"AUC={auc:.3f} P={p:.3f} R={r:.3f} F1={f:.3f}  (default nu=0.05 cut)")
if workload.get("0.90"):
    w = workload["0.90"]
    print(
        f"  @recall>=0.90: P={w['precision']:.3f} FP={int(w['fp'])} "
        f"workload_reduction={w['workload_reduction']:.3f}"
    )

save_anomaly_results(
    name="cscas_anomaly_ocsvm",
    description=(
        "OneClassSVM(kernel='rbf', nu=0.05) fit on benign-only rows of "
        "this project's reduced base schema (5 features -- SignatureID, "
        "SCAS, and all Similarity columns removed, same as cscas_base.py), "
        "evaluated on the shared eval subsample. No attack rows used in "
        "training -- a workaround baseline for splits where train has zero "
        "attack examples (see _ait_ads_data.py's AIT-ADS counterpart). "
        "precision/recall/f1 are at the default nu=0.05 cut; "
        "workload_at_recall is the tuned-threshold view."
    ),
    metrics={"auc": auc, "precision": p, "recall": r, "f1": f},
    workload=workload,
)
