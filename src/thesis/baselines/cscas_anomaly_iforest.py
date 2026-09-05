"""
Second anomaly-detector model family alongside baselines/cscas_anomaly.py
(OneClassSVM) -- IsolationForest (via
training/model_factory.get_model_factory("iforest")) fit on benign-only
rows of the same reduced base schema, same split, same shared eval
subsample. Isolates model choice within the anomaly-detector family the
same way cscas_logreg.py/cscas_xgboost.py isolate model choice within the
classifier family, rather than treating "the anomaly baseline" as a single
fixed model.

Tree-based like RandomForestClassifier/XGBClassifier -- scale-invariant, so
unlike OneClassSVM's model_factory entry, "iforest" isn't wrapped in a
StandardScaler Pipeline (see model_factory.py's own comment on this).

No pool conditions (fit is benign-only, natural count -- nothing to
undersample). But unlike OneClassSVM (a deterministic convex fit, no
random_state), IsolationForest's tree bootstrap sampling is genuinely
stochastic, so this runs the same 5-seed protocol the trainable baselines
use -- IsolationForest(random_state=seed) for seed in range(5), fit on the
identical benign rows every seed -- and reports the seed mean plus the
per-seed breakdown so the comparison notebook can show mean +/- sd. The
OneClassSVM sibling (cscas_anomaly.py) stays a single run: its sd is
exactly 0.

Scoring convention identical to cscas_anomaly.py -- verified empirically
that IsolationForest exposes the same fit/decision_function/predict(-1/+1)
interface OneClassSVM does, so no scoring-logic changes were needed:
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack

Scoring convention identical to cscas_anomaly.py -- verified empirically
that IsolationForest exposes the same fit/decision_function/predict(-1/+1)
interface OneClassSVM does, so no scoring-logic changes were needed:
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack

Run:
    cd src/thesis/baselines
    python cscas_anomaly_iforest.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from thesis.baselines._results import save_anomaly_results
from thesis.baselines._sampling import get_cscas_eval_subsample
from thesis.training.workload import (
    average_workload_at_recall,
    compute_workload_at_recall,
)

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

# 7) Fit + score -- 5 seeds, IsolationForest(random_state=seed), identical
# benign training rows every seed (nothing to resample -- benign-only fit).
# Also collect the tuned-operating-point view per seed (precision / FP /
# workload-reduction at each target recall), seed-averaged before saving.
seed_metrics: list[dict[str, float]] = []
seed_workloads: list[dict] = []
for seed in range(5):
    model = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=seed, n_jobs=-1
    )
    model.fit(X_train)

    scores = -model.decision_function(X_test)  # higher = more anomalous
    y_pred = (model.predict(X_test) == -1).astype(int)  # 1 = anomaly = attack

    m = {
        "auc": roc_auc_score(y_test, scores),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    seed_metrics.append(m)
    seed_workloads.append(compute_workload_at_recall(y_test, scores))
    print(
        f"  seed={seed}: AUC={m['auc']:.3f} P={m['precision']:.3f} "
        f"R={m['recall']:.3f} F1={m['f1']:.3f}"
    )

workload = average_workload_at_recall(seed_workloads)

avg = pd.DataFrame(seed_metrics).mean()
print("\n=== cscas_anomaly_iforest (mean of 5 seeds) ===")
print(
    f"AUC={avg.auc:.3f} P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}  (default cut)"
)
if workload.get("0.90"):
    w = workload["0.90"]
    print(
        f"  @recall>=0.90: P={w['precision']:.3f} FP={w['fp']:.0f} "
        f"workload_reduction={w['workload_reduction']:.3f}"
    )

save_anomaly_results(
    name="cscas_anomaly_iforest",
    description=(
        "IsolationForest(n_estimators=100, contamination=0.05) fit on "
        "benign-only rows of this project's reduced base schema (5 "
        "features -- SignatureID, SCAS, and all Similarity columns "
        "removed, same as cscas_base.py), evaluated on the shared eval "
        "subsample. No attack rows used in training -- second anomaly-"
        "detector model family alongside cscas_anomaly.py's OneClassSVM. "
        "Mean over 5 seeds (random_state=0..4). precision/recall/f1 at the "
        "default contamination=0.05 cut; workload_at_recall is the tuned-"
        "threshold view (seed-averaged)."
    ),
    seeds=seed_metrics,
    workload=workload,
)
