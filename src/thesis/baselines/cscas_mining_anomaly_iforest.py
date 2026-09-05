"""
IsolationForest counterpart to baselines/cscas_mining_anomaly.py: the same
5-feature reduced base schema extended with symbolic features mined by the
attribute-mining pipeline (contrast-set + decision-tree rules) on the train
split, but fit as a one-class IsolationForest on benign-only rows instead
of a OneClassSVM.

This is to cscas_mining_anomaly.py (OneClassSVM) what
cscas_anomaly_iforest.py is to cscas_anomaly.py -- it isolates model
choice within the anomaly-detector family for the *mining* scenario the
same way, rather than treating "the mining anomaly baseline" as a single
fixed model. Tree-based, so unlike OneClassSVM's model_factory entry
"iforest" isn't wrapped in a StandardScaler Pipeline.

Mining runs on the FULL train split (both classes) -- attack rows are still
needed to mine informative attack-vs-benign contrast predicates, even
though the model itself only ever fits on the benign subset of the
resulting (base + mined) feature matrix afterwards. Same
LEAKY_ATTRIBUTE_FIELDS exclusion, same train split, same in-memory schema,
same cache namespace as cscas_mining_anomaly.py, so the mined predicates
are shared between the two scripts.

No pool-condition loop, no seeds -- same "single deterministic run"
precedent as cscas_anomaly.py / cscas_mining_anomaly.py. IsolationForest's
own randomness is pinned by its fixed random_state=42 in model_factory.py.

Scoring convention (matches cscas_anomaly.py / cscas_mining_anomaly.py):
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack

Run:
    cd src/thesis/baselines
    python cscas_mining_anomaly_iforest.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from thesis.baselines._results import save_anomaly_results
from thesis.baselines._sampling import get_cscas_eval_subsample
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import rows_to_cscas_alert_groups, save_alert_groups_json
from thesis.schemas.mining import AttributeMiningConfig
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS
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

# 3) Split into train and test sets based on timestamp -- identical to
# cscas_mining.py/cscas_base.py's split.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# train is an unbroken 0-based prefix slice of df's own 0..N-1 RangeIndex
# (post reset_index(drop=True) above), so its index labels equal positional
# row order -- train_benign.index below gives positions into
# symbolic_train_df because of this invariant.
assert list(train.index) == list(range(len(train)))

# 4) Define feature columns -- same reduced base schema as cscas_base.py.
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
print(f"Base feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

# 5) Shared, frozen eval subsample (same as cscas_base.py/cscas_mining.py).
eval_df = get_cscas_eval_subsample(test)
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
)

# 6) Build AlertGroups for train and eval -- same per-row parser
# cscas_mining.py uses.
print("Building AlertGroups for train/eval splits...")
train_groups = rows_to_cscas_alert_groups(train.to_dict("records"))
eval_groups = rows_to_cscas_alert_groups(eval_df.to_dict("records"))
assert len(train_groups) == len(train), "row parsing dropped rows -- alignment broken"
assert len(eval_groups) == len(eval_df), "row parsing dropped rows -- alignment broken"

train_alert_groups_path = (
    CACHE_DIR
    / "cscas"
    / "groups"
    / "cscas_pregrouped_baseline_mining_anomaly"
    / "alert_groups"
    / "train_alert_groups.json"
)
train_alert_groups_path.parent.mkdir(parents=True, exist_ok=True)
save_alert_groups_json(train_groups, train_alert_groups_path)

# 7) Mine symbolic features on the FULL train split (both classes -- attack
# rows are needed to mine informative contrast predicates even though the
# model below only ever fits on the benign subset), excluding SCAS/
# Similarity-derived candidate fields -- same reasoning as cscas_mining.py.
# Same run_name/config/exclude_fields as cscas_mining_anomaly.py, so the
# mined predicates are cache-shared between the two scripts.
LEAKY_ATTRIBUTE_FIELDS = {
    "scas",
    "similarity",
    "signature_id_similarity",
    *(f"attr_value:{n}" for n in ATTR_SIMILARITY_COLUMNS),
    *(f"attr_populated:{n}" for n in ATTR_SIMILARITY_COLUMNS),
    *(
        f"applicable_layer:{p.lower()}"
        for p in ("Dns", "Email", "Http", "Smtp", "Ssh", "Tls")
    ),
}

print("Mining attribute schema on train split...")
mining_result = run_alert_group_attribute_mining_job(
    alert_groups_path=train_alert_groups_path,
    scenario_name="cscas",
    run_name="cscas_baseline_mining_anomaly",
    config=AttributeMiningConfig(),
    exclude_fields=LEAKY_ATTRIBUTE_FIELDS,
)
print(f"  Mined {len(mining_result.predicates)} predicates from train split.")

symbolic_schema = build_symbolic_feature_schema(
    df=mining_result.mined_df,
    source_label="attack",
    schema_name="cscas_mining_anomaly_symbolic",
    schema_version="0.1.0",
    predicates=mining_result.predicates,
)
print(f"  Built {len(symbolic_schema.features)} symbolic features.")

encoder = SymbolicFeatureEncoder(feature_schema=symbolic_schema)
symbolic_train_df = encoder.transform(train_groups)
symbolic_eval_df = encoder.transform(eval_groups)

# 8) Benign-only training data -- no pool conditions, no undersampling.
train_benign = train[train["Label"] == 0]
print(f"Training on {len(train_benign)} benign-only rows (natural count)")

# train_benign.index gives positions into symbolic_train_df (see step 3's
# invariant).
X_train = pd.concat(
    [
        train_benign[FEATURE_COLS].reset_index(drop=True),
        symbolic_train_df.iloc[train_benign.index].reset_index(drop=True),
    ],
    axis=1,
).values
X_test = pd.concat(
    [
        eval_df[FEATURE_COLS].reset_index(drop=True),
        symbolic_eval_df.reset_index(drop=True),
    ],
    axis=1,
).values
y_test = eval_df["Label"].values

# 9) Fit + score -- 5 seeds, IsolationForest(random_state=seed), identical
# (base + mined) benign training rows every seed (nothing to resample).
# Also collect the tuned-operating-point view per seed, seed-averaged before
# saving.
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
print("\n=== cscas_mining_anomaly_iforest (mean of 5 seeds) ===")
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
    name="cscas_mining_anomaly_iforest",
    description=(
        "IsolationForest(n_estimators=100, contamination=0.05) fit on "
        "benign-only rows of the base schema (5 features) + attribute-mined "
        "symbolic features (contrast-set + decision-tree rules, mined on "
        "the same train split as cscas_mining; SCAS/Similarity-derived "
        "fields excluded from mining), evaluated on the shared eval "
        "subsample. No attack rows used in training -- IsolationForest "
        "sibling of cscas_mining_anomaly.py's OneClassSVM. Mean over 5 "
        "seeds (random_state=0..4). precision/recall/f1 at the default "
        "contamination=0.05 cut; workload_at_recall is the tuned-threshold "
        "view (seed-averaged)."
    ),
    seeds=seed_metrics,
    workload=workload,
)
