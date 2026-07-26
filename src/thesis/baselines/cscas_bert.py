"""
Same experimental setup as baselines/cscas.py / cscas_base.py (temporal
split, training-pool sampling, seeds) -- what changes here is the model
(fine-tuned DistilBERT instead of RandomForestClassifier) and the input
representation (every non-similarity field serialized to text, instead of
numeric columns), per the supervisor-suggested "BERT is good at text"
baseline.

Each row is serialized into one text string, e.g.:
  "SignatureText: ET EXPLOIT D-Link ... | SignatureID: 12345 | Proto: 6 |
   ExtPort: 443 | IntPort: 8080 | AlertCount: 3 |
   SignatureMatchesPerDay: 1.2 | SCAS: 1"
The *Similarity columns (Similarity, SignatureIDSimilarity, and the 33
AttrValueSimilarity columns) are deliberately left out -- they're plain
numeric scores with no textual content, so serializing them to text gains
BERT nothing over just handing them to the RF baselines as numbers.

Fine-tuning happens on the same small, class-balanced training pools
CSCAS's own baselines use (not the full 139,532-row imbalanced train
split), because (a) it keeps fine-tuning cost tractable per seed and
(b) it keeps the comparison apples-to-apples with cscas.py/cscas_base.py,
which solve the imbalance problem the same way. Final evaluation is on the
full, untouched post-split_time test set, same as those scripts.

Run:
    cd src/thesis/baselines
    python cscas_bert.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.

Test-set inference (1.255M rows x 2 baselines x N_SEEDS models) is the
dominant cost here, not fine-tuning. Before committing to the full run, set
QUICK_SANITY_CHECK = True below for a cheap 1-seed / 20,000-row timing
check, then flip it back to False for real, reportable numbers.
"""

import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from thesis.baselines._results import save_baseline_results

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # bump if you see truncation warnings
N_SEEDS = 5
BATCH_SIZE = 8
NUM_EPOCHS = 3
VAL_FRAC_WITHIN_POOL = 0.15  # stratified carve-out from the training pool, for epoch-level eval/model selection only -- not the final test set

# Test-set inference over 1.255M rows x 2 baselines x N_SEEDS models is the
# dominant cost here (fine-tuning itself only ever sees ~3,530-row pools).
# Flip this on for a cheap timing sanity check -- runs a single seed per
# baseline against a small stratified sample of the test set instead of the
# full sweep -- before committing to the full run with QUICK_SANITY_CHECK=False.
QUICK_SANITY_CHECK = True
QUICK_SEEDS = 1
QUICK_TEST_SAMPLE_N = 20_000

# 1) Load and sort dataset

df = pd.read_csv("../../../data/cscas/dataset-labeled-anon-ip.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2) Verify dataset against paper's numbers
assert len(df) == 1_395_324, f"got {len(df)}"
assert df["Label"].sum() == 20_952, f"got {df['Label'].sum()}"
assert df["SCAS"].sum() == 72_672, f"got {df['SCAS'].sum()}"

# 3) Split into train and test sets based on timestamp -- identical boundary
# to cscas.py/cscas_base.py, so the final test set (and its row count) is the
# same across the paper baseline, the base-schema baseline, and this one.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# 4) Fields serialized into text. Dropped: Timestamp (used only for the
# split, not a per-row signal here), Label (the target), ExtIP/IntIP
# (anonymized per-connection identifiers -- same reasoning cscas.py already
# applies by excluding them from FEATURE_COLS), and every *Similarity column
# (plain numeric scores, no textual content -- see module docstring above).
DROP_COLS = ["Timestamp", "Label", "ExtIP", "IntIP"]
TEXT_FIELD_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]
assert (
    len(TEXT_FIELD_COLS) == 8
), f"got {len(TEXT_FIELD_COLS)}"  # SignatureText, SignatureID, SignatureMatchesPerDay, AlertCount, Proto, ExtPort, IntPort, SCAS
print(f"Fields serialized into text: {len(TEXT_FIELD_COLS)}")
print(TEXT_FIELD_COLS)


def build_text_column(frame: pd.DataFrame) -> pd.Series:
    """
    Serialize every TEXT_FIELD_COLS field into one "Key: value | Key: value"
    string per row. SignatureText goes first since it's the actual natural
    language and most valuable if MAX_LENGTH ever causes truncation.
    Vectorized (not .apply(axis=1)) since the test set has 1.25M rows.
    """
    parts = ["SignatureText: " + frame["SignatureText"].astype(str)]
    for col in TEXT_FIELD_COLS:
        if col == "SignatureText":
            continue
        parts.append(f"{col}: " + frame[col].astype(str))
    text = parts[0]
    for p in parts[1:]:
        text = text.str.cat(p, sep=" | ")
    return text


# 5) Build training pools (same as cscas.py)
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare the (large, fixed) test set once, outside the seed loop.
# In QUICK_SANITY_CHECK mode, evaluate against a small stratified sample
# instead of the full 1.255M rows -- these numbers are NOT comparable to the
# paper/RF baselines, they're only for timing.
if QUICK_SANITY_CHECK:
    print(
        f"[QUICK_SANITY_CHECK] Sampling {QUICK_TEST_SAMPLE_N} stratified rows "
        f"out of {len(test)} for a timing-only test set -- NOT for reporting."
    )
    test_eval, _ = train_test_split(
        test, train_size=QUICK_TEST_SAMPLE_N, stratify=test["Label"], random_state=0
    )
else:
    test_eval = test

print(
    f"Serializing test set to text ({len(test_eval)} rows"
    f"{' -- quick sanity check subset' if QUICK_SANITY_CHECK else ''})..."
)
test_df = pd.DataFrame(
    {
        "text": build_text_column(test_eval),
        "label": test_eval["Label"].astype(int).values,
    }
)
test_ds_base = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(ds: Dataset) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


print("Tokenizing test set...")
test_ds = tokenize(test_ds_base)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f = f1_score(labels, preds, zero_division=0)
    return {"precision": p, "recall": r, "f1": f}


def run_seed(pool: pd.DataFrame, seed: int) -> dict[str, float]:
    """
    Fine-tune DistilBERT from scratch on `pool`, evaluate once on the fixed
    CSCAS test set, and return precision/recall/f1 -- one point in the
    5-seed average, mirroring cscas.py's results_b1/results_b2 pattern.
    """
    t_start = time.time()
    pool_df = pd.DataFrame(
        {"text": build_text_column(pool), "label": pool["Label"].astype(int).values}
    )
    train_part, val_part = train_test_split(
        pool_df,
        test_size=VAL_FRAC_WITHIN_POOL,
        stratify=pool_df["label"],
        random_state=seed,
    )

    train_ds = tokenize(Dataset.from_pandas(train_part, preserve_index=False))
    val_ds = tokenize(Dataset.from_pandas(val_part, preserve_index=False))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"/tmp/cscas_bert_seed{seed}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    t_train_done = time.time()
    trainer.train()
    t_fit = time.time()
    test_result = trainer.predict(test_ds)
    t_predict = time.time()
    preds = np.argmax(test_result.predictions, axis=1)
    labels = test_result.label_ids

    print(
        f"    [timing] setup={t_train_done - t_start:.1f}s "
        f"fit={t_fit - t_train_done:.1f}s "
        f"test_inference={t_predict - t_fit:.1f}s "
        f"({len(test_ds)} test rows)"
    )

    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


SEEDS_TO_RUN = range(QUICK_SEEDS) if QUICK_SANITY_CHECK else range(N_SEEDS)

# 7) Baseline 1 = random undersampling
print("=== Baseline 1: Random undersampling (BERT, non-similarity fields as text) ===")
print("    Paper reference (RF, 42 numeric features): P=0.669, R=0.963, F1=0.789")

results_b1 = []
irrelevant = train[train["Label"] == 0]

for seed in SEEDS_TO_RUN:
    irr_sample = irrelevant.sample(n=1_765, random_state=seed)
    pool = pd.concat([important, irr_sample])

    metrics = run_seed(pool, seed)
    results_b1.append(metrics)
    print(
        f"  seed={seed}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )

avg_b1 = pd.DataFrame(results_b1).mean()
print(f"  AVERAGE: P={avg_b1.precision:.3f} R={avg_b1.recall:.3f} F1={avg_b1.f1:.3f}")


# 8) Baseline 2 = guided by CSCAS
print("\n=== Baseline 2: Guided by CSCAS (BERT, non-similarity fields as text) ===")
print("    Paper reference (RF, 42 numeric features): P=0.868, R=0.952, F1=0.908")

results_b2 = []

for seed in SEEDS_TO_RUN:
    irr_inl_sample = irr_inliers.sample(n=882, random_state=seed)
    irr_out_sample = irr_outliers.sample(n=883, random_state=seed)
    pool = pd.concat([important, irr_inl_sample, irr_out_sample])

    metrics = run_seed(pool, seed)
    results_b2.append(metrics)
    print(
        f"  seed={seed}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )

avg_b2 = pd.DataFrame(results_b2).mean()
print(f"  AVERAGE: P={avg_b2.precision:.3f} R={avg_b2.recall:.3f} F1={avg_b2.f1:.3f}")


if QUICK_SANITY_CHECK:
    print(
        f"\n[QUICK_SANITY_CHECK] Numbers below are from {QUICK_SEEDS} seed(s) on a "
        f"{QUICK_TEST_SAMPLE_N}-row test sample -- timing/smoke-test only, "
        "NOT comparable to the paper or the full run. Set QUICK_SANITY_CHECK=False for real numbers."
    )

print("\n=== Summary: paper (RF, 42 numeric features) vs BERT (8 fields as text) ===")
print(
    f"{'Baseline 1 (random undersampling)':<40}"
    f"paper P=0.669 R=0.963 F1=0.789  |  "
    f"BERT P={avg_b1.precision:.3f} R={avg_b1.recall:.3f} F1={avg_b1.f1:.3f}"
)
print(
    f"{'Baseline 2 (guided by CSCAS)':<40}"
    f"paper P=0.868 R=0.952 F1=0.908  |  "
    f"BERT P={avg_b2.precision:.3f} R={avg_b2.recall:.3f} F1={avg_b2.f1:.3f}"
)

if QUICK_SANITY_CHECK:
    print("[QUICK_SANITY_CHECK] Not saving results -- these numbers are timing-only.")
else:
    save_baseline_results(
        name="cscas_bert",
        description="8 non-similarity fields serialized to text, fine-tuned DistilBERT",
        results_b1=results_b1,
        results_b2=results_b2,
    )
