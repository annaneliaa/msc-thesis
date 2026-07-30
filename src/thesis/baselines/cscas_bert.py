"""
Same experimental setup as baselines/cscas.py / cscas_base.py (temporal
split, shared eval subsample) -- what changes here is the model (fine-tuned
DistilBERT instead of RandomForestClassifier) and the input representation
(every non-similarity field serialized to text, instead of numeric
columns), per the supervisor-suggested "BERT is good at text" baseline.

Each row is serialized into one text string, e.g.:
  "SignatureText: ET EXPLOIT D-Link ... | Proto: 6 | ExtPort: 443 |
   IntPort: 8080 | AlertCount: 3 | SignatureMatchesPerDay: 1.2"
Left out, same reduced-feature-set reasoning as cscas_base.py's module
docstring (none of these are things a real deployment could compute for a
fresh alert without already knowing the answer or running CSCAS's offline
pipeline): the *Similarity columns (Similarity, SignatureIDSimilarity, and
the 33 AttrValueSimilarity columns -- plain numeric scores with no textual
content anyway, so serializing them gains BERT nothing over handing them
to the RF baselines as numbers), SignatureID (nominal identifier), and
SCAS (the paper's own outlier/inlier flag).

Same three training-pool conditions as the RF/LogReg/XGBoost baselines --
random undersampling, class-weighted (natural-ratio), guided by CSCAS --
via the same POOL_BUILDERS pattern cscas_base.py uses. This is
N_SEEDS x 3 fine-tuning runs (15 at the default N_SEEDS=5), noticeably more
GPU time than the class-weighted-only version this script started as; see
git history if you want the original single-condition scope and its
rationale.

Unlike sklearn's `class_weight='balanced'`, fine-tuning needs an explicit
weighted cross-entropy loss -- see WeightedLossTrainer below. Only the
class-weighted condition actually needs this: random/guided pools are
already ~balanced by construction (that's the point of undersampling), so
run_seed only builds class_weights when the pool's own extra_kwargs say
`class_weight == "balanced"` -- same conditional cscas_base.py uses for
sklearn's `class_weight` param.

CLASS_WEIGHTED_POOL_CAP (below) can be set (directly, or via the
CSCAS_CLASS_WEIGHTED_POOL_CAP env var) to bound fine-tuning cost -- left
uncapped (None) by default, the full 139,532-row natural-ratio pool is
used, and a single seed at that size measured well over an hour of fit
time alone.

Evaluates on the shared, frozen eval subsample (see
_sampling.get_cscas_eval_subsample) -- NOT the full 1.255M-row test set.
This matches every other non-replication baseline (only cscas.py/
cscas_base.py's paper-adjacent full-test-set eval is exempt).

Run:
    cd src/thesis/baselines
    python cscas_bert.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.

Before committing to the full N_SEEDS-seed run, set QUICK_SANITY_CHECK =
True below for a cheap 1-seed timing/smoke check (still against the real
shared eval subsample and the real pool, just fewer seeds and not saved),
then flip it back to False for real, reportable numbers.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from thesis.baselines._results import save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # bump if you see truncation warnings
N_SEEDS = 5  # same seed count as the RF/LogReg/XGBoost baselines
BATCH_SIZE = 8
NUM_EPOCHS = 3
VAL_FRAC_WITHIN_POOL = 0.15  # stratified carve-out from the training pool, for epoch-level eval/model selection only -- not the final eval set
# Both overridable via env var so an unattended/overnight run can flip them
# without hand-editing this file -- CSCAS_CLASS_WEIGHTED_POOL_CAP=15000,
# CSCAS_QUICK_SANITY_CHECK=0. Unset envs keep the safe interactive defaults
# below (uncapped, quick-check on).
_cap_env = os.environ.get("CSCAS_CLASS_WEIGHTED_POOL_CAP")
CLASS_WEIGHTED_POOL_CAP = (
    int(_cap_env) if _cap_env else None
)  # see module docstring -- open "size TBD" decision

# Fine-tuning always sees a small pool (~3,530 rows for random/guided; see
# CLASS_WEIGHTED_POOL_CAP above for class-weighted). Set this for a cheap
# 1-seed timing/smoke check before committing to the full N_SEEDS sweep --
# both modes now evaluate on the same shared eval subsample (see below),
# so the only difference is seed count and whether results get saved.
QUICK_SANITY_CHECK = os.environ.get("CSCAS_QUICK_SANITY_CHECK", "1") == "1"
QUICK_SEEDS = 1

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
# applies by excluding them from FEATURE_COLS), SignatureID (nominal
# identifier), SCAS (the paper's own outlier/inlier flag), and every
# *Similarity column (plain numeric scores, no textual content -- see
# module docstring above).
DROP_COLS = ["Timestamp", "Label", "ExtIP", "IntIP", "SignatureID", "SCAS"]
TEXT_FIELD_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]
assert (
    len(TEXT_FIELD_COLS) == 6
), (
    f"got {len(TEXT_FIELD_COLS)}"
)  # SignatureText, SignatureMatchesPerDay, AlertCount, Proto, ExtPort, IntPort
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


# 5) Prepare the shared, frozen eval subsample once, outside the seed loop.
# Same subsample every other non-replication baseline uses -- see module
# docstring for why this is no longer a timing-only ad hoc sample.
eval_df = get_cscas_eval_subsample(test)
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, "
    f"{int(eval_df['Label'].sum())} positive"
)

print(f"Serializing eval subsample to text ({len(eval_df)} rows)...")
test_df = pd.DataFrame(
    {
        "text": build_text_column(eval_df),
        "label": eval_df["Label"].astype(int).values,
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


print("Tokenizing eval subsample...")
test_ds = tokenize(test_ds_base)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f = f1_score(labels, preds, zero_division=0)
    return {"precision": p, "recall": r, "f1": f}


class WeightedLossTrainer(Trainer):
    """Trainer with a weighted cross-entropy loss, weighted by the pool's
    own class balance (see class_weights_tensor below) -- this is what
    "class-weighted" means for a fine-tuned model, the analog of
    sklearn's `class_weight='balanced'` for RF/LogReg.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def run_seed(
    pool: pd.DataFrame,
    seed: int,
    class_weights: torch.Tensor | None,
    condition: str,
) -> dict[str, float]:
    """
    Fine-tune DistilBERT from scratch on `pool`, evaluate once on the
    shared eval subsample, and return precision/recall/f1 -- one point in
    the N_SEEDS-seed average for `condition`.
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
        output_dir=f"/tmp/cscas_bert_{condition}_seed{seed}",
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

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
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
        f"({len(test_ds)} eval rows)"
    )

    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def class_weights_tensor(pool: pd.DataFrame, label_col: str = "Label") -> torch.Tensor:
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=pool[label_col].values
    )
    return torch.tensor(weights, dtype=torch.float)


SEEDS_TO_RUN = range(QUICK_SEEDS) if QUICK_SANITY_CHECK else range(N_SEEDS)

# 7) Three training-pool conditions -- same POOL_BUILDERS pattern as
# cscas_base.py. `important` (all positive-label rows) feeds random/guided,
# same as every other trainable baseline.
important = train[train["Label"] == 1]

POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(
        train, seed=seed, cap=CLASS_WEIGHTED_POOL_CAP
    ),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}

for condition, build_pool in POOL_BUILDERS.items():
    print(f"\n=== {condition} (BERT, reduced fields as text) ===")

    for seed in SEEDS_TO_RUN:
        pool, extra_kwargs = build_pool(seed)
        weights = (
            class_weights_tensor(pool)
            if extra_kwargs.get("class_weight") == "balanced"
            else None
        )

        metrics = run_seed(pool, seed, class_weights=weights, condition=condition)
        results[condition].append(metrics)
        print(
            f"  seed={seed}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )

    avg = pd.DataFrame(results[condition]).mean()
    print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")

if QUICK_SANITY_CHECK:
    print(
        f"\n[QUICK_SANITY_CHECK] Numbers above are from {QUICK_SEEDS} seed(s) -- "
        f"smoke-test only, NOT comparable to the real {N_SEEDS}-seed average. "
        "Set QUICK_SANITY_CHECK=False for real numbers."
    )
    print(
        "[QUICK_SANITY_CHECK] Not saving results -- these numbers are smoke-test only."
    )
else:
    save_baseline_results(
        name="cscas_bert",
        description=(
            "6 reduced fields (no SignatureID/SCAS/Similarity) serialized to text, "
            "fine-tuned DistilBERT, all three training-pool conditions (random "
            "undersampling, class-weighted, guided by CSCAS), evaluated on the "
            "shared eval subsample"
        ),
        results=results,
    )
