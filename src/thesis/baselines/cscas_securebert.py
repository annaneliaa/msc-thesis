"""
Same experimental setup as cscas_bert.py -- same temporal split, shared
eval subsample, weighted cross-entropy via WeightedLossTrainer, 6-field
reduced text serialization (no SignatureID/SCAS/Similarity), same three
training-pool conditions (random
undersampling, class-weighted, guided by CSCAS) via the same POOL_BUILDERS
pattern -- see cscas_bert.py's module docstring for the full rationale.
What changes here is the checkpoint: `cisco-ai/SecureBERT2.0-base` (Cisco,
2025) instead of distilbert-base-uncased, the domain-adapted encoder in
the project's LLM-baseline axis (see Docs/Baselines.md) -- fine-tuned from
`answerdotai/ModernBERT-base` on threat reports, vulnerability
descriptions, and MITRE ATT&CK-style text.

Unlike v1 (`ehsanaghaei/SecureBERT`, RoBERTa architecture, needed
Roberta-specific tokenizer/model classes), SecureBERT 2.0 is architecturally
ModernBERT (its own config.json: `"model_type": "modernbert"`,
`"architectures": ["ModernBertForMaskedLM"]`) and loads cleanly through the
`Auto*` classes, same as cscas_bert.py -- the checkpoint ships a
`tokenizer.json`, so `AutoTokenizer` resolves a fast tokenizer directly.
`AutoModelForSequenceClassification` loads the pretrained encoder body and
freshly initializes a new classification head (same "some weights not
initialized from the checkpoint" pattern as DistilBERT -- expected, that's
what fine-tuning is for). ModernBERT support landed in transformers 4.48;
this repo's transformers dependency has been floored there accordingly.
MAX_LENGTH stays at 256 (unchanged) for parity with the other BERT-family
baselines here, well under ModernBERT's 8192-token native context -- the
serialized 6-field rows are short enough that this doesn't truncate.

Everything else -- WeightedLossTrainer, CLASS_WEIGHTED_POOL_CAP's open
"size TBD" decision, shared eval subsample, N_SEEDS -- is identical to
cscas_bert.py; see that module's docstring for the full rationale.

Run:
    cd src/thesis/baselines
    python cscas_securebert.py

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

MODEL_NAME = "cisco-ai/SecureBERT2.0-base"
MAX_LENGTH = 256  # bump if you see truncation warnings
N_SEEDS = 5  # same seed count as cscas_bert.py and the RF/LogReg/XGBoost baselines
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
)  # see cscas_bert.py's module docstring -- open "size TBD" decision, same call here

# Fine-tuning always sees a small pool (~3,530 rows for random/guided; see
# CLASS_WEIGHTED_POOL_CAP above for class-weighted). Set this for a cheap
# 1-seed timing/smoke check before committing to the full N_SEEDS sweep --
# both modes evaluate on the same shared eval subsample (see below), so the
# only difference is seed count and whether results get saved.
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
# to cscas.py/cscas_base.py/cscas_bert.py, so the final test set (and its row
# count) is the same across every CSCAS baseline.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# 4) Fields serialized into text -- identical to cscas_bert.py. Dropped:
# Timestamp (used only for the split), Label (the target), ExtIP/IntIP
# (anonymized identifiers), SignatureID (nominal identifier), SCAS (the
# paper's own outlier/inlier flag), and every *Similarity column (plain
# numeric scores, no textual content).
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
# Same subsample every other non-replication baseline uses.
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
    Fine-tune SecureBERT 2.0 from its pretrained checkpoint on `pool`, evaluate
    once on the shared eval subsample, and return precision/recall/f1 --
    one point in the N_SEEDS-seed average for `condition`.
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
        output_dir=f"/tmp/cscas_securebert_{condition}_seed{seed}",
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

# 6) Three training-pool conditions -- same POOL_BUILDERS pattern as
# cscas_bert.py/cscas_base.py. `important` (all positive-label rows) feeds
# random/guided, same as every other trainable baseline.
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
    print(f"\n=== {condition} (SecureBERT 2.0, reduced fields as text) ===")

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
        name="cscas_securebert",
        description=(
            "6 reduced fields (no SignatureID/SCAS/Similarity) serialized to text, "
            "fine-tuned SecureBERT 2.0 (ModernBERT architecture), all three "
            "training-pool conditions (random undersampling, class-weighted, "
            "guided by CSCAS), evaluated on the shared eval subsample"
        ),
        results=results,
    )
