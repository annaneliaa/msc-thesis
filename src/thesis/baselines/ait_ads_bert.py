"""
AIT-ADS counterpart to baselines/cscas_bert.py: fine-tuned DistilBERT on one
text column per alert_group (see _ait_ads_data.py -- raw_items tokens +
n_alerts/hour, the AIT-ADS analog of CSCAS's SignatureText field), evaluated
per (grouping_method, scenario) -- unlike CSCAS (pregrouped), AIT-ADS needs
a grouping step first, and which of the 5 grouping methods
(_ait_ads_data.GROUPING_METHODS: fixed_window, time_delta, cscas_grouping,
alertbert, deepcase) is used is itself a real axis, not a fixed choice. See
_ait_ads_grouping.py's module docstring for the alertbert/deepcase
operating points and why shaw/wardbeck/wheeler/wilson are skipped for those
two methods (both were trained on exactly those 4 scenarios -- using them
for a baseline result there would be leakage).

Loops every AIT-ADS scenario (thesis.configs.load_scenarios("ait-ads"): fox,
harrison, russellmitchell, santos, shaw, wardbeck, wheeler, wilson) x every
grouping method by default.

Two training-pool conditions only -- random undersampling and class-weighted
(natural-ratio) -- not three: "guided" is CSCAS-only (no SCAS-equivalent
outlier signal for AIT-ADS, see _sampling.py's own docstring), so it's
dropped here rather than faked with a substitute signal.

Same WeightedLossTrainer/N_SEEDS=5/QUICK_SANITY_CHECK pattern as
cscas_bert.py -- see that module's docstring for the full rationale, only
repeated here where AIT-ADS actually differs:
  - Data: _ait_ads_data.load_ait_ads_baseline_split(scenario) instead of
    pd.read_csv(...) -- no fixed CSV, no paper row-count assertions (no
    paper to replicate here).
  - No shared eval subsample: AIT-ADS scenarios are far smaller than
    CSCAS's 1.25M-row test set, so this scores each scenario's own full
    test split directly.
  - The `text` column is already built by the loader (one column, not 6
    discrete fields to serialize), so there's no build_text_column here.

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists, so a partial run followed by a restart doesn't redo
everything from scratch. Set AIT_ADS_FORCE=1 to force a full re-run. (Only
applies to real runs -- QUICK_SANITY_CHECK passes never save, so they never
trigger this skip either.)

class_weighted's pool is capped at AIT_ADS_CLASS_WEIGHTED_POOL_CAP rows
(default 15000, matching cscas_bert.py's own CLASS_WEIGHTED_POOL_CAP
convention) -- uncapped, this condition fine-tunes on the *entire* natural-
ratio train split every seed, which measured ~100x longer fit time per seed
than the random condition's small capped pool in an uncapped run (~19min vs
~11s per seed on CPU). Set it to "none"/"uncapped"/"0" for the old uncapped
behavior; any other integer overrides the cap.

Device: auto-detects mps -> cuda -> cpu and prints a
[device diagnostics] line (torch/cuda versions, availability, device count)
so a silent CPU fallback is diagnosable at a glance instead of a one-word
log line. Set AIT_ADS_REQUIRE_GPU=1 to fail immediately at startup if
neither mps nor cuda is available, rather than silently fine-tuning on CPU
for hours/days -- this can't *make* torch see a GPU it doesn't (that's a
container/driver/torch-install issue, not something this script controls),
it just turns a multi-day silent CPU run into an immediate, loud one.

Run:
    cd src/thesis/baselines
    python ait_ads_bert.py

Before committing to the full N_SEEDS-seed run, set QUICK_SANITY_CHECK =
True below (or AIT_ADS_QUICK_SANITY_CHECK=1) for a cheap 1-seed timing/smoke
check per scenario, then flip it back to False for real, reportable numbers.
"""

import os
import time
import traceback

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

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import class_weighted_pool, random_undersample_pool
from thesis.configs import load_scenarios

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # bump if you see truncation warnings
N_SEEDS = 5  # same seed count as cscas_bert.py and the RF/LogReg/XGBoost baselines
BATCH_SIZE = 8
NUM_EPOCHS = 3
VAL_FRAC_WITHIN_POOL = 0.15  # stratified carve-out from the training pool, for epoch-level eval/model selection only -- not the final eval set

_cap_env = os.environ.get("AIT_ADS_CLASS_WEIGHTED_POOL_CAP")
if _cap_env is None:
    CLASS_WEIGHTED_POOL_CAP = 15000  # default cap -- see module docstring
elif _cap_env.strip().lower() in ("", "none", "uncapped", "0"):
    CLASS_WEIGHTED_POOL_CAP = None
else:
    CLASS_WEIGHTED_POOL_CAP = int(_cap_env)

# Set True (or AIT_ADS_QUICK_SANITY_CHECK=1) for a cheap 1-seed timing/smoke
# check per scenario before committing to the full N_SEEDS sweep.
QUICK_SANITY_CHECK = os.environ.get("AIT_ADS_QUICK_SANITY_CHECK", "1") == "1"
QUICK_SEEDS = 1

FORCE = os.environ.get("AIT_ADS_FORCE", "0") == "1"

_scenarios_env = os.environ.get("AIT_ADS_SCENARIOS")
SCENARIOS = (
    [s.strip() for s in _scenarios_env.split(",") if s.strip()]
    if _scenarios_env
    else load_scenarios("ait-ads")
)

_grouping_methods_env = os.environ.get("AIT_ADS_GROUPING_METHODS")
GROUPING_METHODS = (
    [g.strip() for g in _grouping_methods_env.split(",") if g.strip()]
    if _grouping_methods_env
    else ALL_GROUPING_METHODS
)

REQUIRE_GPU = os.environ.get("AIT_ADS_REQUIRE_GPU", "0") == "1"

_mps_ok = torch.backends.mps.is_available()
_cuda_ok = torch.cuda.is_available()
device = "mps" if _mps_ok else ("cuda" if _cuda_ok else "cpu")
print(f"Using device: {device}")
print(
    f"  [device diagnostics] torch={torch.__version__} cuda_build={torch.version.cuda} "
    f"mps_available={_mps_ok} cuda_available={_cuda_ok} "
    f"cuda_device_count={torch.cuda.device_count() if _cuda_ok else 0}"
)
if REQUIRE_GPU and device == "cpu":
    raise RuntimeError(
        "AIT_ADS_REQUIRE_GPU=1 but neither MPS nor CUDA is available (see the "
        "[device diagnostics] line above) -- refusing to silently fine-tune on "
        "CPU. Fix the container's GPU/CUDA visibility first (check `nvidia-smi` "
        'and `python3 -c "import torch; print(torch.cuda.is_available())"` '
        "inside the container), or unset AIT_ADS_REQUIRE_GPU to proceed on CPU "
        "anyway."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(ds: Dataset) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


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
    own class balance -- the fine-tuned analog of sklearn's
    class_weight='balanced'. Identical to cscas_bert.py's own class."""

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
    test_ds: Dataset,
    seed: int,
    class_weights: torch.Tensor | None,
    condition: str,
    run_tag: str,
) -> dict[str, float]:
    """Fine-tune DistilBERT from scratch on `pool`, evaluate once on
    `scenario`'s test split, return precision/recall/f1 -- one point in the
    N_SEEDS-seed average for `condition`."""
    t_start = time.time()
    pool_df = pool[["text", "Label"]].rename(columns={"Label": "label"})
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
        output_dir=f"/tmp/ait_ads_bert_{run_tag}_{condition}_seed{seed}",
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


def run_scenario(scenario: str, grouping_method: str) -> None:
    run_tag = f"{grouping_method}_{scenario}"
    result_name = f"ait_ads_bert_{run_tag}"
    print(
        f"\n{'=' * 70}\n  SCENARIO: {scenario} / GROUPING: {grouping_method}\n{'=' * 70}"
    )
    if not QUICK_SANITY_CHECK and not FORCE and results_exist(result_name):
        print(
            f"  [skip] {run_tag}: {result_name}.json already exists (set AIT_ADS_FORCE=1 to re-run)."
        )
        return
    if grouping_method in LEARNED_METHODS and scenario in LEAKAGE_SCENARIOS:
        print(
            f"  [skip] {scenario}/{grouping_method}: would be self-training "
            f"leakage -- {grouping_method}'s model was trained on this scenario. "
            "See _ait_ads_grouping.py's module docstring."
        )
        return

    train, test = load_ait_ads_baseline_split(scenario, grouping_method=grouping_method)
    print(
        f"  {len(train)} train / {len(test)} test alert_groups, "
        f"{int(train['Label'].sum())} train positive, "
        f"{int(test['Label'].sum())} test positive"
    )
    if train["Label"].nunique() < 2 or test["Label"].nunique() < 2:
        print(f"  [skip] {scenario}: single-class train or test split.")
        return

    important = train[train["Label"] == 1]
    if len(important) == 0:
        print(f"  [skip] {scenario}: no positive (attack) rows in train split.")
        return

    test_ds_base = Dataset.from_pandas(
        test[["text", "Label"]].rename(columns={"Label": "label"}),
        preserve_index=False,
    )
    test_ds = tokenize(test_ds_base)

    # Two conditions only -- no "guided" -- see module docstring.
    pool_builders = {
        "random": lambda seed: random_undersample_pool(train, important, seed),
        "class_weighted": lambda seed: class_weighted_pool(
            train, seed=seed, cap=CLASS_WEIGHTED_POOL_CAP
        ),
    }

    results: dict[str, list[dict[str, float]]] = {name: [] for name in pool_builders}

    for condition, build_pool in pool_builders.items():
        print(f"\n=== {run_tag} / {condition} (BERT, tokens as text) ===")
        for seed in SEEDS_TO_RUN:
            pool, extra_kwargs = build_pool(seed)
            weights = (
                class_weights_tensor(pool)
                if extra_kwargs.get("class_weight") == "balanced"
                else None
            )
            metrics = run_seed(pool, test_ds, seed, weights, condition, run_tag)
            results[condition].append(metrics)
            print(
                f"  seed={seed}: P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
            )
        avg = pd.DataFrame(results[condition]).mean()
        print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")

    if QUICK_SANITY_CHECK:
        print(
            f"\n[QUICK_SANITY_CHECK] {run_tag}: numbers above are from "
            f"{QUICK_SEEDS} seed(s) -- smoke-test only, NOT comparable to the "
            f"real {N_SEEDS}-seed average. Set AIT_ADS_QUICK_SANITY_CHECK=0 "
            "for real numbers. Not saving."
        )
    else:
        save_baseline_results(
            name=result_name,
            description=(
                f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
                "alert_group tokens (sig:/host:/short:) serialized to text, "
                "fine-tuned DistilBERT, random + class-weighted training-pool "
                "conditions (no 'guided' -- CSCAS-only, no SCAS-equivalent "
                "signal for AIT-ADS), evaluated on the scenario's full test split"
            ),
            results=results,
        )


for _grouping_method in GROUPING_METHODS:
    for _scenario in SCENARIOS:
        try:
            run_scenario(_scenario, _grouping_method)
        except Exception:
            print(
                f"\n[ERROR] {_grouping_method}_{_scenario}: unhandled exception -- "
                "continuing to the next combo."
            )
            traceback.print_exc()
