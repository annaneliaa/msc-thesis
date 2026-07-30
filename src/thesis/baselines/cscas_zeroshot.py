"""
Zero-shot LLM baseline: no training step, so none of the three training-
pool conditions (random/class-weighted/guided) apply here. Llama-3.1-8B-
Instruct is prompted directly with the same 6-field reduced text
serialization cscas_bert.py/cscas_securebert.py use (no SignatureID/SCAS/
Similarity -- see cscas_base.py's module docstring for why; reused as the body of a prompt
instead of fed straight to a fine-tuned classifier), and the generated
label is parsed into precision/recall/f1. Single deterministic run --
greedy decoding, no seed loop -- this is the one baseline script that
stays single-condition/single-run (see Docs/Baselines.md: architecturally
distinct from BERT/SecureBERT -- decoder/generative, not encoder/fine-
tuned).

Evaluates ONLY on the shared, frozen eval subsample (see
_sampling.get_cscas_eval_subsample) -- mandatory here, not optional like
the RF/LogReg/XGBoost/BERT/SecureBERT scripts, since prompting an LLM over
the full 1.255M-row test set is not tractable.

Requires:
  1. Accepting the Llama 3.1 license on the model's Hugging Face page
     (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) -- the repo
     is gated ("manual" approval, confirmed via the HF Hub API).
  2. Being authenticated locally so `from_pretrained` can fetch the
     weights: `huggingface-cli login`, or an `HF_TOKEN` environment
     variable.

Run:
    cd src/thesis/baselines
    python cscas_zeroshot.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.

Before committing to the full 20,000-row eval subsample, set
QUICK_SANITY_CHECK = True below for a cheap smoke check on a handful of
rows (still drawn from the real shared eval subsample, just fewer of its
rows, and not saved), then flip it back to False for real, reportable
numbers. An 8B-parameter model prompted 20,000 times is a substantial
local run even with the eval subsample already bounding the cost -- size
GENERATION_BATCH_SIZE to what your GPU/unified memory can hold.
"""

import time

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from thesis.baselines._results import save_zeroshot_results
from thesis.baselines._sampling import get_cscas_eval_subsample

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 8
GENERATION_BATCH_SIZE = 8

QUICK_SANITY_CHECK = True
QUICK_EVAL_N = 50

SYSTEM_PROMPT = (
    "You are a network security analyst triaging intrusion-detection alerts. "
    "You will be given the serialized fields of one alert. Decide whether it "
    "represents a real attack (relevant, should be escalated) or an "
    "irrelevant/benign alert (should be dismissed). "
    "Respond with exactly one word: ATTACK or BENIGN. Do not explain your reasoning."
)

# 1) Load and sort dataset

df = pd.read_csv("../../../data/cscas/dataset-labeled-anon-ip.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2) Verify dataset against paper's numbers
assert len(df) == 1_395_324, f"got {len(df)}"
assert df["Label"].sum() == 20_952, f"got {df['Label'].sum()}"
assert df["SCAS"].sum() == 72_672, f"got {df['SCAS'].sum()}"

# 3) Split into train and test sets based on timestamp -- identical boundary
# to every other CSCAS baseline. Zero-shot has no training step, so this is
# just a sanity check on the split itself (and to derive `test`); the train
# side is never materialized since nothing here trains on it.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

n_train_rows = int((df["Timestamp"] <= split_time).sum())
test = df[df["Timestamp"] > split_time].copy()

assert n_train_rows == 139_532, f"got {n_train_rows}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# 4) Fields serialized into text -- identical to cscas_bert.py/cscas_securebert.py.
DROP_COLS = ["Timestamp", "Label", "ExtIP", "IntIP", "SignatureID", "SCAS"]
TEXT_FIELD_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]
assert (
    len(TEXT_FIELD_COLS) == 6
), (
    f"got {len(TEXT_FIELD_COLS)}"
)  # SignatureText, SignatureMatchesPerDay, AlertCount, Proto, ExtPort, IntPort


def build_text_column(frame: pd.DataFrame) -> pd.Series:
    """Same serialization as cscas_bert.py/cscas_securebert.py -- reused
    here as the body of a prompt instead of raw classifier input."""
    parts = ["SignatureText: " + frame["SignatureText"].astype(str)]
    for col in TEXT_FIELD_COLS:
        if col == "SignatureText":
            continue
        parts.append(f"{col}: " + frame[col].astype(str))
    text = parts[0]
    for p in parts[1:]:
        text = text.str.cat(p, sep=" | ")
    return text


# 5) Prepare the shared, frozen eval subsample -- mandatory here (see
# module docstring), optionally truncated for a cheap smoke check.
eval_df = get_cscas_eval_subsample(test)
if QUICK_SANITY_CHECK:
    eval_df = eval_df.iloc[:QUICK_EVAL_N]
    print(
        f"[QUICK_SANITY_CHECK] Using only the first {len(eval_df)} rows "
        "of the shared eval subsample."
    )
print(f"Evaluating on {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive")

eval_df = eval_df.copy()
eval_df["_prompt_text"] = build_text_column(eval_df)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")

print(f"Loading {MODEL_NAME} (gated -- requires license acceptance + HF auth)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"  # required for correct batched causal-LM generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()


def build_prompt(alert_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_label(generated_text: str) -> int | None:
    """Parse the model's one-word reply into a 0/1 label. Returns None if
    neither expected word is present (or both are) -- caller decides the
    fallback."""
    text = generated_text.strip().upper()
    has_attack = "ATTACK" in text
    has_benign = "BENIGN" in text
    if has_attack and not has_benign:
        return 1
    if has_benign and not has_attack:
        return 0
    return None


@torch.no_grad()
def generate_batch(prompts: list[str]) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# 6) Single deterministic pass over the eval set -- no seeds, no conditions.
prompts = [build_prompt(t) for t in eval_df["_prompt_text"]]
labels = eval_df["Label"].astype(int).tolist()

preds: list[int] = []
n_unparsed = 0
t_start = time.time()

for start in range(0, len(prompts), GENERATION_BATCH_SIZE):
    batch_prompts = prompts[start : start + GENERATION_BATCH_SIZE]
    generations = generate_batch(batch_prompts)
    for gen in generations:
        label = parse_label(gen)
        if label is None:
            n_unparsed += 1
            label = (
                0  # fallback: unparseable output treated as the majority class (benign)
            )
        preds.append(label)

    done = start + len(batch_prompts)
    if done % (GENERATION_BATCH_SIZE * 10) == 0 or done >= len(prompts):
        elapsed = time.time() - t_start
        print(
            f"  {min(done, len(prompts))}/{len(prompts)} rows ({elapsed:.0f}s elapsed)"
        )

if n_unparsed:
    print(
        f"WARNING: {n_unparsed}/{len(preds)} generations didn't contain "
        "ATTACK or BENIGN (or contained both) -- defaulted to BENIGN (0)."
    )

p = precision_score(labels, preds, zero_division=0)
r = recall_score(labels, preds, zero_division=0)
f = f1_score(labels, preds, zero_division=0)
print(f"\n=== Zero-shot ({MODEL_NAME}) ===")
print(f"P={p:.3f} R={r:.3f} F1={f:.3f}  ({len(preds)} rows, {n_unparsed} unparsed)")

if QUICK_SANITY_CHECK:
    print(
        "[QUICK_SANITY_CHECK] Not saving results -- these numbers are smoke-test only."
    )
else:
    save_zeroshot_results(
        name="cscas_zeroshot",
        description=(
            f"Zero-shot {MODEL_NAME}, 6 reduced fields (no SignatureID/SCAS/"
            "Similarity) serialized to a prompt, evaluated on the shared eval subsample"
        ),
        metrics={"precision": p, "recall": r, "f1": f},
    )
