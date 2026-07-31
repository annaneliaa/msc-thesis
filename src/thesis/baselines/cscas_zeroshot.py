"""
Zero-shot LLM baseline: no training step, so none of the three training-
pool conditions (random/class-weighted/guided) apply here. The model is
prompted directly with the same 6-field reduced text serialization
cscas_bert.py/cscas_securebert.py use (no SignatureID/SCAS/Similarity --
see cscas_base.py's module docstring for why; reused as the body of a
prompt instead of fed straight to a fine-tuned classifier), and the
generated label is parsed into precision/recall/f1. Single deterministic
run -- greedy-equivalent decoding (temperature=0, fixed seed), no seed
loop -- this is the one baseline script that stays single-condition/
single-run (see Docs/Baselines.md: architecturally distinct from BERT/
SecureBERT -- decoder/generative, not encoder/fine-tuned).

Evaluates ONLY on the shared, frozen eval subsample (see
_sampling.get_cscas_eval_subsample) -- mandatory here, not optional like
the RF/LogReg/XGBoost/BERT/SecureBERT scripts, since prompting an LLM over
the full 1.255M-row test set is not tractable.

Backend: inference is delegated to a locally running Ollama server rather
than loading weights directly via `transformers` -- Ollama handles model
loading, quantization, and GPU placement itself, and its model library
sidesteps Hugging Face's gated-access flow for Llama.

Requires:
  1. Ollama installed and running on the DGX host (NOT inside this
     container): `curl -fsSL https://ollama.com/install.sh | sh`, then
     `ollama serve` (or the systemd service the installer sets up --
     check `systemctl status ollama` first).
  2. The target model pulled once on the host, e.g. `ollama pull
     llama3.1:8b`. Swap OLLAMA_MODEL below (or the OLLAMA_MODEL env var)
     to use qwen2.5:7b, llama3.1:70b, qwen2.5:72b, etc. -- no other code
     changes needed, Ollama handles quantized weights for all of them.
  3. This container launched with `--network host` (or otherwise able to
     reach the host's Ollama port on 11434), so OLLAMA_HOST below
     resolves. See dgx-spark-workflow.md.

Run:
    cd src/thesis/baselines
    OLLAMA_MODEL=llama3.1:8b python cscas_zeroshot.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.

Before committing to the full 20,000-row eval subsample, set
QUICK_SANITY_CHECK = True below for a cheap smoke check on a handful of
rows (still drawn from the real shared eval subsample, just fewer of its
rows, and not saved), then flip it back to False for real, reportable
numbers. Prompting even an 8B-class model 20,000 times over a network call
per request adds up -- GENERATION_BATCH_SIZE now controls how many
in-flight concurrent requests are sent to Ollama at once, not a padded
tensor batch (Ollama has no native equivalent of HF's batched
model.generate) -- raise it only as far as OLLAMA_NUM_PARALLEL on the
server side actually supports concurrently.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from sklearn.metrics import f1_score, precision_score, recall_score

from thesis.baselines._results import save_zeroshot_results
from thesis.baselines._sampling import get_cscas_eval_subsample

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
MAX_NEW_TOKENS = 8
GENERATION_BATCH_SIZE = 8  # concurrent in-flight requests to Ollama, not a padded batch

QUICK_SANITY_CHECK = False
QUICK_EVAL_N = 50

SYSTEM_PROMPT = (
    "You are a network security analyst triaging intrusion-detection alerts. "
    "You will be given the serialized fields of one alert GROUP -- a cluster "
    "of alerts sharing the same signature, not a single isolated event. "
    "AlertCount is how many alerts are in this specific group; "
    "SignatureMatchesPerDay is how often this signature fires per day "
    "overall. Use these frequency signals as part of your judgment (e.g. a "
    "signature firing constantly across many alerts looks different from a "
    "rare, isolated one). Decide whether the group represents a real attack "
    "(relevant, should be escalated) or an irrelevant/benign pattern "
    "(should be dismissed). "
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


def check_ollama_ready() -> None:
    """Fail fast with a clear message rather than erroring 20,000 requests
    in if Ollama isn't reachable or the model hasn't been pulled."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running on the "
            "DGX host, and was this container launched with --network host? "
            f"Original error: {e}"
        ) from e
    available = [m["name"] for m in resp.json().get("models", [])]
    if not any(MODEL_NAME in m for m in available):
        raise RuntimeError(
            f"'{MODEL_NAME}' not found on the Ollama server at {OLLAMA_HOST}. "
            f"Run `ollama pull {MODEL_NAME}` on the DGX host first. "
            f"Available models: {available or '(none pulled yet)'}"
        )


print(f"Using Ollama model '{MODEL_NAME}' at {OLLAMA_HOST}")
check_ollama_ready()


def build_prompt(alert_text: str) -> list[dict]:
    """Message list for Ollama's /api/chat -- Ollama applies the target
    model's own chat template server-side, so no tokenizer/
    apply_chat_template call is needed here (unlike the HF version)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_text},
    ]


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


def generate_one(messages: list[dict]) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": MAX_NEW_TOKENS,
                "temperature": 0,  # deterministic, greedy-equivalent
                "seed": 0,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def generate_batch(prompt_batches: list[list[dict]]) -> list[str]:
    """Ollama serves one request per call -- there's no native equivalent
    of HF's padded-tensor model.generate() batching. Fan requests out
    across a thread pool instead for some concurrency; real throughput
    still depends on OLLAMA_NUM_PARALLEL on the server side."""
    with ThreadPoolExecutor(max_workers=len(prompt_batches)) as ex:
        return list(ex.map(generate_one, prompt_batches))


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
    # name includes the model so a multi-model sweep (see
    # run_zeroshot_sweep.sh) writes one results/cscas_zeroshot_<model>.json
    # per model instead of every run overwriting the same file.
    model_slug = MODEL_NAME.replace(":", "-").replace("/", "-")
    save_zeroshot_results(
        name=f"cscas_zeroshot_{model_slug}",
        description=(
            f"Zero-shot {MODEL_NAME} (via Ollama), 6 reduced fields (no "
            "SignatureID/SCAS/Similarity) serialized to a prompt, evaluated "
            "on the shared eval subsample"
        ),
        metrics={"precision": p, "recall": r, "f1": f},
    )
