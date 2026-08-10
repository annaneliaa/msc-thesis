"""
AIT-ADS counterpart to baselines/cscas_zeroshot.py: zero-shot LLM baseline
via Ollama, prompted with one alert_group's serialized text (see
_ait_ads_data.py -- raw_items tokens + n_alerts/hour, the AIT-ADS analog of
CSCAS's SignatureText field), evaluated per (grouping_method, scenario) --
see ait_ads_bert.py's module docstring for why grouping method is a real
axis here (unlike CSCAS, which is pregrouped) and for the leakage skip
(shaw/wardbeck/wheeler/wilson under alertbert/deepcase).

No training pool, no seeds, no conditions -- same single deterministic pass
as cscas_zeroshot.py (temperature=0, fixed seed). Evaluates on each
scenario's full test split directly: unlike CSCAS's 1.255M-row test set,
AIT-ADS scenarios are small enough that no frozen eval subsample is needed
(see _ait_ads_data.py's module docstring).

Requires the same Ollama setup as cscas_zeroshot.py -- see that module's
docstring for the full requirements list (Ollama running and reachable,
target model pulled, --network host if containerized).

Run:
    cd src/thesis/baselines
    OLLAMA_MODEL=llama3.1:8b python ait_ads_zeroshot.py

Set AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS (comma-separated) to run a
subset of either axis.

Before committing to a full scenario run, set QUICK_SANITY_CHECK = True
below for a cheap smoke check on a handful of rows per scenario (not
saved), then flip it back to False for real, reportable numbers.

Resumable: skips a (grouping_method, scenario) combo whose results/*.json
already exists, so a partial run followed by a restart doesn't redo
everything from scratch. Set AIT_ADS_FORCE=1 to force a full re-run. (Only
applies to real runs -- QUICK_SANITY_CHECK passes never save, so they never
trigger this skip either.)
"""

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import f1_score, precision_score, recall_score

from thesis.baselines._ait_ads_data import (
    GROUPING_METHODS as ALL_GROUPING_METHODS,
    load_ait_ads_baseline_split,
)
from thesis.baselines._ait_ads_grouping import LEAKAGE_SCENARIOS, LEARNED_METHODS
from thesis.baselines._results import results_exist, save_zeroshot_results
from thesis.configs import load_scenarios

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
MODEL_SLUG = MODEL_NAME.replace(":", "-").replace("/", "-")
MAX_NEW_TOKENS = 8
GENERATION_BATCH_SIZE = 8  # concurrent in-flight requests to Ollama, not a padded batch

QUICK_SANITY_CHECK = False
QUICK_EVAL_N = 50

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

SYSTEM_PROMPT = (
    "You are a network security analyst triaging intrusion-detection alerts. "
    "You will be given the serialized tokens of one alert GROUP -- a cluster "
    "of alerts grouped together by a fixed time window, not a single "
    "isolated event. 'Tokens' are the group's signature/host/category tags "
    "(sig:*, host:*, short:*); N_alerts is how many individual alerts are "
    "in this specific group. Use these signals as part of your judgment "
    "(e.g. a group with many alerts and suspicious signature tokens looks "
    "different from a small, benign-looking one). Decide whether the group "
    "represents a real attack (relevant, should be escalated) or an "
    "irrelevant/benign pattern (should be dismissed). "
    "Respond with exactly one word: ATTACK or BENIGN. Do not explain your reasoning."
)


def check_ollama_ready() -> None:
    """Fail fast with a clear message rather than erroring partway through
    a scenario if Ollama isn't reachable or the model hasn't been pulled."""
    import requests

    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running on the "
            "host, and was this container launched with --network host? "
            f"Original error: {e}"
        ) from e
    available = [m["name"] for m in resp.json().get("models", [])]
    if not any(MODEL_NAME in m for m in available):
        raise RuntimeError(
            f"'{MODEL_NAME}' not found on the Ollama server at {OLLAMA_HOST}. "
            f"Run `ollama pull {MODEL_NAME}` first. "
            f"Available models: {available or '(none pulled yet)'}"
        )


def build_prompt(alert_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_text},
    ]


def parse_label(generated_text: str) -> int | None:
    text = generated_text.strip().upper()
    has_attack = "ATTACK" in text
    has_benign = "BENIGN" in text
    if has_attack and not has_benign:
        return 1
    if has_benign and not has_attack:
        return 0
    return None


def generate_one(messages: list[dict]) -> str:
    import requests

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
    with ThreadPoolExecutor(max_workers=len(prompt_batches)) as ex:
        return list(ex.map(generate_one, prompt_batches))


def run_scenario(scenario: str, grouping_method: str) -> None:
    run_tag = f"{grouping_method}_{scenario}"
    result_name = f"ait_ads_zeroshot_{run_tag}_{MODEL_SLUG}"
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

    _train, test = load_ait_ads_baseline_split(
        scenario, grouping_method=grouping_method
    )
    if test["Label"].nunique() < 2:
        print(f"  [skip] {run_tag}: single-class test split.")
        return

    eval_df = test
    if QUICK_SANITY_CHECK:
        eval_df = eval_df.iloc[:QUICK_EVAL_N]
        print(
            f"[QUICK_SANITY_CHECK] {run_tag}: using only the first "
            f"{len(eval_df)} rows of the test split."
        )
    print(
        f"  Evaluating on {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
    )

    prompts = [build_prompt(t) for t in eval_df["text"]]
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
                label = 0  # fallback: unparseable output treated as the majority class (benign)
            preds.append(label)

        done = start + len(batch_prompts)
        if done % (GENERATION_BATCH_SIZE * 10) == 0 or done >= len(prompts):
            elapsed = time.time() - t_start
            print(
                f"  {min(done, len(prompts))}/{len(prompts)} rows ({elapsed:.0f}s elapsed)"
            )

    if n_unparsed:
        print(
            f"  WARNING: {n_unparsed}/{len(preds)} generations didn't contain "
            "ATTACK or BENIGN (or contained both) -- defaulted to BENIGN (0)."
        )

    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f = f1_score(labels, preds, zero_division=0)
    print(f"\n=== {run_tag} / Zero-shot ({MODEL_NAME}) ===")
    print(f"P={p:.3f} R={r:.3f} F1={f:.3f}  ({len(preds)} rows, {n_unparsed} unparsed)")

    if QUICK_SANITY_CHECK:
        print(f"[QUICK_SANITY_CHECK] {run_tag}: not saving -- smoke-test only.")
    else:
        save_zeroshot_results(
            name=result_name,
            description=(
                f"AIT-ADS scenario '{scenario}' grouped with '{grouping_method}': "
                f"zero-shot {MODEL_NAME} (via Ollama), alert_group tokens "
                "(sig:/host:/short:) serialized to a prompt, evaluated on the "
                "scenario's full test split"
            ),
            metrics={"precision": p, "recall": r, "f1": f},
        )


print(f"Using Ollama model '{MODEL_NAME}' at {OLLAMA_HOST}")
check_ollama_ready()

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
