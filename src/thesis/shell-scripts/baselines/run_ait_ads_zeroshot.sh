#!/usr/bin/env bash
# Zero-shot AIT-ADS baseline sweep: runs ait_ads_zeroshot.py once per model
# listed in MODELS below, each as its own OLLAMA_MODEL-scoped run --
# ait_ads_zeroshot.py already loops every AIT-ADS scenario x grouping
# method internally (AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS below to
# run a subset of either), so this script only needs to loop MODELS, not
# scenarios/grouping methods. Mirrors run_zeroshot.sh's conventions
# (timestamped log under logs/, run_step wrapper, no `set -e` so one
# model's failure doesn't abort the sweep).
#
# Like run_ait_ads_bert_securebert.sh, alertbert grouping needs the
# `thesis-alertbert` conda env (graph-tool -- see _ait_ads_grouping.py's
# module docstring); ait_ads_zeroshot.py's own deps (requests, sklearn) are
# light enough to usually already be present there too, but this script
# still runs alertbert separately via `conda run` for correctness rather
# than assuming that.
#
# Before running for real: ait_ads_zeroshot.py has QUICK_SANITY_CHECK = False
# hardcoded near the top already (unlike cscas_zeroshot.py, which defaults
# True) -- flip it to True there first if you want a cheap smoke check on a
# handful of rows per scenario before committing to a full run.
#
# Requires (see ait_ads_zeroshot.py's own docstring for details, same as
# cscas_zeroshot.py's):
#   - Ollama running on the DGX host (not in this container)
#   - This container launched with --network host
#   - Each model in MODELS gets pulled automatically via Ollama's HTTP API
#     (POST /api/pull) before its eval run.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_zeroshot.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_zeroshot_sweep_$(date +%Y%m%d_%H%M%S).log"

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export AIT_ADS_SCENARIOS="${AIT_ADS_SCENARIOS:-}"  # empty = every AIT-ADS scenario
NON_ALERTBERT_METHODS="${AIT_ADS_GROUPING_METHODS:-fixed_window,time_delta,cscas_grouping,deepcase}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//alertbert/}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//,,/,}"

MODELS=(
    "llama3.1:8b"
    "llama3.1:70b"
    "qwen2.5:7b"
    "qwen2.5:72b"
)

run_step() {
    local label="$1"
    shift
    echo ""
    echo "--- $label started at $(date) ---"
    local start end status
    start=$(date +%s)
    "$@"
    status=$?
    end=$(date +%s)
    echo "--- $label finished at $(date) (exit=$status, $((end - start))s) ---"
    return 0  # never abort the sweep on a single model's failure
}

ollama_pull() {
    local model="$1"
    curl -sf -X POST "$OLLAMA_HOST/api/pull" \
        -d "{\"name\": \"$model\", \"stream\": false}"
}

{
    echo "=== AIT-ADS zero-shot sweep started at $(date) ==="
    echo "Models: ${MODELS[*]}"
    echo "Ollama host: $OLLAMA_HOST"
    echo "Scenarios: ${AIT_ADS_SCENARIOS:-<all>}"
    echo "Non-alertbert grouping methods (plain venv): $NON_ALERTBERT_METHODS"

    for model in "${MODELS[@]}"; do
        run_step "ollama pull $model" ollama_pull "$model"
        AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
            run_step "ait_ads_zeroshot.py ($model, fixed_window/time_delta/cscas_grouping/deepcase)" \
            env OLLAMA_MODEL="$model" python ait_ads_zeroshot.py
        run_step "ait_ads_zeroshot.py ($model, alertbert)" \
            env OLLAMA_MODEL="$model" AIT_ADS_GROUPING_METHODS=alertbert \
            conda run -n thesis-alertbert --no-capture-output python ait_ads_zeroshot.py
    done

    echo ""
    echo "=== AIT-ADS zero-shot sweep finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"
