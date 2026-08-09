#!/usr/bin/env bash
# Zero-shot baseline sweep: runs cscas_zeroshot.py once per model listed in
# MODELS below, each as its own OLLAMA_MODEL-scoped run. Mirrors
# run_overnight.sh's conventions (timestamped log under logs/, run_step
# wrapper, no `set -e` so one model's failure doesn't abort the sweep).
#
# Before running for real: cscas_zeroshot.py has QUICK_SANITY_CHECK = True
# hardcoded near the top -- flip it to False there once you've smoke-tested
# each model works end-to-end, otherwise every run below is a 50-row smoke
# check, not the full 20,000-row eval subsample.
#
# Requires (see cscas_zeroshot.py's own docstring for details):
#   - Ollama running on the DGX host (not in this container)
#   - This container launched with --network host
#   - Each model in MODELS gets pulled automatically via Ollama's HTTP API
#     (POST /api/pull) before its eval run -- NOT the `ollama` CLI, which
#     only exists on the host, not inside this container. The API call
#     blocks (stream:false) until the pull finishes, so this doubles as a
#     fail-fast check if Ollama itself isn't reachable.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_zeroshot.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/zeroshot_sweep_$(date +%Y%m%d_%H%M%S).log"

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

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
    echo "=== Zero-shot sweep started at $(date) ==="
    echo "Models: ${MODELS[*]}"
    echo "Ollama host: $OLLAMA_HOST"

    for model in "${MODELS[@]}"; do
        run_step "ollama pull $model" ollama_pull "$model"
        run_step "cscas_zeroshot.py ($model)" env OLLAMA_MODEL="$model" python cscas_zeroshot.py
    done

    echo ""
    echo "=== Zero-shot sweep finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"