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
#   - Each model in MODELS already pulled on the host, e.g.
#     `ollama pull llama3.1:8b` -- this script attempts `ollama pull` for
#     each model first (idempotent/cheap if already present) so a missing
#     model fails fast and clearly instead of mid-sweep inside Python.
#
# Run:
#   cd src/thesis/baselines
#   nohup ./run_zeroshot_sweep.sh > /dev/null 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/zeroshot_sweep_$(date +%Y%m%d_%H%M%S).log"

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

{
    echo "=== Zero-shot sweep started at $(date) ==="
    echo "Models: ${MODELS[*]}"

    for model in "${MODELS[@]}"; do
        run_step "ollama pull $model" ollama pull "$model"
        run_step "cscas_zeroshot.py ($model)" env OLLAMA_MODEL="$model" python cscas_zeroshot.py
    done

    echo ""
    echo "=== Zero-shot sweep finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"