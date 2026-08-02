#!/usr/bin/env bash
#
# Wraps scripts/full/run_model_comparison_attribute.py so a long GPU run
# survives closing the terminal it was launched from. The python script
# already tees its own stdout to a log under
# artifacts/experiments/run_model_comparison_attribute/<run>/plots/ (see
# that script's main()) -- this wrapper doesn't duplicate that, it just
# makes sure the process is properly detached so closing your terminal
# doesn't kill it, and adds a second start/end summary log that exists even
# if the python script dies before creating its own run dir.
#
# Edit SCENARIO/MODELS/EXTRA_ARGS below, or override at the command line:
#   ./run_model_comparison_attribute.sh cscas --models xgboost torch_nn
#
# --- Getting this to survive BOTH `docker exec` and SSH closing ---
#
# `docker run -d ... sleep infinity` (per Dockerfile.dgx) keeps the
# container itself alive independent of SSH -- that part's already handled
# if you followed Dockerfile.dgx's own instructions. What still needs
# detaching is the `docker exec -it` shell you're about to run this from:
# by default, closing that terminal sends SIGHUP to whatever's running
# inside it, container or not.
#
# Option A -- nohup + disown inside an interactive `docker exec -it` shell
# (lets you watch it start before disconnecting):
#   docker exec -it thesis-run bash
#   cd /workspace/src/thesis/shell-scripts
#   nohup ./run_model_comparison_attribute.sh cscas > /dev/null 2>&1 &
#   disown
#   exit                    # closes this exec session -- the run keeps going
#
# Option B -- skip the interactive session entirely, launch already-detached
# from the host in one shot:
#   docker exec -d thesis-run bash -lc \
#     'cd /workspace/src/thesis/shell-scripts && ./run_model_comparison_attribute.sh cscas'
#
# Either way, you can now safely close SSH. Reattach later to check on it:
#   docker exec -it thesis-run bash -lc \
#     'tail -f "$(ls -t /workspace/artifacts/logs/run_model_comparison_attribute/*.log | head -1)"'

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SCENARIO="${1:-cscas}"
if [[ $# -gt 0 ]]; then shift; fi
MODELS=(logreg rf mlp xgboost torch_nn rf_gpu)
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 || "${EXTRA_ARGS[*]}" != *"--force"* ]]; then
  EXTRA_ARGS+=(--force)
fi

LOG_DIR="$REPO_ROOT/artifacts/logs/run_model_comparison_attribute"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S)_${SCENARIO}.log"

{
  echo "=== run_model_comparison_attribute started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "PID: $$"
  echo "scenario: $SCENARIO"
  echo "models: ${MODELS[*]}"
  echo "extra args: ${EXTRA_ARGS[*]}"
  echo

  python3 src/thesis/scripts/full/run_model_comparison_attribute.py \
    "$SCENARIO" \
    --models "${MODELS[@]}" \
    "${EXTRA_ARGS[@]}"
  status=$?

  echo
  echo "=== run_model_comparison_attribute finished $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$status) ==="
} 2>&1 | tee "$LOG_FILE"
