#!/usr/bin/env bash
# Batch runner for the grouping comparison: every method's script
# (fixed_window.py, time_delta.py, cscas_grouping.py,
# cscas_grouping_sensitivity.py, alertbert_sweep.py, deepcase_sweep.py), followed by
# re-executing the comparison notebook so every plot is baked in and ready
# to look at when the run finishes. Mirrors src/thesis/shell-scripts/baselines/run_overnight.sh's
# skeleton (run_step wrapper, no set -e, one tee'd log file).
#
# alertbert_sweep.py needs the `thesis-alertbert` conda env (graph-tool +
# KMP_DUPLICATE_LIB_OK=TRUE) -- run via `conda run` so this script doesn't
# need to be launched from inside that env itself. Every other step runs in
# whatever plain venv/interpreter this script itself is invoked with.
#
# On a fresh machine/container (e.g. a DGX Spark Docker container based on
# nvcr.io/nvidia/pytorch), run setup_container.sh once first -- it installs
# this project (and, in a separate conda env, graph-tool + alertbert) into
# whatever environment this script then expects to already be set up. See
# setup_container.sh and Dockerfile.dgx (repo root) for that one-time setup.
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out (e.g. the thesis-alertbert conda env isn't set up on
# this machine), its exit code is logged and the run continues to the next
# step, including the final notebook execution, so partial results still
# get visualized. The notebook itself tolerates a missing results/*.json
# artifact (prints a "[skip] ... not found" note).
#
# GROUPING_DEVICE: "auto" (default, mps->cuda->cpu)/"cpu"/"cuda"/"mps"/...
#   -- forwarded to alertbert_sweep.py and deepcase_sweep.py, see thesis.grouping._device.
# GROUPING_RUN_FULL_ALERTBERT_SWEEP / GROUPING_RUN_FULL_DEEPCASE_SWEEP:
#   default to "1" here (full sweeps) -- override to "0" for a quick
#   smoke-test-only run.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/grouping/run_grouping.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
GROUPING_DIR="$REPO_ROOT/src/thesis/baselines/grouping"
cd "$GROUPING_DIR"

PYTHON="python3"
LOG_DIR="$GROUPING_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/grouping_run_$(date +%Y%m%d_%H%M%S).log"

export GROUPING_DEVICE="${GROUPING_DEVICE:-auto}"
export GROUPING_RUN_FULL_ALERTBERT_SWEEP="${GROUPING_RUN_FULL_ALERTBERT_SWEEP:-1}"
export GROUPING_RUN_FULL_DEEPCASE_SWEEP="${GROUPING_RUN_FULL_DEEPCASE_SWEEP:-1}"

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
    return 0  # never abort the batch on a single step's failure
}

{
    echo "=== Grouping run started at $(date) ==="
    echo "GROUPING_DEVICE=$GROUPING_DEVICE GROUPING_RUN_FULL_ALERTBERT_SWEEP=$GROUPING_RUN_FULL_ALERTBERT_SWEEP GROUPING_RUN_FULL_DEEPCASE_SWEEP=$GROUPING_RUN_FULL_DEEPCASE_SWEEP"

    run_step "fixed_window.py" "$PYTHON" fixed_window.py
    run_step "time_delta.py" "$PYTHON" time_delta.py
    run_step "cscas_grouping.py" "$PYTHON" cscas_grouping.py
    run_step "cscas_grouping_sensitivity.py" "$PYTHON" cscas_grouping_sensitivity.py

    export KMP_DUPLICATE_LIB_OK=TRUE
    run_step "alertbert_sweep.py" conda run -n thesis-alertbert --no-capture-output \
        "$PYTHON" alertbert_sweep.py

    run_step "deepcase_sweep.py" "$PYTHON" deepcase_sweep.py

    run_step "notebook execution" "$PYTHON" -m jupyter nbconvert \
        --to notebook --execute --inplace \
        ../../notebooks/baselines/grouping_comparison.ipynb

    echo ""
    echo "=== Grouping run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"
