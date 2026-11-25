#!/usr/bin/env bash
#
# DeepCASE comparison, DeepCASE side of the B6 grid: DeepCASE frozen /
# fixed-schedule / monitor-triggered over the CSCAS test windows, reusing the
# step-5.2 run's tuned settings and cached W0 ContextBuilders. Uses the GPU
# (ContextBuilder retraining + attention queries) -- check the GPU is free.
#
# POLICY: never | schedule | combined.  AUDIT: auto | on | off.
# SCHEDULE_ACTION: REMINE_AND_RETRAIN (default; DeepCASE's own update) |
# RETRAIN_ONLY.  UPDATE_EPOCHS: ContextBuilder epochs per retrain.
#
# Usage (on the DGX):
#   POLICY=never src/thesis/shell-scripts/paper/run_deepcase_updates.sh   # = 5.2 check
#   POLICY=schedule src/thesis/shell-scripts/paper/run_deepcase_updates.sh
#   POLICY=combined src/thesis/shell-scripts/paper/run_deepcase_updates.sh
# Smoke test: SEEDS=0 POLICY=never ...

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, deepcase, torch, pandas, pyarrow" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import thesis/deepcase/torch/pyarrow." >&2
  exit 1
fi

BASE_RUN="${BASE_RUN:-$REPO_ROOT/artifacts/experiments/deepcase_comparison/cscas/20261123_200913}"
POLICY="${POLICY:-never}"
AUDIT="${AUDIT:-auto}"
AUDIT_RATE="${AUDIT_RATE:-0.05}"
SCHEDULE_ACTION="${SCHEDULE_ACTION:-REMINE_AND_RETRAIN}"
SEEDS="${SEEDS:-}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-}"

LOG_DIR="$REPO_ROOT/artifacts/logs/deepcase_updates"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S)_${POLICY}_audit-${AUDIT}.log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_deepcase_updates.py"
  --base-run "$BASE_RUN" --policy "$POLICY" --audit "$AUDIT"
  --audit-rate "$AUDIT_RATE" --schedule-action "$SCHEDULE_ACTION")
[[ -n "$SEEDS" ]] && cmd+=(--seeds $SEEDS)
[[ -n "$UPDATE_EPOCHS" ]] && cmd+=(--update-epochs "$UPDATE_EPOCHS")

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "seed [0-9]+:|Saved →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi
