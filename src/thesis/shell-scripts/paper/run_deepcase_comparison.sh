#!/usr/bin/env bash
#
# DeepCASE comparison, step 5.2: frozen DeepCASE on CSCAS. Tunes context
# length x tau x eps on W0 (paper App. D), trains on W0 x 5 seeds, simulated
# analyst (10 samples/cluster + outliers = B0), scores the 54-day test period
# per horizon window (gran 0.1).
#
# ContextBuilders are cached in artifacts/cache/deepcase/, so a rerun only
# repeats the attention query and the clustering.
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_deepcase_comparison.sh
# Paper defaults, no tuning:
#   NO_TUNE=1 src/thesis/shell-scripts/paper/run_deepcase_comparison.sh
# Smoke test:
#   SMOKE=1 src/thesis/shell-scripts/paper/run_deepcase_comparison.sh

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
  echo "FATAL: active python ($(command -v python)) can't import thesis/deepcase/torch/pyarrow." \
       "Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

GRAN="${GRAN:-0.1}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-100}"
TUNE_EPOCHS="${TUNE_EPOCHS:-}"      # empty = EPOCHS
TUNE_LENGTHS="${TUNE_LENGTHS:-5 10 20}"
MISSING_HOST="${MISSING_HOST:-none}"
DEVICE="${DEVICE:-}"
NO_TUNE="${NO_TUNE:-0}"
SMOKE="${SMOKE:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/deepcase_comparison"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_deepcase_comparison.py"
  --gran "$GRAN" --seeds $SEEDS --epochs "$EPOCHS" --tune-lengths $TUNE_LENGTHS
  --missing-host "$MISSING_HOST")
[[ -n "$TUNE_EPOCHS" ]] && cmd+=(--tune-epochs "$TUNE_EPOCHS")
[[ -n "$DEVICE" ]] && cmd+=(--device "$DEVICE")
[[ "$NO_TUNE" -eq 1 ]] && cmd+=(--no-tune)
[[ "$SMOKE" -eq 1 ]] && cmd+=(--smoke)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →|Tuned:" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi
