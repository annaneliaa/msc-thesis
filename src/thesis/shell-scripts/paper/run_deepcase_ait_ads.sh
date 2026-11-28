#!/usr/bin/env bash
#
# Ours vs DeepCASE end to end on AIT-ADS (secondary robustness check for the
# CSCAS DeepCASE comparison): frozen, within-scenario 70/30 split, same label
# budget, no hour_of_day for either side. See
# src/thesis/system_eval/deepcase_ait_ads.py.
#
#   * DeepCASE: one ContextBuilder + Interpreter per seed on the training
#     alerts of all 8 scenarios pooled; tuned on those alerts (each
#     scenario's training part split 50:50 by time) unless NO_TUNE=1;
#     simulated analyst (10 samples/cluster + outliers = B0).
#   * Ours: contrast tree + RF (and the no-mining arm) per scenario on
#     DeepCASE's per-scenario budget (stratified_b0, deepcase_items) and on
#     all training labels (full); scenarios without training attacks
#     (harrison, russellmitchell, santos) are DeepCASE-only.
#   * One process, sequential: GPU for DeepCASE, CPU for the RF/mining.
#
# ContextBuilders are cached in artifacts/cache/deepcase/ (keyed on the exact
# training rows), so a rerun only repeats the attention query + clustering.
# MINING_CACHE_DIR reuses the AIT-ADS mining comparison's contrast-tree cache
# for the "full" variant (same training rows and settings).
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_deepcase_ait_ads.sh
# Paper defaults for DeepCASE, no tuning:
#   NO_TUNE=1 src/thesis/shell-scripts/paper/run_deepcase_ait_ads.sh
# Smoke test (fox only, 1 seed, 2 epochs):
#   SMOKE=1 src/thesis/shell-scripts/paper/run_deepcase_ait_ads.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, deepcase, torch, sklearn, pandas, pyarrow" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import thesis/deepcase/torch/sklearn/pyarrow." \
       "Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

SCENARIOS="${SCENARIOS:-fox harrison russellmitchell santos shaw wardbeck wheeler wilson}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-}"     # empty = SCENARIOS
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-100}"
TUNE_EPOCHS="${TUNE_EPOCHS:-}"           # empty = EPOCHS
TUNE_LENGTHS="${TUNE_LENGTHS:-5 10 20}"
EVENT_FIELD="${EVENT_FIELD:-signature}"  # signature (alert name) | short
VARIANTS="${VARIANTS:-stratified_b0 deepcase_items full}"
DEVICE="${DEVICE:-}"
NO_TUNE="${NO_TUNE:-0}"
SMOKE="${SMOKE:-0}"
MINING_CACHE_DIR="${MINING_CACHE_DIR:-$REPO_ROOT/artifacts/cache/mining_comparison}"

LOG_DIR="$REPO_ROOT/artifacts/logs/deepcase_ait_ads"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_deepcase_ait_ads.py"
  --scenarios $SCENARIOS --seeds $SEEDS --epochs "$EPOCHS" --tune-lengths $TUNE_LENGTHS
  --event-field "$EVENT_FIELD" --budget-variants $VARIANTS
  --mining-cache-dir "$MINING_CACHE_DIR")
# shellcheck disable=SC2206
[[ -n "$EVAL_SCENARIOS" ]] && cmd+=(--eval-scenarios $EVAL_SCENARIOS)
[[ -n "$TUNE_EPOCHS" ]] && cmd+=(--tune-epochs "$TUNE_EPOCHS")
[[ -n "$DEVICE" ]] && cmd+=(--device "$DEVICE")
[[ "$NO_TUNE" -eq 1 ]] && cmd+=(--no-tune)
[[ "$SMOKE" -eq 1 ]] && cmd+=(--smoke)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →|Tuned:|Maximum resident" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi
