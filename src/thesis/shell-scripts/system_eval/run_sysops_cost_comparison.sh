#!/usr/bin/env bash
#
# System-operationality cost comparison: runs temporal_decay (never-retrain
# anchor), rolling_walk_forward (always-retrain anchor), and monitor_drift
# (observe-only -- feeds the Drift Signal EDA threshold sweep) back to back,
# all narrowed to ONE config -- single mining setting
# (gr3_md1_mda4_rounds2), single granularity (0.05), single model (rf), no
# baseline row, no SHAP/LIME -- so the three points on the never-retrain <->
# monitor <-> always-retrain spectrum finish in a manageable time and land
# on directly comparable fast-route/funnel/adaptation-cost columns (see
# thesis.experiments._shared.fast_route_and_funnel, shared by all three).
#
# This is a SEPARATE, narrower run from run_temporal_decay.sh /
# run_rolling_walk_forward.sh / run_monitor_drift.sh, which run the full
# {xgboost,rf} x {0.1,0.05} grid for the decay-curve/importance analysis --
# use those for that; use this one only for the operationality cost
# comparison. Same mining setting and source_split_mode, so results from
# this run's rf/gran=0.05 point are directly comparable to the corresponding
# cell of the full-grid runs.
#
# Monitor Attached is NOT included here -- it needs psi_threshold/
# cal_threshold selected by Drift Signal EDA's Analysis 3 sweep
# (03_monitor_signal_drift.ipynb), which in turn needs this script's
# monitor_drift run to have finished first. Run it as a separate, manual
# step once you have those two numbers:
#   src/thesis/shell-scripts/system_eval/run_monitor_attached.sh \
#     <psi_threshold> <cal_threshold> 3 30 \
#     --mining-settings src/thesis/configs/monitor_eda_mining_setting.yaml \
#     --granularities 0.05 --models rf --source-split-mode baseline_split
# (run_monitor_attached.sh doesn't take --granularities/--models overrides
# directly -- edit its GRANULARITIES/MODELS arrays to (0.05) / (rf) for a
# matching single-config run, same as this script does inline below.)
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_sysops_cost_comparison.sh

set -uo pipefail

CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi

SCENARIO="cscas"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"
GRANULARITY="0.05"
MODEL="rf"
SOURCE_SPLIT_MODE="baseline_split"
THRESHOLD_MODE="fixed"
MONITOR_CONSECUTIVE_WINDOWS="3"
MONITOR_MIN_SAMPLES_SIGNAL_2="30"

LOG_DIR="$REPO_ROOT/artifacts/logs/sysops_cost_comparison"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

run_step() {
  local name="$1"
  shift
  total=$((total + 1))
  local log_file="$LOG_DIR/${RUN_TS}_${name}.log"
  echo "[$total] $name"
  # -u: unbuffered stdout -- see run_monitor_drift.sh for why this matters
  # when redirecting to a log file.
  python -u "$@" >"$log_file" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$name")
  else
    echo "    OK — see $log_file"
  fi
}

run_step temporal_decay \
  "$REPO_ROOT/src/thesis/scripts/system_eval/run_temporal_decay.py" \
  "$SCENARIO" \
  --mining-settings "$MINING_SETTINGS" \
  --granularities "$GRANULARITY" \
  --models "$MODEL" \
  --no-baseline \
  --source-split-mode "$SOURCE_SPLIT_MODE" \
  --threshold-mode "$THRESHOLD_MODE" \
  --no-explanations

run_step rolling_walk_forward \
  "$REPO_ROOT/src/thesis/scripts/system_eval/run_rolling_walk_forward.py" \
  "$SCENARIO" \
  --mining-settings "$MINING_SETTINGS" \
  --granularities "$GRANULARITY" \
  --models "$MODEL" \
  --no-baseline \
  --source-split-mode "$SOURCE_SPLIT_MODE" \
  --threshold-mode "$THRESHOLD_MODE" \
  --no-explanations

run_step monitor_drift \
  "$REPO_ROOT/src/thesis/scripts/system_eval/run_monitor_drift.py" \
  "$SCENARIO" \
  --mining-settings "$MINING_SETTINGS" \
  --granularities "$GRANULARITY" \
  --models "$MODEL" \
  --no-baseline \
  --source-split-mode "$SOURCE_SPLIT_MODE" \
  --threshold-mode "$THRESHOLD_MODE" \
  --monitor-consecutive-windows "$MONITOR_CONSECUTIVE_WINDOWS" \
  --monitor-min-samples-signal-2 "$MONITOR_MIN_SAMPLES_SIGNAL_2"

echo
echo "============================================================"
echo "  SYSOPS COST COMPARISON SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed steps:"
  for name in "${failed[@]}"; do
    echo "  - $name"
  done
  exit 1
fi

echo
echo "Next: run 03_monitor_signal_drift.ipynb sections 1-9 against this"
echo "monitor_drift run, pick psi_threshold/cal_threshold from Analysis 3,"
echo "then run Monitor Attached manually (see this script's header comment)."
