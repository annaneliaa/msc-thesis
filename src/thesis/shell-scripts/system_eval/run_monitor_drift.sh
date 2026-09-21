#!/usr/bin/env bash
#
# Drift-Monitor Evaluation (Experiment 4, observe-only): for each scenario,
# runs run_monitor_drift.py over the parameter grid -- MINING_SETTINGS below
# crossed with GRANULARITIES x MODELS, plus a baseline row per
# (granularity, model). Locked to the same "monitor EDA" scope as
# run_temporal_decay.sh / run_rolling_walk_forward.sh (see those scripts'
# own comments) so all three experiments' horizons/configs line up and can
# be overlaid directly: single mining setting gr3_md1_mda4_rounds2 (="two_tree",
# the config baselines/_mining_modes.py's CSCAS_MINING_MODE=two_tree uses),
# models in {xgboost, rf}, granularities in {0.1, 0.05}, source window
# anchored to the CSCAS baseline's own train/test boundary
# (source_split_mode=baseline_split). Edit MINING_SETTINGS to point at a
# different grid if you deliberately want to diverge from that lock.
#
# For each resulting config, mines/fits once on W_src's train split -- plus,
# for symbolic configs, builds a deployment-scoped DynamicSchema (Vk) from
# that same mining pass -- then walks the frozen schema/model/Vk forward one
# window at a time, running the drift monitor at every horizon and logging
# every signal/alarm it raises. The monitor never triggers an actual
# re-mine/retrain here -- it only observes and records what it would have
# done. Unlike the other two experiments, mining is never cached here (see
# system_eval/monitor_drift.py's module docstring), so every run mines fresh
# for symbolic configs.
#
# CONSECUTIVE_WINDOWS/MIN_SAMPLES_SIGNAL_2 are required arguments, not
# defaults baked into this script -- they gate what "elevated"/"action" this
# run logs, and Analysis 3 in 03_monitor_signal_drift.ipynb treats this
# run's own MIN_SAMPLES_SIGNAL_2 as a hard floor it can't sweep below
# without a rerun, so a silently-stale value here would silently cap that
# sweep too.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_monitor_drift.sh <consecutive_windows> <min_samples_signal_2>
#
# Example:
#   src/thesis/shell-scripts/system_eval/run_monitor_drift.sh 3 30
#
# Edit SCENARIOS/MINING_SETTINGS/GRANULARITIES/MODELS below to change what
# else runs.

set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <consecutive_windows> <min_samples_signal_2>" >&2
  echo "  e.g. $0 3 30" >&2
  exit 1
fi
MONITOR_CONSECUTIVE_WINDOWS="$1"
MONITOR_MIN_SAMPLES_SIGNAL_2="$2"

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"
GRANULARITIES=(0.1 0.05)
MODELS=(xgboost rf)
SOURCE_SPLIT_MODE="baseline_split"  # anchors W_src to the CSCAS baseline's own train/test boundary
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with the other experiment scripts
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall

LOG_DIR="$REPO_ROOT/artifacts/logs/monitor_drift"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  total=$((total + 1))
  log_file="$LOG_DIR/${RUN_TS}_${scenario}.log"

  echo "[$total] $scenario"

  if [[ ! -f "$MINING_SETTINGS" ]]; then
    echo "    FAILED — no mining-settings grid at $MINING_SETTINGS"
    failed+=("$scenario")
    continue
  fi

  # Built up incrementally (rather than expanding a possibly-empty array)
  # since "${empty_array[@]}" errors under `set -u` on bash <4.4 -- macOS's
  # default /usr/bin/bash is 3.2.
  # -u: unbuffered stdout -- without it, redirecting to $log_file makes
  # Python fully block-buffer stdout (prints only flush every ~8KB or at
  # exit) while warnings.warn() writes straight to unbuffered stderr, so the
  # log looks stuck spewing only warnings for the whole run with none of the
  # "[n/4] ..."/"Saved →" progress prints showing up until process exit.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_monitor_drift.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}" \
    --models "${MODELS[@]}" \
    --source-split-mode "$SOURCE_SPLIT_MODE" \
    --threshold-mode "$THRESHOLD_MODE" \
    --monitor-consecutive-windows "$MONITOR_CONSECUTIVE_WINDOWS" \
    --monitor-min-samples-signal-2 "$MONITOR_MIN_SAMPLES_SIGNAL_2")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Monitor drift results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  MONITOR DRIFT SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi
