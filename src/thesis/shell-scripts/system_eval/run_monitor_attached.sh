#!/usr/bin/env bash
#
# Monitor Attached: for each scenario, runs run_monitor_attached.py over the
# parameter grid -- every entry in MINING_SETTINGS
# (configs/screening_mining_settings.yaml) crossed with GRANULARITIES below.
# Only feature_set='symbolic' configs ever produce a DynamicSchema, so the
# derived shortlist is filtered to those (see run_monitor_attached.py). For
# each config, mines/fits once on window 0's train split -- identical setup
# to run_monitor_drift.py -- then walks the schema/model/Vk forward one
# window at a time, but here a sustained monitor signal actually retrains or
# remines in place instead of only being logged. Every horizon is timed
# (fast-route serving cost, workload funnel) and every retrain/remine event
# is timed and diffed against the schema it replaced.
#
# PSI_THRESHOLD/CAL_THRESHOLD below should be the values selected by Drift
# Signal EDA's threshold sweep (03_monitor_signal_drift.ipynb, Analysis 3)
# for this scenario -- the defaults here are the untuned starting values,
# not a result.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_monitor_attached.sh
#
# Edit the variables below to change the scenario(s), granularities,
# thresholds, or monitor sensitivity.

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
GRANULARITIES=(0.1)
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with the other experiment scripts
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
PSI_THRESHOLD="0.1"    # Drift Signal EDA's selected Signal-1 elevation cutoff -- untuned default shown
CAL_THRESHOLD="0.10"   # Drift Signal EDA's selected Signal-2 elevation cutoff -- untuned default shown
MONITOR_CONSECUTIVE_WINDOWS="3"
MONITOR_MIN_SAMPLES_SIGNAL_2="30"
LATENCY_SAMPLE_N="200"  # total individually-timed single-alert-group calls per config; 0 disables

LOG_DIR="$REPO_ROOT/artifacts/logs/monitor_attached"
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

  # -u: unbuffered stdout -- see run_monitor_drift.sh for why this matters
  # when redirecting to a log file.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_monitor_attached.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}" \
    --threshold-mode "$THRESHOLD_MODE" \
    --psi-threshold "$PSI_THRESHOLD" \
    --cal-threshold "$CAL_THRESHOLD" \
    --monitor-consecutive-windows "$MONITOR_CONSECUTIVE_WINDOWS" \
    --monitor-min-samples-signal-2 "$MONITOR_MIN_SAMPLES_SIGNAL_2" \
    --latency-sample-n "$LATENCY_SAMPLE_N")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Monitor Attached results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  MONITOR ATTACHED SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi
