#!/usr/bin/env bash
#
# Runs Experiment 2 (run_temporal_decay.sh) then Experiment 3
# (run_rolling_walk_forward.sh) back-to-back, unattended -- so both can be
# kicked off in one shot (e.g. on the DGX) instead of babysitting the first
# to finish before manually starting the second. Both sub-scripts already
# carry the settings locked down for the monitor EDA (SCENARIOS=(cscas),
# MODELS=(xgboost rf), GRANULARITIES=(0.1 0.05),
# MINING_SETTINGS=configs/monitor_eda_mining_setting.yaml -- the
# gr3_md1_mda4_rounds2 sys-eval-adopted two_tree point --
# SOURCE_SPLIT_MODE="baseline_split" locking W_src/step-0 to the CSCAS
# baseline's own train/test boundary regardless of granularity,
# CSCAS_FULL_SYMBOLIC=1), so this script takes no arguments and passes none
# through -- edit the variables inside each sub-script directly if a setting
# needs to change. Same "run every step regardless, report failures at the
# end" behavior as run_overnight.sh (which chains Experiment 3 then 4
# instead -- this one deliberately stops after Experiment 3, since
# Experiment 4 (the monitor) is the next thing to run once both of these
# have produced fresh results, not before).
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_decay_and_walkforward.sh
#   # or, to actually survive a closed terminal (e.g. over an SSH session to
#   # the DGX):
#   nohup src/thesis/shell-scripts/system_eval/run_decay_and_walkforward.sh \
#     > artifacts/logs/decay_and_walkforward_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &
#
# Each sub-script still writes its own per-scenario log under
# artifacts/logs/<experiment>/ as usual (temporal_decay/,
# rolling_walk_forward/) -- this script's own stdout is just start/end
# timestamps and each sub-script's SUMMARY lines.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Plain array of "label:script" pairs rather than an associative array --
# macOS's default /usr/bin/bash is 3.2, which doesn't have those.
steps=(
  "Experiment 2 (temporal decay):$HERE/run_temporal_decay.sh"
  "Experiment 3 (rolling walk-forward):$HERE/run_rolling_walk_forward.sh"
)

overall_failed=0

echo "DECAY + WALK-FORWARD RUN started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for step in "${steps[@]}"; do
  label="${step%%:*}"
  script="${step#*:}"
  echo
  echo "============================================================"
  echo "  START: $label  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "============================================================"
  "$script"
  status=$?
  echo "  END:   $label  ($(date -u +%Y-%m-%dT%H:%M:%SZ))  exit=$status"
  if [[ $status -ne 0 ]]; then
    overall_failed=1
    echo "  [warn] $label failed (exit $status) -- continuing to the next step anyway"
  fi
done

echo
echo "============================================================"
if [[ $overall_failed -eq 0 ]]; then
  echo "  DECAY + WALK-FORWARD RUN: all steps succeeded  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "  Next: run_monitor_drift.sh for the monitor EDA."
else
  echo "  DECAY + WALK-FORWARD RUN: at least one step failed -- check logs above  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
fi
echo "============================================================"
exit $overall_failed
