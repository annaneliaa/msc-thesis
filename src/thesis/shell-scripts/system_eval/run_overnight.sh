#!/usr/bin/env bash
#
# Overnight batch: runs Experiment 3 (run_rolling_walk_forward.sh) then
# Experiment 4 (run_monitor_drift.sh) back-to-back, unattended -- so both
# can be kicked off before bed instead of babysitting the first one to
# finish before manually starting the second. The two are independent (both
# just cross the mining-settings grid x GRANULARITIES on their own), so a
# failure in the first doesn't block the second -- this script runs every
# step regardless and reports failures at the end. Doesn't touch
# run_temporal_decay.sh (Experiment 2) -- start/queue that separately if
# it isn't already running.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_overnight.sh
#   # or, to actually survive a closed terminal overnight:
#   nohup src/thesis/shell-scripts/system_eval/run_overnight.sh > artifacts/logs/overnight_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &
#
# Each sub-script still writes its own per-scenario log under
# artifacts/logs/<experiment>/ as usual (rolling_walk_forward/,
# monitor_drift/) -- this script's own stdout is just start/end timestamps
# and each sub-script's SUMMARY lines.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Plain array of "label:script" pairs rather than an associative array --
# macOS's default /usr/bin/bash is 3.2, which doesn't have those.
steps=(
  "Experiment 3 (rolling walk-forward):$HERE/run_rolling_walk_forward.sh"
  "Experiment 4 (monitor drift):$HERE/run_monitor_drift.sh"
)

overall_failed=0

echo "OVERNIGHT RUN started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
  echo "  OVERNIGHT RUN: all steps succeeded  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
else
  echo "  OVERNIGHT RUN: at least one step failed -- check logs above  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
fi
echo "============================================================"
exit $overall_failed
