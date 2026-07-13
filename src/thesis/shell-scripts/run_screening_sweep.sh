#!/usr/bin/env bash
#
# In-Window Baseline (Screening Sweep): for each scenario, runs
# run_screening_sweep.py over the full (mining_setting x granularity) grid
# defined in MINING_SETTINGS/GRANULARITIES below. Unlike
# sweep_attribute_schema.sh (which loops per (growth_rate, max_depth,
# mine_frac) combination itself), run_screening_sweep.py already sweeps its
# whole grid internally per scenario -- mined schemas are cached per
# (scenario, granularity, window, mining_setting) on disk (see
# thesis.mining.window_schema_cache), so rerunning this script only mines
# whatever isn't already cached.
#
# Usage:
#   src/thesis/shell-scripts/run_screening_sweep.sh
#
# Edit the variables below to change the grid, scenario(s), or window
# subsampling.

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
GRANULARITIES=(0.1 0.2 0.25 0.5 1)
# Set to a number (e.g. 3) to evaluate only N evenly-spaced windows per
# granularity instead of all of them; leave empty for the full sweep.
WINDOWS_PER_GRAN=""

LOG_DIR="$REPO_ROOT/artifacts/logs/screening_sweep"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  total=$((total + 1))
  log_file="$LOG_DIR/${RUN_TS}_${scenario}.log"

  echo "[$total] $scenario"

  # Built up incrementally (rather than expanding a possibly-empty array)
  # since "${empty_array[@]}" errors under `set -u` on bash <4.4 -- macOS's
  # default /usr/bin/bash is 3.2.
  cmd=(python "$REPO_ROOT/src/thesis/scripts/mining/run_screening_sweep.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}")
  if [[ -n "$WINDOWS_PER_GRAN" ]]; then
    cmd+=(--windows-per-gran "$WINDOWS_PER_GRAN")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Screening sweep results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  SWEEP SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi
