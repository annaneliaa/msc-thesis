#!/usr/bin/env bash
#
# Pre-mine attribute schemas for cscas across a grid of mining parameters,
# via mine_attribute_schema.py --windowed. Each MINE_FRACS value is treated as
# a granularity: instead of mining once on the first mine_frac fraction of the
# timeline, --windowed mines one schema per chronological window of that size,
# covering the *entire* timeline (mine_frac=0.1 -> 10 windows, mine_frac=1.0 ->
# 1 window). Every (growth_rate, max_depth, granularity, window) combination is
# cached on disk (see thesis.mining.attribute_schema_cache) keyed by its own
# fingerprint, so this sweep is idempotent -- rerunning it only mines whatever
# isn't already cached.
#
# This means the grid isn't just 3x3x5=45 schemas -- for each (growth_rate,
# max_depth) pair, granularity 0.1/0.2/0.33/0.5/1.0 contributes 10/5/3/2/1
# windows respectively (21 total), so the full grid is 9*21=189 schemas,
# giving every grid point at every point in the timeline for later analysis
# (e.g. checking whether a configuration's mined schema is stable over time,
# not just over how much data it sees).
#
# Usage:
#   src/thesis/shell-scripts/sweep_attribute_schema.sh
#
# Edit the arrays below to change the grid or the scenario(s).

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
MIN_GROWTH_RATES=(3.0 4.0 5.0)
MAX_DEPTHS=(3 4 5)
MINE_FRACS=(0.1 0.2 0.33 0.5 1.0)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/logs/attribute_schema_sweep"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  for growth_rate in "${MIN_GROWTH_RATES[@]}"; do
    for max_depth in "${MAX_DEPTHS[@]}"; do
      for mine_frac in "${MINE_FRACS[@]}"; do
        total=$((total + 1))
        tag="${scenario}_gr${growth_rate}_md${max_depth}_mf${mine_frac}"
        log_file="$LOG_DIR/${RUN_TS}_${tag}.log"

        echo "[$total] $tag"
        python "$REPO_ROOT/src/thesis/scripts/mining/mine_attribute_schema.py" \
          "$scenario" \
          --min-growth-rate "$growth_rate" \
          --max-depth "$max_depth" \
          --mine-frac "$mine_frac" \
          --windowed \
          >"$log_file" 2>&1

        if [[ $? -ne 0 ]]; then
          echo "    FAILED — see $log_file"
          failed+=("$tag")
        else
          grep -E "windowed|window .*/.*\]" "$log_file" | tail -n 3 | sed 's/^/    /'
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "  SWEEP SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed combinations:"
  for tag in "${failed[@]}"; do
    echo "  - $tag"
  done
  exit 1
fi
