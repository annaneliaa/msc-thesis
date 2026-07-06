#!/usr/bin/env bash
#
# Pre-mine attribute schemas for cscas across a grid of mining parameters,
# via mine_attribute_schema.py. Each combination is cached on disk (see
# thesis.mining.attribute_schema_cache) keyed by its own fingerprint, so
# later comparison runs (run_model_comparison_attribute.py) pick up the
# matching schema instead of re-mining, as long as their --min-growth-rate/
# --max-depth/--mine-frac match one of the combinations mined here.
#
# Usage:
#   src/thesis/scripts/mining/sweep_attribute_schema.sh
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
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
          >"$log_file" 2>&1

        if [[ $? -ne 0 ]]; then
          echo "    FAILED — see $log_file"
          failed+=("$tag")
        else
          tail -n 2 "$log_file" | sed 's/^/    /'
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
