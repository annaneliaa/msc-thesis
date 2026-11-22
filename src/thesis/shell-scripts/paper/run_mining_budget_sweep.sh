#!/usr/bin/env bash
#
# Budget robustness sweep for the mining comparison (HANDOFF "Deferred" 1):
# every method trained at K in {10, 25, 50, 60} per class and K = 60 pooled,
# all from the same cached mining output; checks whether the ranking holds.
# One process (mine stage once, then a training run per variant), so memory is
# one job's worth: a few GB, plus PrefixSpan's ~7 GB if AIT-ADS mining isn't
# cached yet.
#
# Usage (on the DGX):
#   DATASET=cscas src/thesis/shell-scripts/paper/run_mining_budget_sweep.sh
#   DATASET=ait_ads src/thesis/shell-scripts/paper/run_mining_budget_sweep.sh
#     (after the full AIT-ADS comparison run, so its mining is cached)
#   DATASET=ait_ads NO_HOUR=1 src/thesis/shell-scripts/paper/run_mining_budget_sweep.sh
#     (the AIT-ADS main protocol: hour_of_day left out of the base features)
# Smoke test:
#   DATASET=cscas MAX_STEPS=2 SEEDS=0 KS="10 60" POOLED_AT="" \
#     src/thesis/shell-scripts/paper/run_mining_budget_sweep.sh

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
if ! python -c "import thesis, sklearn, numpy, pandas, wittgenstein" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps." >&2
  exit 1
fi

DATASET="${DATASET:-cscas}"
KS="${KS:-10 25 50 60}"
POOLED_AT="${POOLED_AT-60}"   # empty = no pooled variant
SEEDS="${SEEDS:-0 1 2 3 4}"
MAX_STEPS="${MAX_STEPS:-}"
SKIP_MINE="${SKIP_MINE:-0}"
NO_HOUR="${NO_HOUR:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/mining_budget_sweep"
mkdir -p "$LOG_DIR"
tag="$DATASET"; [[ "$NO_HOUR" == 1 ]] && tag+="_no_hour"
log_file="$LOG_DIR/${tag}_$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_budget_sweep.py"
  --dataset "$DATASET" --ks $KS --pooled-at $POOLED_AT --seeds $SEEDS)
[[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")
[[ "$SKIP_MINE" == 1 ]] && cmd+=(--skip-mine)
[[ "$NO_HOUR" == 1 ]] && cmd+=(--no-hour)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi
