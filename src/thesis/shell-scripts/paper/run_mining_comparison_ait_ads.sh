#!/usr/bin/env bash
#
# Mining comparison, AIT-ADS arm (paper plan 7.6): leave-one-scenario-out,
# one process per held-out scenario, all folds in parallel, then one merge.
# Each fold trains on every other scenario (cscas_grouping, token arm T2) and
# runs all five methods + "none" x SEEDS. Folds are independent, so running
# them side by side gives the same results as one sequential run.
#
# Output: artifacts/experiments/mining_comparison_ait_ads/<ts>/
#   folds/<scenario>/   one run per held-out scenario
#   per_step_results.csv, features.csv, summary.csv, helps_hurts.csv,
#   stats.json, ...   the merged run (written once every fold has finished)
# Logs: artifacts/logs/mining_comparison_ait_ads/<ts>/<scenario>.log, merge.log
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
# Smoke test:
#   SCENARIOS="fox wilson" SEEDS="0" METHODS="eclat prefixspan" \
#     src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh

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
  echo "FATAL: active python ($(command -v python)) can't import the core deps" \
       "(incl. wittgenstein for RIPPER). Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

# The scenario pool every fold trains on; held-out folds run for all of them
# unless TEST_SCENARIOS narrows it.
SCENARIOS="${SCENARIOS:-fox harrison russellmitchell santos shaw wardbeck wheeler wilson}"
TEST_SCENARIOS="${TEST_SCENARIOS:-$SCENARIOS}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METHODS="${METHODS:-}"   # empty = all five
K="${K:-}"               # empty = COMPARISON_K

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/artifacts/experiments/mining_comparison_ait_ads/$TS"
LOG_DIR="$REPO_ROOT/artifacts/logs/mining_comparison_ait_ads/$TS"
mkdir -p "$OUT_DIR/folds" "$LOG_DIR"

echo "Output: $OUT_DIR"
echo "Logs:   $LOG_DIR"
echo "Folds:  $TEST_SCENARIOS (pool: $SCENARIOS) | seeds: $SEEDS"

pids=()
folds=()
for s in $TEST_SCENARIOS; do
  # shellcheck disable=SC2206
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_comparison.py"
    --dataset ait_ads --scenarios $SCENARIOS --test-scenarios "$s"
    --seeds $SEEDS --output-dir "$OUT_DIR/folds/$s")
  # shellcheck disable=SC2206
  [[ -n "$METHODS" ]] && cmd+=(--methods $METHODS)
  [[ -n "$K" ]] && cmd+=(--k "$K")
  echo "  start $s: ${cmd[*]}" | tee "$LOG_DIR/$s.log"
  "${cmd[@]}" >>"$LOG_DIR/$s.log" 2>&1 &
  pids+=($!)
  folds+=("$s")
done

failed=()
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  done   ${folds[$i]}"
  else
    echo "  FAILED ${folds[$i]} -- see $LOG_DIR/${folds[$i]}.log"
    failed+=("${folds[$i]}")
  fi
done

if (( ${#failed[@]} )); then
  echo "FAILED folds: ${failed[*]} -- not merging. Finished folds are in $OUT_DIR/folds/."
  exit 1
fi

runs=()
for s in $TEST_SCENARIOS; do runs+=("$OUT_DIR/folds/$s"); done
if python -u "$REPO_ROOT/src/thesis/scripts/system_eval/merge_mining_comparison.py" \
     --runs "${runs[@]}" --output-dir "$OUT_DIR" >"$LOG_DIR/merge.log" 2>&1; then
  grep -E "Saved →" "$LOG_DIR/merge.log" | sed 's/^/  /'
else
  echo "Merge FAILED -- see $LOG_DIR/merge.log"
  exit 1
fi
