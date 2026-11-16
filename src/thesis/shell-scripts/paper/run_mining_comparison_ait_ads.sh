#!/usr/bin/env bash
#
# Mining comparison, AIT-ADS arm (paper plan 7.6): leave-one-scenario-out,
# one process per held-out scenario, then one merge. Each fold trains on every
# other scenario (cscas_grouping, token arm T2) and runs all five methods +
# "none" x SEEDS. Folds are independent, so running them side by side gives
# the same results as one sequential run.
#
# Memory safety (added after the 2026-11-16 run froze the shared DGX: 8 folds
# started at once, each holding every scenario's data, all reached their
# memory peak together):
#   - MAX_PARALLEL folds at a time; the rest queue.
#   - A queued fold starts only after STAGGER_SECONDS since the last start AND
#     when MemAvailable >= START_FREE_GB, so peaks don't line up.
#   - A watchdog kills the newest running fold (its whole process group) when
#     MemAvailable < CRIT_FREE_GB. That fold counts as failed; nothing is
#     merged. Better one lost fold than a frozen machine.
#   - Each fold's peak memory is logged (GNU time -v on Linux, time -l on
#     macOS) to <log dir>/<scenario>.log and summarised in peak_memory.txt.
# The memory checks read /proc/meminfo (Linux only); elsewhere they're off.
#
# Measure one fold's real peak before a full run (sizes MAX_PARALLEL):
#   TEST_SCENARIOS=fox SEEDS=0 MAX_PARALLEL=1 \
#     src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
#
# Output: artifacts/experiments/mining_comparison_ait_ads/<ts>/
#   folds/<scenario>/   one run per held-out scenario
#   per_step_results.csv, features.csv, summary.csv, helps_hurts.csv,
#   stats.json, ...   the merged run (written once every fold has finished)
# Logs: artifacts/logs/mining_comparison_ait_ads/<ts>/<scenario>.log,
#   merge.log, peak_memory.txt, killed.txt (watchdog kills, if any)
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
# Smoke test:
#   SCENARIOS="fox wilson" SEEDS="0" METHODS="eclat prefixspan" \
#     src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh

set -uo pipefail
set -m  # each background fold gets its own process group (watchdog kills the group)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -z "${FOLD_CMD:-}" ]]; then
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
fi

# The scenario pool every fold trains on; held-out folds run for all of them
# unless TEST_SCENARIOS narrows it.
SCENARIOS="${SCENARIOS:-fox harrison russellmitchell santos shaw wardbeck wheeler wilson}"
TEST_SCENARIOS="${TEST_SCENARIOS:-$SCENARIOS}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METHODS="${METHODS:-}"   # empty = all five
K="${K:-}"               # empty = COMPARISON_K
MAX_PARALLEL="${MAX_PARALLEL:-3}"
STAGGER_SECONDS="${STAGGER_SECONDS:-300}"
START_FREE_GB="${START_FREE_GB:-40}"
CRIT_FREE_GB="${CRIT_FREE_GB:-12}"
POLL_SECONDS="${POLL_SECONDS:-15}"
# Testing only: run "$FOLD_CMD <scenario>" instead of the real fold, and read
# memory from a fake meminfo file.
FOLD_CMD="${FOLD_CMD:-}"
MEMINFO="${MEMINFO:-/proc/meminfo}"

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/experiments/mining_comparison_ait_ads}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/artifacts/logs/mining_comparison_ait_ads}"
OUT_DIR="$OUT_ROOT/$TS"
LOG_DIR="$LOG_ROOT/$TS"
mkdir -p "$OUT_DIR/folds" "$LOG_DIR"
RUNNING_FILE="$LOG_DIR/.running"   # "<pid> <scenario>" per running fold, oldest first
KILLED_FILE="$LOG_DIR/killed.txt"
: >"$RUNNING_FILE"

# --- memory -----------------------------------------------------------------

mem_available_gb() {  # prints MemAvailable in whole GB, or nothing if unknown
  [[ -r "$MEMINFO" ]] || return 0
  awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' "$MEMINFO"
}

TIME_CMD=()
if /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -v)   # GNU: "Maximum resident set size (kbytes): N"
elif /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -l)   # BSD/macOS: "N  maximum resident set size" (bytes)
fi

peak_gb() {  # peak RSS of a finished fold from its log, in GB
  local log="$1"
  awk '/Maximum resident set size \(kbytes\)/ {printf "%.1f", $NF / 1048576; exit}
       /maximum resident set size/ {printf "%.1f", $1 / 1073741824; exit}' "$log"
}

if [[ -z "$(mem_available_gb)" ]]; then
  echo "  [warn] $MEMINFO not readable: memory guard and watchdog are off"
fi

# --- watchdog ---------------------------------------------------------------

watchdog() {
  while true; do
    sleep "$POLL_SECONDS"
    local avail
    avail="$(mem_available_gb)"
    [[ -z "$avail" ]] && continue
    if (( avail < CRIT_FREE_GB )); then
      local newest pid scen
      newest="$(tail -n 1 "$RUNNING_FILE" 2>/dev/null)"
      [[ -z "$newest" ]] && continue
      read -r pid scen <<<"$newest"
      echo "$(date -u +%H:%M:%S) MemAvailable ${avail}G < ${CRIT_FREE_GB}G: killing fold $scen (pgid $pid)" \
        | tee -a "$KILLED_FILE"
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 20
      kill -KILL -- "-$pid" 2>/dev/null
    fi
  done
}

# --- run --------------------------------------------------------------------

echo "Output: $OUT_DIR"
echo "Logs:   $LOG_DIR"
echo "Folds:  $TEST_SCENARIOS (pool: $SCENARIOS) | seeds: $SEEDS"
echo "Safety: max $MAX_PARALLEL folds, ${STAGGER_SECONDS}s stagger, start at >= ${START_FREE_GB}G free," \
     "kill newest at < ${CRIT_FREE_GB}G free"

watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null' EXIT

pids=()       # running pids (parallel to scens)
scens=()
failed=()
last_start=0

start_fold() {
  local s="$1" cmd
  if [[ -n "$FOLD_CMD" ]]; then
    # shellcheck disable=SC2206
    cmd=($FOLD_CMD "$s")
  else
    # shellcheck disable=SC2206
    cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_comparison.py"
      --dataset ait_ads --scenarios $SCENARIOS --test-scenarios "$s"
      --seeds $SEEDS --output-dir "$OUT_DIR/folds/$s")
    # shellcheck disable=SC2206
    [[ -n "$METHODS" ]] && cmd+=(--methods $METHODS)
    [[ -n "$K" ]] && cmd+=(--k "$K")
  fi
  echo "$(date -u +%H:%M:%S) start $s (free: $(mem_available_gb)G): ${cmd[*]}" | tee "$LOG_DIR/$s.log"
  "${TIME_CMD[@]}" "${cmd[@]}" >>"$LOG_DIR/$s.log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  scens+=("$s")
  echo "$pid $s" >>"$RUNNING_FILE"
  last_start=$(date +%s)
}

reap() {  # collect finished folds; keep the rest in pids/scens and RUNNING_FILE
  local i keep_p=() keep_s=()
  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}" s="${scens[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      keep_p+=("$pid"); keep_s+=("$s")
      continue
    fi
    if wait "$pid"; then
      echo "$(date -u +%H:%M:%S) done   $s (peak $(peak_gb "$LOG_DIR/$s.log")G)"
    else
      echo "$(date -u +%H:%M:%S) FAILED $s -- see $LOG_DIR/$s.log"
      failed+=("$s")
    fi
    echo "$s $(peak_gb "$LOG_DIR/$s.log")G" >>"$LOG_DIR/peak_memory.txt"
    grep -v "^$pid " "$RUNNING_FILE" >"$RUNNING_FILE.tmp"; mv "$RUNNING_FILE.tmp" "$RUNNING_FILE"
  done
  pids=("${keep_p[@]+"${keep_p[@]}"}")
  scens=("${keep_s[@]+"${keep_s[@]}"}")
}

can_start() {
  (( ${#pids[@]} < MAX_PARALLEL )) || return 1
  (( ${#pids[@]} == 0 )) && return 0          # nothing running: never deadlock
  (( $(date +%s) - last_start >= STAGGER_SECONDS )) || return 1
  local avail
  avail="$(mem_available_gb)"
  [[ -z "$avail" ]] || (( avail >= START_FREE_GB ))
}

for s in $TEST_SCENARIOS; do
  until can_start; do sleep "$POLL_SECONDS"; reap; done
  start_fold "$s"
done
while (( ${#pids[@]} )); do sleep "$POLL_SECONDS"; reap; done

if (( ${#failed[@]} )); then
  echo "FAILED folds: ${failed[*]} -- not merging. Finished folds are in $OUT_DIR/folds/."
  [[ -s "$KILLED_FILE" ]] && echo "Watchdog kills: $KILLED_FILE"
  exit 1
fi

[[ -n "$FOLD_CMD" ]] && { echo "FOLD_CMD test run: skipping merge"; exit 0; }

runs=()
for s in $TEST_SCENARIOS; do runs+=("$OUT_DIR/folds/$s"); done
if python -u "$REPO_ROOT/src/thesis/scripts/system_eval/merge_mining_comparison.py" \
     --runs "${runs[@]}" --output-dir "$OUT_DIR" >"$LOG_DIR/merge.log" 2>&1; then
  grep -E "Saved →" "$LOG_DIR/merge.log" | sed 's/^/  /'
else
  echo "Merge FAILED -- see $LOG_DIR/merge.log"
  exit 1
fi
