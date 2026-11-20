#!/usr/bin/env bash
#
# Mining comparison, AIT-ADS arm (paper plan 7.6): leave-one-scenario-out
# over the AIT-ADS scenarios (cscas_grouping, token arm T2), all five methods
# + "none" x SEEDS, run in two stages so mining and training are separate jobs:
#
#   0. prepare  one job: build the per-scenario look-back cache
#               (artifacts/cache/<scenario>/lookback/), so later jobs only read it.
#   1. mine     one job per (held-out fold, method): mine -> gate -> dedupes,
#               cached per (fold, method, mining seed) before the budget K in
#               MINING_CACHE_DIR. Slowest methods are queued first.
#   2. train    one job per fold: load the cached candidates, budget K, encode,
#               fit + score the RF for every arm and seed.
#   3. merge    the per-fold runs into one run + summary and statistics.
#
# The mining cache persists across runs: rerunning after a crash only mines
# what's missing, and K / model variants (TRAIN_ONLY=1, K=...) skip mining.
#
# Memory safety (added after the 2026-11-16 run froze the shared DGX: 8 folds
# started at once, each holding every scenario's data, all reached their
# memory peak together):
#   - MINE_PARALLEL / TRAIN_PARALLEL jobs at a time; the rest queue.
#   - A queued job starts only STAGGER_SECONDS after the last start AND when
#     MemAvailable >= START_FREE_GB, so peaks don't line up.
#   - A watchdog kills the newest running job (its whole process group) when
#     MemAvailable < CRIT_FREE_GB. That job counts as failed and the run stops
#     before the next stage. Better one lost job than a frozen machine.
#   - Each job's peak memory is logged (GNU time -v on Linux, time -l on
#     macOS) to its log and summarised in peak_memory.txt.
# The memory checks read /proc/meminfo (Linux only); elsewhere they're off.
#
# Measure peak memory before a full run (sizes the *_PARALLEL settings):
#   TEST_SCENARIOS=fox SEEDS=0 MINE_PARALLEL=1 TRAIN_PARALLEL=1 \
#     src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
#
# Output: artifacts/experiments/mining_comparison_ait_ads/<ts>/
#   folds/<scenario>/   one trained run per held-out scenario
#   per_step_results.csv, features.csv, summary.csv, helps_hurts.csv,
#   stats.json, ...   the merged run (written once every fold has finished)
# Logs: artifacts/logs/mining_comparison_ait_ads/<ts>/: prepare.log,
#   mine__<scenario>__<method>.log, train__<scenario>.log, merge.log,
#   peak_memory.txt, killed.txt (watchdog kills, if any)
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
# Settings-sensitivity variants (output under mining_comparison_ait_ads_settings_<v>/):
#   SETTINGS_VARIANT=low  src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
#   SETTINGS_VARIANT=high src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
# Smoke test:
#   SCENARIOS="fox wilson" SEEDS="0" METHODS="eclat prefixspan" \
#     src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh

set -uo pipefail
set -m  # each background job gets its own process group (watchdog kills the group)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Testing only: run "$FOLD_CMD <job name>" instead of the real jobs, and read
# memory from a fake meminfo file.
FOLD_CMD="${FOLD_CMD:-}"
MEMINFO="${MEMINFO:-/proc/meminfo}"

if [[ -z "$FOLD_CMD" ]]; then
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
# Slowest first, so the long mining jobs start early.
METHODS="${METHODS:-mctsextent ripper prefixspan eclat contrast_tree}"
K="${K:-}"               # empty = COMPARISON_K
TRAIN_ONLY="${TRAIN_ONLY:-0}"   # 1 = skip prepare + mine (everything must be cached)
# Mining settings grid point (settings-sensitivity check): low | default | high.
SETTINGS_VARIANT="${SETTINGS_VARIANT:-default}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
MINE_PARALLEL="${MINE_PARALLEL:-$MAX_PARALLEL}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-$MAX_PARALLEL}"
STAGGER_SECONDS="${STAGGER_SECONDS:-300}"
START_FREE_GB="${START_FREE_GB:-40}"
CRIT_FREE_GB="${CRIT_FREE_GB:-12}"
POLL_SECONDS="${POLL_SECONDS:-15}"
MINING_CACHE_DIR="${MINING_CACHE_DIR:-$REPO_ROOT/artifacts/cache/mining_comparison}"

TS="$(date -u +%Y%m%d_%H%M%S)"
_suffix=""
[[ "$SETTINGS_VARIANT" != default ]] && _suffix="_settings_$SETTINGS_VARIANT"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/experiments/mining_comparison_ait_ads$_suffix}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/artifacts/logs/mining_comparison_ait_ads}"
OUT_DIR="$OUT_ROOT/$TS"
LOG_DIR="$LOG_ROOT/$TS"
mkdir -p "$OUT_DIR/folds" "$LOG_DIR"
RUNNING_FILE="$LOG_DIR/.running"   # "<pid> <job>" per running job, oldest first
KILLED_FILE="$LOG_DIR/killed.txt"
: >"$RUNNING_FILE"

RUNNER="$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_comparison.py"
# shellcheck disable=SC2206
COMMON=(--dataset ait_ads --scenarios $SCENARIOS --seeds $SEEDS
  --mining-cache-dir "$MINING_CACHE_DIR" --settings-variant "$SETTINGS_VARIANT")

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

peak_gb() {  # peak RSS of a finished job from its log, in GB
  awk '/Maximum resident set size \(kbytes\)/ {printf "%.1f", $NF / 1048576; exit}
       /maximum resident set size/ {printf "%.1f", $1 / 1073741824; exit}' "$1"
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
      local newest pid job
      newest="$(tail -n 1 "$RUNNING_FILE" 2>/dev/null)"
      [[ -z "$newest" ]] && continue
      read -r pid job <<<"$newest"
      echo "$(date -u +%H:%M:%S) MemAvailable ${avail}G < ${CRIT_FREE_GB}G: killing $job (pgid $pid)" \
        | tee -a "$KILLED_FILE"
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 20
      kill -KILL -- "-$pid" 2>/dev/null
    fi
  done
}

# --- job queue --------------------------------------------------------------
# Jobs are queued as JOB_NAMES[i] + JOB_ARGS[i] (runner args, word-split).

JOB_NAMES=()
JOB_ARGS=()
pids=()
names=()
failed=()
last_start=0

start_job() {
  local name="$1" args="$2" cmd
  if [[ -n "$FOLD_CMD" ]]; then
    # shellcheck disable=SC2206
    cmd=($FOLD_CMD "$name")
  else
    # shellcheck disable=SC2206
    cmd=(python -u "$RUNNER" "${COMMON[@]}" $args)
  fi
  echo "$(date -u +%H:%M:%S) start $name (free: $(mem_available_gb)G)"
  echo "${cmd[*]}" >"$LOG_DIR/$name.log"
  "${TIME_CMD[@]}" "${cmd[@]}" >>"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  names+=("$name")
  echo "$pid $name" >>"$RUNNING_FILE"
  last_start=$(date +%s)
}

reap() {  # collect finished jobs; keep the rest
  local i keep_p=() keep_n=()
  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}" name="${names[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      keep_p+=("$pid"); keep_n+=("$name")
      continue
    fi
    if wait "$pid"; then
      echo "$(date -u +%H:%M:%S) done   $name (peak $(peak_gb "$LOG_DIR/$name.log")G)"
    else
      echo "$(date -u +%H:%M:%S) FAILED $name -- see $LOG_DIR/$name.log"
      failed+=("$name")
    fi
    echo "$name $(peak_gb "$LOG_DIR/$name.log")G" >>"$LOG_DIR/peak_memory.txt"
    grep -v "^$pid " "$RUNNING_FILE" >"$RUNNING_FILE.tmp"; mv "$RUNNING_FILE.tmp" "$RUNNING_FILE"
  done
  pids=("${keep_p[@]+"${keep_p[@]}"}")
  names=("${keep_n[@]+"${keep_n[@]}"}")
}

can_start() {  # $1 = max jobs at once
  (( ${#pids[@]} < $1 )) || return 1
  (( ${#pids[@]} == 0 )) && return 0          # nothing running: never deadlock
  (( $(date +%s) - last_start >= STAGGER_SECONDS )) || return 1
  local avail
  avail="$(mem_available_gb)"
  [[ -z "$avail" ]] || (( avail >= START_FREE_GB ))
}

run_queue() {  # $1 = phase name, $2 = max parallel; runs JOB_* then clears them
  local phase="$1" par="$2" i
  echo "== $phase: ${#JOB_NAMES[@]} jobs, up to $par at once"
  for i in "${!JOB_NAMES[@]}"; do
    until can_start "$par"; do sleep "$POLL_SECONDS"; reap; done
    start_job "${JOB_NAMES[$i]}" "${JOB_ARGS[$i]}"
  done
  while (( ${#pids[@]} )); do sleep "$POLL_SECONDS"; reap; done
  JOB_NAMES=(); JOB_ARGS=()
  if (( ${#failed[@]} )); then
    echo "FAILED in $phase: ${failed[*]} -- stopping. Mined candidates so far stay" \
         "in $MINING_CACHE_DIR; rerun to resume."
    [[ -s "$KILLED_FILE" ]] && echo "Watchdog kills: $KILLED_FILE"
    exit 1
  fi
}

queue() { JOB_NAMES+=("$1"); JOB_ARGS+=("$2"); }

# --- run --------------------------------------------------------------------

echo "Output: $OUT_DIR"
echo "Logs:   $LOG_DIR"
echo "Folds:  $TEST_SCENARIOS (pool: $SCENARIOS) | methods: $METHODS | seeds: $SEEDS"
echo "Mining cache: $MINING_CACHE_DIR | settings variant: $SETTINGS_VARIANT"
echo "Safety: mine $MINE_PARALLEL / train $TRAIN_PARALLEL at once, ${STAGGER_SECONDS}s stagger," \
     "start at >= ${START_FREE_GB}G free, kill newest at < ${CRIT_FREE_GB}G free"

watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null' EXIT

k_args=""
[[ -n "$K" ]] && k_args="--k $K"

if [[ "$TRAIN_ONLY" != 1 ]]; then
  queue prepare "--stage prepare --methods $METHODS"
  run_queue prepare 1
  for m in $METHODS; do
    for s in $TEST_SCENARIOS; do
      queue "mine__${s}__${m}" \
        "--stage mine --test-scenarios $s --methods $m --output-dir $OUT_DIR/mine/${s}__${m}"
    done
  done
  run_queue mine "$MINE_PARALLEL"
fi

for s in $TEST_SCENARIOS; do
  queue "train__$s" \
    "--stage train --test-scenarios $s --methods $METHODS $k_args --output-dir $OUT_DIR/folds/$s"
done
run_queue train "$TRAIN_PARALLEL"

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
