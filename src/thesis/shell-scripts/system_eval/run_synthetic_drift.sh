#!/usr/bin/env bash
#
# Synthetic drift test (positive control for the monitor signals) on CSCAS.
# Injects controlled drift into the second half of the test period and logs
# which monitor signals fire. See thesis.system_eval.synthetic_drift.
#
# Env overrides: MODELS ("xgboost rf"), GRANULARITY (0.05), KINDS (all),
# SMOKE (1 = strongest fraction per kind only), N_JOBS (4).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, sklearn, numpy, pandas" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps." >&2
  exit 1
fi

read -r -a MODELS <<< "${MODELS:-xgboost rf}"
GRANULARITY="${GRANULARITY:-0.05}"
N_JOBS="${N_JOBS:-4}"

LOG_DIR="$REPO_ROOT/artifacts/logs/synthetic_drift"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S)_cscas.log"

cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_synthetic_drift.py" \
  --models "${MODELS[@]}" --granularity "$GRANULARITY" --n-jobs "$N_JOBS")
if [[ -n "${KINDS:-}" ]]; then
  read -r -a KIND_ARR <<< "$KINDS"
  cmd+=(--kinds "${KIND_ARR[@]}")
fi
if [[ "${SMOKE:-0}" -eq 1 ]]; then
  cmd+=(--smoke)
fi

echo "Logging to $log_file"
"${cmd[@]}" >"$log_file" 2>&1
status=$?
grep -E "Saved →|results →" "$log_file" | tail -n 5
exit $status
