#!/usr/bin/env bash
#
# Synthetic drift test for the monitor on frozen DeepCASE (labels unchanged):
# which monitor signals fire when attacks are disguised as benign, benign as
# attack, new signatures appear, contexts shift, or the benign mix shifts?
# GPU: one attention query per (seed, injection), ~7-9 min each.
#
# Usage (on the DGX; check the GPU is free first):
#   src/thesis/shell-scripts/paper/run_deepcase_synthetic_drift.sh
#   SEEDS="0 1 2" src/thesis/shell-scripts/paper/run_deepcase_synthetic_drift.sh
# Smoke test: KINDS=new_signature src/thesis/shell-scripts/paper/run_deepcase_synthetic_drift.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, deepcase, torch, pandas, pyarrow" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import thesis/deepcase/torch/pyarrow." >&2
  exit 1
fi

BASE_RUN="${BASE_RUN:-$REPO_ROOT/artifacts/experiments/deepcase_comparison/cscas/20261125_150901}"
SEEDS="${SEEDS:-0}"
INJECT_FROM_H="${INJECT_FROM_H:-6}"
KINDS="${KINDS:-}"

LOG_DIR="$REPO_ROOT/artifacts/logs/deepcase_synthetic_drift"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"
# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_deepcase_synthetic_drift.py"
  --base-run "$BASE_RUN" --seeds $SEEDS --inject-from-h "$INJECT_FROM_H")
[[ -n "$KINDS" ]] && cmd+=(--kinds $KINDS)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "done \(|Saved →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi
