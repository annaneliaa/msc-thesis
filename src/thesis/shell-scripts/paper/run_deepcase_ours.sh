#!/usr/bin/env bash
#
# DeepCASE comparison, our side (implementation plan B3, and the B6 grid via
# POLICY/AUDIT): our system (RF + XGBoost, base(5) + mined, combined monitor)
# on CSCAS under DeepCASE's label budget, matched per seed to a finished
# deepcase_comparison run (step 5.2):
#   * all 6 days of W_src are training (TRAIN_FRAC=1.0, no held-out window 0),
#     like DeepCASE; horizons at granularity 0.1;
#   * the initial fit only sees B0 labels (one run per VARIANT):
#       stratified_b0            random stratified sample, size = DeepCASE's B0
#                                (cluster samples + outliers) for that seed
#       deepcase_items           exactly the items DeepCASE's analyst inspected
#       stratified_clusters      random stratified sample, size = DeepCASE's
#                                clusters-only budget (10 per cluster)
#       deepcase_cluster_samples exactly DeepCASE's 10-per-cluster samples
#   * updates (if the policy has any) train on the cumulative analyst pool;
#   * per-row predictions are saved for the seen/unseen-signature split (B4).
#
# POLICY: never (frozen, B3; default) | schedule (every window) | combined
# (the monitor). AUDIT: auto (only combined audits) | on | off.
#
# Output: artifacts/experiments/deepcase_comparison_ours/cscas/
#   <POLICY>_audit-<AUDIT>/<VARIANT>/seed<k>/cscas/<ts>/
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_deepcase_ours.sh
#   POLICY=schedule AUDIT=on src/thesis/shell-scripts/paper/run_deepcase_ours.sh
# Smoke test (one seed, one variant):
#   SEEDS=0 VARIANTS=stratified_b0 src/thesis/shell-scripts/paper/run_deepcase_ours.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, sklearn, pandas, pyarrow" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import thesis/sklearn/pandas/pyarrow." >&2
  exit 1
fi

DEEPCASE_RUN="${DEEPCASE_RUN:-$REPO_ROOT/artifacts/experiments/deepcase_comparison/cscas/20261123_200913}"
SEEDS="${SEEDS:-0 1 2 3 4}"
VARIANTS="${VARIANTS:-stratified_b0 deepcase_items stratified_clusters deepcase_cluster_samples}"
MODELS="${MODELS:-xgboost rf}"
POLICY="${POLICY:-never}"
AUDIT="${AUDIT:-auto}"
AUDIT_RATE="${AUDIT_RATE:-0.05}"
GRAN="${GRAN:-0.1}"
N_JOBS="${N_JOBS:-2}"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"

if [[ ! -f "$DEEPCASE_RUN/budget.csv" ]]; then
  echo "FATAL: no budget.csv in DEEPCASE_RUN=$DEEPCASE_RUN" >&2
  exit 1
fi

budget_for() {  # budget_for <seed> <column>
  python -c "import pandas as pd,sys; b=pd.read_csv(sys.argv[1]); print(int(b.loc[b.seed==int(sys.argv[2]), sys.argv[3]].iloc[0]))" \
    "$DEEPCASE_RUN/budget.csv" "$1" "$2"
}

OUT_ROOT="$REPO_ROOT/artifacts/experiments/deepcase_comparison_ours/cscas/${POLICY}_audit-${AUDIT}"
LOG_DIR="$REPO_ROOT/artifacts/logs/deepcase_comparison_ours/$(date -u +%Y%m%d_%H%M%S)_${POLICY}_audit-${AUDIT}"
mkdir -p "$LOG_DIR"
echo "DeepCASE run: $DEEPCASE_RUN"
echo "Policy: $POLICY, audit: $AUDIT, variants: $VARIANTS, seeds: $SEEDS"
echo "Logs: $LOG_DIR"

failed=()
for seed in $SEEDS; do
  items="$DEEPCASE_RUN/seed${seed}/analyst_items.parquet"
  for variant in $VARIANTS; do
    budget_args=()
    case "$variant" in
      stratified_b0)
        budget_args=(--initial-budget-mode stratified
          --initial-budget-size "$(budget_for "$seed" budget_b0)") ;;
      stratified_clusters)
        budget_args=(--initial-budget-mode stratified
          --initial-budget-size "$(budget_for "$seed" budget_clusters_only)") ;;
      deepcase_items | deepcase_cluster_samples)
        budget_args=(--initial-budget-mode "$variant" --deepcase-items "$items") ;;
      *) echo "unknown variant $variant" >&2; failed+=("$variant/seed$seed"); continue ;;
    esac

    log_file="$LOG_DIR/${variant}_seed${seed}.log"
    # shellcheck disable=SC2206
    cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_monitor_attached.py" cscas
      --mining-settings "$MINING_SETTINGS"
      --granularities "$GRAN" --models $MODELS
      --source-split-mode baseline_split --train-frac 1.0 --threshold-mode fixed
      --trigger "$POLICY" --schedule-action REMINE_AND_RETRAIN
      --feedback-consecutive-windows 1 --pattern-label-mode audit
      --update-labels cumulative_budget --audit "$AUDIT" --audit-rate "$AUDIT_RATE"
      "${budget_args[@]}"
      --seed "$seed" --model-seed "$seed" --save-predictions
      --latency-sample-n 0 --n-jobs "$N_JOBS"
      --output-dir "$OUT_ROOT/$variant/seed${seed}")
    echo "[$variant seed $seed] $(date -u +%H:%M:%S)"
    if "${cmd[@]}" >"$log_file" 2>&1; then
      grep -E "\[budget\]" "$log_file" | head -1 | sed 's/^/    /'
    else
      echo "    FAILED — see $log_file"
      failed+=("$variant/seed$seed")
    fi
  done
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed: ${failed[*]}"
  exit 1
fi
echo "Done → $OUT_ROOT"
