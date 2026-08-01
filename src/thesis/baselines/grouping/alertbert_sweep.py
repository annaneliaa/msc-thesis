"""
AlertBERT grouping baseline: delta/theta swept separately, matching the
AlertBERT paper's own asymmetric design (alertbert/eval_grouping.py) --
delta only does cheap pre-clustering in AlertBERT's algorithm (not the
actual grouping signal), so the paper swept it narrowly; theta is the real
clustering-distance threshold, swept broadly. Pairs where theta < delta are
skipped (group_alerts_alertbert raises on those -- theta is the max
possible scaled cosine distance, so it must be >= delta).

Uses the pretrained mlm_1l_4h_16d_original_1_60k checkpoint
(external/AlertBERT/saved_models) for inference only -- no training. That
checkpoint was trained on shaw/wardbeck/wheeler/wilson, so running it on
TEST_SCENARIOS = ["fox", "russellmitchell"] evaluates it
out-of-distribution, same as every other method here is evaluated on
held-out scenarios -- kept this way deliberately for a fair cross-method
comparison.

**Requires the `thesis-alertbert` conda env** (alertbert.models hard-imports
graph_tool, conda/mamba-only, not in this repo's plain venv/) with
KMP_DUPLICATE_LIB_OK=TRUE set (numpy/torch both bundle libomp on macOS,
which otherwise SIGABRTs on import):

    conda activate thesis-alertbert
    KMP_DUPLICATE_LIB_OK=TRUE python3 alertbert_sweep.py

Env vars:
    GROUPING_DEVICE: "auto" (default, mps->cuda->cpu)/"cpu"/"cuda"/"mps"/...
        -- see thesis.grouping._device.resolve_device.
    GROUPING_RUN_FULL_ALERTBERT_SWEEP: "1" for the full 39-pair grid
        (~30s/pair, ~39 min total, measured); unset/"0" (default) runs a
        small 3x3 smoke-test subset instead (~6 min).

Writes results/alertbert.json (+ _sizes.npz).
"""

from __future__ import annotations

import os
import time

import numpy as np

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import load_test_scenario_alerts
from thesis.grouping.group_alerts import group_alerts_alertbert

# delta: the AlertBERT paper's own alertbert_deltas list
# (alertbert/eval_grouping.py) -- narrow on purpose, since delta only does
# cheap pre-clustering in AlertBERT's algorithm, not the actual grouping
# signal, so the paper didn't sweep it widely.
ALERTBERT_DELTAS = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]

# theta: the real clustering-distance threshold -- the paper swept this
# broadly (alertbert_theta_roc_traj_all spans ~0.0078 to ~4096). This is a
# coarser log-spaced subset of that same wide range, not the narrow delta
# list -- reusing the delta list for theta too would collapse that
# asymmetry.
ALERTBERT_THETAS = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]

GROUPING_DEVICE = os.environ.get("GROUPING_DEVICE", "auto")
RUN_FULL_ALERTBERT_SWEEP = (
    os.environ.get("GROUPING_RUN_FULL_ALERTBERT_SWEEP", "0") == "1"
)


def main() -> None:
    print(f"GROUPING_DEVICE={GROUPING_DEVICE!r}")
    print(f"RUN_FULL_ALERTBERT_SWEEP={RUN_FULL_ALERTBERT_SWEEP}")

    if RUN_FULL_ALERTBERT_SWEEP:
        alertbert_deltas = ALERTBERT_DELTAS
        alertbert_thetas = ALERTBERT_THETAS
    else:
        alertbert_deltas = [2.0, 8.0, 24.0]
        alertbert_thetas = [4.0, 64.0, 1024.0]

    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()

    rows: list[dict] = []
    size_arrays: dict[tuple, np.ndarray] = {}
    for scenario, alerts in alerts_by_scenario.items():
        alert_index = alert_index_by_scenario[scenario]
        for delta in alertbert_deltas:
            for theta in alertbert_thetas:
                if theta < delta:
                    continue
                _start = time.perf_counter()
                records = group_alerts_alertbert(
                    alerts, delta=delta, theta=theta, device=GROUPING_DEVICE
                )
                _elapsed = time.perf_counter() - _start
                param_label = f"d={delta:.4f}_t={theta:.4f}"
                row, sizes = evaluate(
                    "alertbert",
                    param_label,
                    records,
                    alerts,
                    alert_index,
                    inference_time_seconds=_elapsed,
                )
                row["scenario"] = scenario
                rows.append(row)
                size_arrays[("alertbert", param_label, scenario)] = sizes

    save_grouping_results(
        "alertbert",
        "AlertBERT grouping: delta/theta swept separately (asymmetric grid matching the paper's own protocol).",
        rows,
        size_arrays,
    )


if __name__ == "__main__":
    main()
