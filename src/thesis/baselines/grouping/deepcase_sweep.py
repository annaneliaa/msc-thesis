"""
DeepCASE grouping baseline: two-tier sweep, cost-ordered explicitly rather
than a flat nested loop -- context length L (outer -- changes the
ContextBuilder's input width and attention size, so each value needs its
own preprocessing pass and its own training run) x DBSCAN eps (inner -- a
clustering-only parameter applied on top of one trained ContextBuilder's
already-computed attention vectors, via group_alerts_deepcase_many_eps, so
it never retrains or recomputes the attention-query optimization per eps).

Fixed (not swept): hidden_size=128 (default), min_samples=5, threshold=0.2
(the paper's confidence threshold tau_confidence), and a single fixed
seed=0 per L (not averaged over multiple SGD training runs).

Trained on the DeepCASE train scenarios (shaw/wardbeck/wheeler/wilson),
evaluated on TEST_SCENARIOS (fox/russellmitchell) -- same train/test
discipline as every other method here.

Outlier convention: a sequence can fail to land in a DBSCAN cluster for two
different reasons -- the ContextBuilder wasn't confident enough (below
threshold), or DBSCAN itself calls it noise. For comparability with the
other methods' metrics here, each such alert becomes its own singleton
group (group_alerts_deepcase's convention). Each outlier's `reason`
("deepcase_low_confidence" vs "deepcase_dbscan_noise") also feeds
build_manual_review_records, which turns the rejected alerts at each
(scenario, L)'s default eps (DEFAULT_EPS -- not every swept eps value,
which would inflate the queue with sweep artifacts) into
ManualReviewRecords -- the actual queue those alerts would be routed to for
analyst review, mirroring DeepCASE's own semi-automatic mode. Written to
results/deepcase_manual_review.json.

Runs fine in the plain venv (no conda env needed) -- unlike alertbert_sweep.py.

Env vars:
    GROUPING_DEVICE: "auto" (default, mps->cuda->cpu)/"cpu"/"cuda"/"mps"/...
        -- see thesis.grouping._device.resolve_device.
    GROUPING_RUN_FULL_DEEPCASE_SWEEP: "1" for the full grid (3 context
        lengths x 2 scenarios = 6 runs x ~20 min = ~2 hours, measured);
        unset/"0" (default) only checks the code path runs (1 scenario, 1
        context length, epochs=2, cluster_iterations=5 -- a couple
        minutes, not a scientifically meaningful result).

Writes results/deepcase.json (+ _sizes.npz) and
results/deepcase_manual_review.json.
"""

from __future__ import annotations

import os
import time

import numpy as np

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import (
    DEEPCASE_TRAIN_ID,
    TEST_SCENARIOS,
    load_deepcase_train_alerts,
    load_test_scenario_alerts,
)
from thesis.grouping.deepcase_grouping import (
    DEFAULT_EPS,
    group_alerts_deepcase_many_eps,
)
from thesis.grouping.group_alerts import build_manual_review_records

DEEPCASE_CONTEXT_LENGTHS_FULL = [2, 10, 20]  # short/medium/long, cut from 5 values
DEEPCASE_EPS_VALUES = [round(0.1 * i, 1) for i in range(1, 11)]

GROUPING_DEVICE = os.environ.get("GROUPING_DEVICE", "auto")
RUN_FULL_DEEPCASE_SWEEP = os.environ.get("GROUPING_RUN_FULL_DEEPCASE_SWEEP", "0") == "1"


def main() -> None:
    print(f"GROUPING_DEVICE={GROUPING_DEVICE!r}")
    print(f"RUN_FULL_DEEPCASE_SWEEP={RUN_FULL_DEEPCASE_SWEEP}")

    if RUN_FULL_DEEPCASE_SWEEP:
        deepcase_scenarios = TEST_SCENARIOS
        deepcase_context_lengths = DEEPCASE_CONTEXT_LENGTHS_FULL
        deepcase_epochs = 10
        deepcase_cluster_iterations = 100
    else:
        deepcase_scenarios = TEST_SCENARIOS[:1]
        deepcase_context_lengths = [2]
        deepcase_epochs = 2
        # cluster_iterations (the attention-query optimization step) is the
        # real bottleneck, not epochs -- reducing epochs alone leaves a
        # ~15-20 minute smoke test since it still runs 100 iterations over
        # every target alert.
        deepcase_cluster_iterations = 5

    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()
    deepcase_train_alerts = load_deepcase_train_alerts()

    rows: list[dict] = []
    size_arrays: dict[tuple, np.ndarray] = {}
    manual_review_rows: list[dict] = []
    for scenario in deepcase_scenarios:
        alerts = alerts_by_scenario[scenario]
        alert_index = alert_index_by_scenario[scenario]
        for L in deepcase_context_lengths:
            _start = time.perf_counter()
            results_by_eps = group_alerts_deepcase_many_eps(
                alerts,
                train_alerts=deepcase_train_alerts,
                train_id=DEEPCASE_TRAIN_ID,
                eps_values=DEEPCASE_EPS_VALUES,
                context_length=L,
                min_samples=5,
                threshold=0.2,
                seed=0,
                epochs=deepcase_epochs,
                cluster_iterations=deepcase_cluster_iterations,
                device=GROUPING_DEVICE,
            )
            # Shared cost across every eps value in this call for this
            # (scenario, L) pair -- logged identically on each resulting
            # row rather than divided among them, since training only
            # happens once.
            _elapsed = time.perf_counter() - _start
            for eps, records in results_by_eps.items():
                param_label = f"L={L}_eps={eps:.1f}"
                row, sizes = evaluate(
                    "deepcase",
                    param_label,
                    records,
                    alerts,
                    alert_index,
                    train_time_seconds=_elapsed,
                )
                row["scenario"] = scenario
                rows.append(row)
                size_arrays[("deepcase", param_label, scenario)] = sizes

            # Manual-review queue: only at this (scenario, L)'s default eps
            # (DEFAULT_EPS), not every swept eps value -- otherwise the
            # queue would be inflated with sweep artifacts rather than
            # reflecting one representative run's actual rejections.
            for review in build_manual_review_records(
                alerts, results_by_eps[DEFAULT_EPS]
            ):
                manual_review_rows.append(
                    {
                        "scenario": scenario,
                        "context_length": L,
                        "alert_id": review.alert_id,
                        "ts": review.ts,
                        "host": review.host,
                        "reason": review.reason,
                    }
                )

    save_grouping_results(
        "deepcase",
        "DeepCASE grouping: context_length x eps two-tier sweep.",
        rows,
        size_arrays,
    )
    save_grouping_results(
        "deepcase_manual_review",
        "DeepCASE manual-review queue: outlier alerts (DBSCAN noise / low-confidence) "
        "at each (scenario, context_length)'s default eps.",
        manual_review_rows,
    )


if __name__ == "__main__":
    main()
