"""
Time-delta grouping baseline: log-scale sweep matching the original
method's own validated evaluation protocol (`delta = a * 2**i` for
`i = -7 ... 13`, `a in {1, 1.5}`), both host-split variants, every
TEST_SCENARIOS entry. Writes results/time_delta.json (+ _sizes.npz).
"""

from __future__ import annotations

import time

import numpy as np

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import load_test_scenario_alerts
from thesis.grouping.group_alerts import (
    group_alerts_time_delta,
    group_alerts_time_delta_host,
)


def time_delta_grid():
    for a in (1.0, 1.5):
        for i in range(-7, 14):
            yield a * (2**i)


TIME_DELTA_VALUES = sorted(set(time_delta_grid()))


def main() -> None:
    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()

    rows: list[dict] = []
    size_arrays: dict[tuple, np.ndarray] = {}
    for scenario, alerts in alerts_by_scenario.items():
        alert_index = alert_index_by_scenario[scenario]
        for delta in TIME_DELTA_VALUES:
            for fn, method_name in [
                (group_alerts_time_delta, "time_delta"),
                (group_alerts_time_delta_host, "time_delta_host"),
            ]:
                _start = time.perf_counter()
                records = fn(alerts, delta=delta)
                _elapsed = time.perf_counter() - _start
                param_label = f"{delta:.4f}"
                row, sizes = evaluate(
                    method_name,
                    param_label,
                    records,
                    alerts,
                    alert_index,
                    inference_time_seconds=_elapsed,
                )
                row["scenario"] = scenario
                rows.append(row)
                size_arrays[(method_name, param_label, scenario)] = sizes

    save_grouping_results(
        "time_delta",
        "Time-delta grouping: log-scale sweep over TIME_DELTA_VALUES, both host-split variants.",
        rows,
        size_arrays,
    )


if __name__ == "__main__":
    main()
