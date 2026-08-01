"""
Fixed-window grouping baseline: linear sweep over an absolute group-size
parameter (not a gap threshold, so no log-scale sweep here -- see
time_delta.py for why that distinction matters), both host-split variants,
every TEST_SCENARIOS entry. Writes results/fixed_window.json (+ _sizes.npz).
"""

from __future__ import annotations

import time

import numpy as np

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import load_test_scenario_alerts
from thesis.grouping.group_alerts import (
    group_alerts_fixed_window,
    group_alerts_fixed_window_host,
)

FIXED_WINDOW_SIZES = [1, 2, 5, 10, 30, 60]


def main() -> None:
    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()

    rows: list[dict] = []
    size_arrays: dict[tuple, np.ndarray] = {}
    for scenario, alerts in alerts_by_scenario.items():
        alert_index = alert_index_by_scenario[scenario]
        for window_size in FIXED_WINDOW_SIZES:
            for fn, method_name in [
                (group_alerts_fixed_window, "fixed_window"),
                (group_alerts_fixed_window_host, "fixed_window_host"),
            ]:
                _start = time.perf_counter()
                records = fn(alerts, window_size=window_size)
                _elapsed = time.perf_counter() - _start
                row, sizes = evaluate(
                    method_name,
                    str(window_size),
                    records,
                    alerts,
                    alert_index,
                    inference_time_seconds=_elapsed,
                )
                row["scenario"] = scenario
                rows.append(row)
                size_arrays[(method_name, str(window_size), scenario)] = sizes

    save_grouping_results(
        "fixed_window",
        "Fixed-window grouping: linear sweep over FIXED_WINDOW_SIZES, both host-split variants.",
        rows,
        size_arrays,
    )


if __name__ == "__main__":
    main()
