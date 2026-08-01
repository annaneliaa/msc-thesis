"""
CSCAS-style grouping baseline: single fixed configuration -- the authors'
own validated production settings (SessionLength=300s, SessionTimeout=60s),
not swept. See cscas_grouping_sensitivity.py for the separate robustness
check on those two parameters. Writes results/cscas_grouping.json
(+ _sizes.npz).

Named cscas_grouping.py, not cscas.py, to avoid clashing with the unrelated
classification baseline src/thesis/baselines/cscas.py.
"""

from __future__ import annotations

import time

import numpy as np

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import load_test_scenario_alerts
from thesis.grouping.group_alerts import (
    CSCAS_SESSION_LENGTH_SECONDS,
    CSCAS_SESSION_TIMEOUT_SECONDS,
    group_alerts_cscas,
)


def main() -> None:
    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()

    rows: list[dict] = []
    size_arrays: dict[tuple, np.ndarray] = {}
    param_label = f"len={CSCAS_SESSION_LENGTH_SECONDS:.0f}_timeout={CSCAS_SESSION_TIMEOUT_SECONDS:.0f}"
    for scenario, alerts in alerts_by_scenario.items():
        alert_index = alert_index_by_scenario[scenario]
        _start = time.perf_counter()
        records = group_alerts_cscas(
            alerts,
            session_length=CSCAS_SESSION_LENGTH_SECONDS,
            session_timeout=CSCAS_SESSION_TIMEOUT_SECONDS,
        )
        _elapsed = time.perf_counter() - _start
        row, sizes = evaluate(
            "cscas_grouping",
            param_label,
            records,
            alerts,
            alert_index,
            inference_time_seconds=_elapsed,
        )
        row["scenario"] = scenario
        rows.append(row)
        size_arrays[("cscas_grouping", param_label, scenario)] = sizes

    save_grouping_results(
        "cscas_grouping",
        "CSCAS-style grouping: single fixed configuration (authors' validated production settings).",
        rows,
        size_arrays,
    )


if __name__ == "__main__":
    main()
