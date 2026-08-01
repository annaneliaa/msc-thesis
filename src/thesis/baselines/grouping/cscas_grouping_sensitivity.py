"""
CSCAS sensitivity sweep (secondary): a robustness check, not a search for a
better operating point. cscas_grouping.py's SessionLength=300s/
SessionTimeout=60s are the authors' own validated production settings,
tuned for Suricata's alert density specifically -- CSCAS-style grouping has
since been generalized here to all three IDS sources, so it's a fair
question whether 60s/300s still makes sense once the input population has
changed.

Varies each parameter one at a time rather than a full factorial --
session_timeout in {30, 60, 120} with session_length held at its primary
value (300), and session_length in {150, 300, 600} with session_timeout
held at its primary value (60). The two grids share the 60/300 primary
point, so this is 5 real group_alerts_cscas/evaluate calls per scenario,
not 3x3=9 -- a full factorial would answer "what's the best CSCAS-shaped
grouping we can find," which is deliberately not the question here, and
would require interpreting interaction effects nobody asked about. Each
row is tagged `varied` ("session_timeout"/"session_length") and
`varied_value` (the swept value), so the two grids can be filtered and
plotted independently.

Deliberately its own script/artifact (results/cscas_grouping_sensitivity.json,
no _sizes.npz -- not needed by any plot here), recomputing the 60/300
primary point independently rather than reading cscas_grouping.py's
artifact, to avoid a run-order dependency between the two scripts (cheap,
no training either way). This isolation from the primary result is now
structural -- separate process, separate file, separate artifact -- not
just a DataFrame-naming convention: mixing these rows into
cscas_grouping.json would let a sensitivity setting compete with the
primary config for CSCAS's "best-reduction setting" slot in the combined
cross-method comparison, which would misrepresent CSCAS's operating point
since the primary config is a validated default, not something to
cherry-pick against.
"""

from __future__ import annotations

from thesis.baselines.grouping._metrics import evaluate
from thesis.baselines.grouping._results import save_grouping_results
from thesis.baselines.grouping._setup import load_test_scenario_alerts
from thesis.grouping.group_alerts import (
    CSCAS_SESSION_LENGTH_SECONDS,
    CSCAS_SESSION_TIMEOUT_SECONDS,
    group_alerts_cscas,
)

CSCAS_SENSITIVITY_TIMEOUTS = [30.0, 60.0, 120.0]
CSCAS_SENSITIVITY_LENGTHS = [150.0, 300.0, 600.0]


def _run_cscas_sensitivity_point(
    alerts, alert_index, scenario, session_timeout, session_length
) -> dict:
    records = group_alerts_cscas(
        alerts, session_length=session_length, session_timeout=session_timeout
    )
    row, _sizes = evaluate(
        "cscas_grouping",
        f"len={session_length:.0f}_timeout={session_timeout:.0f}",
        records,
        alerts,
        alert_index,
    )
    row["scenario"] = scenario
    row["session_timeout"] = session_timeout
    row["session_length"] = session_length
    return row


def main() -> None:
    alerts_by_scenario, alert_index_by_scenario = load_test_scenario_alerts()

    rows: list[dict] = []
    for scenario, alerts in alerts_by_scenario.items():
        alert_index = alert_index_by_scenario[scenario]

        # The authors' validated config -- computed once, then reused (not
        # recomputed) for whichever grid below reaches its own primary
        # value, since it's the shared point where both one-at-a-time
        # grids meet.
        primary_row = _run_cscas_sensitivity_point(
            alerts,
            alert_index,
            scenario,
            CSCAS_SESSION_TIMEOUT_SECONDS,
            CSCAS_SESSION_LENGTH_SECONDS,
        )

        # Grid 1: session_timeout varies, session_length held at its primary value.
        for session_timeout in CSCAS_SENSITIVITY_TIMEOUTS:
            if session_timeout == CSCAS_SESSION_TIMEOUT_SECONDS:
                row = dict(primary_row)
            else:
                row = _run_cscas_sensitivity_point(
                    alerts,
                    alert_index,
                    scenario,
                    session_timeout,
                    CSCAS_SESSION_LENGTH_SECONDS,
                )
            row["varied"] = "session_timeout"
            row["varied_value"] = session_timeout
            rows.append(row)

        # Grid 2: session_length varies, session_timeout held at its primary value.
        for session_length in CSCAS_SENSITIVITY_LENGTHS:
            if session_length == CSCAS_SESSION_LENGTH_SECONDS:
                row = dict(primary_row)
            else:
                row = _run_cscas_sensitivity_point(
                    alerts,
                    alert_index,
                    scenario,
                    CSCAS_SESSION_TIMEOUT_SECONDS,
                    session_length,
                )
            row["varied"] = "session_length"
            row["varied_value"] = session_length
            rows.append(row)

    # 5 real group_alerts_cscas/evaluate calls per scenario (1 primary + 2
    # timeout-only + 2 length-only), producing 6 tagged rows per scenario
    # (3 for each one-at-a-time grid, sharing the primary point) -- not the
    # 9 a full factorial would need.
    save_grouping_results(
        "cscas_grouping_sensitivity",
        "CSCAS sensitivity sweep (secondary): one-at-a-time robustness check on "
        "session_timeout/session_length, isolated from the primary result.",
        rows,
    )


if __name__ == "__main__":
    main()
