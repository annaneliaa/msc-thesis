"""
Shared alert loading for the grouping baseline scripts. Functions rather
than eager module-level globals, so a script only pays for what it actually
loads (e.g. fixed_window.py never tokenizes the DeepCASE train scenarios).
"""

from __future__ import annotations

from thesis.grouping.deepcase_grouping import train_id_for_scenarios
from thesis.grouping.window_sweep import load_and_tokenize

TEST_SCENARIOS = ["fox", "russellmitchell"]

# DeepCASE trains its own ContextBuilder on the *train*-scenario alerts (no
# pretrained checkpoint exists for this dataset, unlike AlertBERT).
DEEPCASE_TRAIN_SCENARIOS = ["shaw", "wardbeck", "wheeler", "wilson"]
# Canonical, order-independent cache key -- see train_id_for_scenarios's
# docstring for why this matters (an ad hoc string either fragments the
# cache or silently reuses a stale model).
DEEPCASE_TRAIN_ID = train_id_for_scenarios(DEEPCASE_TRAIN_SCENARIOS)


def load_scenario_alerts(scenario: str) -> list:
    """
    Loads and tokenizes raw alerts for a scenario, matching the
    GroupableAlert protocol (alert_id, ts, host, signature) plus a label
    field for purity checks. Cached to
    artifacts/cache/{scenario}/alerts/tokenized_raw.json after first run.
    """
    return load_and_tokenize(scenario)


def load_test_scenario_alerts() -> tuple[dict, dict]:
    """
    Preloads alerts for every TEST_SCENARIOS entry, plus an
    alert_id -> alert index per scenario for fast label lookups when
    computing purity. Every script's sweep loop needs this.
    """
    alerts_by_scenario = {s: load_scenario_alerts(s) for s in TEST_SCENARIOS}
    alert_index_by_scenario = {
        s: {a.alert_id: a for a in alerts} for s, alerts in alerts_by_scenario.items()
    }
    return alerts_by_scenario, alert_index_by_scenario


def load_deepcase_train_alerts() -> list:
    """
    Loads and tokenizes the 4 DeepCASE train scenarios. Only deepcase.py
    calls this -- these have no tokenized_raw.json cache yet, unlike
    fox/russellmitchell, so the first call tokenizes all 4 from raw .txt
    (real one-time cost), which every other script should not pay for.
    """
    train_alerts_by_scenario = {
        s: load_scenario_alerts(s) for s in DEEPCASE_TRAIN_SCENARIOS
    }
    return [a for s in DEEPCASE_TRAIN_SCENARIOS for a in train_alerts_by_scenario[s]]
