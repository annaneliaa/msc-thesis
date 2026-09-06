"""Shared helpers for scripts/mining CLIs (run_screening_sweep.py,
run_temporal_decay.py, run_rolling_walk_forward.py, run_monitor_drift.py,
...)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.config import load_mining_settings
from thesis.configs import dataset_for_scenario
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.paths import CACHE_DIR


def cache_dir_for(
    scenario: str, filtered: bool, method: str | None, window_size: int
) -> Path:
    if dataset_for_scenario(scenario) == "cscas":
        method_tag = CSCAS_PREGROUPED_METHOD
    elif filtered:
        window_tag = f"_w{window_size}" if window_size != 2 else ""
        method_tag = (
            f"filtered_{method}{window_tag}" if method else f"filtered{window_tag}"
        )
    elif window_size != 2:
        method_tag = f"w{window_size}"
    else:
        method_tag = "fixed_window"
    cache_dir = CACHE_DIR / scenario / "groups" / method_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_shortlist_from_mining_grid(
    mining_settings_path: Path,
    granularities: list[float],
    models: list[str],
    include_baseline: bool,
) -> pd.DataFrame:
    """Cross every named entry in the mining-settings YAML
    (configs/screening_mining_settings.yaml -- the curated downstream
    parameter grid) with `granularities` x `models` to build a
    shortlist.csv-shaped DataFrame directly. This is the single input to the
    system-eval experiments (temporal decay, rolling walk-forward, drift
    monitor): no intermediate feasible-config CSV, no notebook export step,
    no real-evaluation ranking. Each row's `mining_setting` is the entry's
    own `name`, resolved back to its full two-tree AttributeMiningConfig at
    run time by experiments._shared.load_scenario_context (which loads the
    same YAML) -- so the name here is guaranteed to resolve, unlike the old
    structural-CSV path where the CSV and the YAML could drift apart. Shared
    by all three runners so their grid expansion stays identical.

    To change what gets evaluated, edit the YAML -- add/remove entries, or
    point --mining-settings at a different file."""
    names = [s.name for s in load_mining_settings(mining_settings_path)]
    rows = []
    for name in names:
        for gran in granularities:
            for model in models:
                rows.append(
                    {
                        "feature_set": "symbolic",
                        "mining_setting": name,
                        "granularity": gran,
                        "model": model,
                    }
                )
    if include_baseline:
        for gran in granularities:
            for model in models:
                rows.append(
                    {
                        "feature_set": "baseline",
                        "mining_setting": None,
                        "granularity": gran,
                        "model": model,
                    }
                )
    return pd.DataFrame(rows)
