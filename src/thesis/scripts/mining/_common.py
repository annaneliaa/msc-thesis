"""Shared helpers for scripts/mining CLIs (run_screening_sweep.py,
run_temporal_decay.py, run_rolling_walk_forward.py, run_monitor_drift.py,
...)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def build_shortlist_from_structural_configs(
    structural_configs_path: Path,
    granularities: list[float],
    models: list[str],
    include_baseline: bool,
) -> pd.DataFrame:
    """Cross attribute_mining_sweep_eda.ipynb's structural shortlist (one row
    per growth_rate/min_growth_rate_attack/max_depth/max_depth_attack combo
    that clears the mining-only precision/recall floors -- section
    5.3/5.3-addendum) with `granularities` x `models` to build a
    shortlist.csv-shaped DataFrame directly -- no separate real-evaluation
    ranking step. `name` uses the exact same convention
    configs/screening_mining_settings.yaml's own entries use
    (gr{growth_rate:g}_md{max_depth}_mda{max_depth_attack}) -- if that yaml
    was regenerated from a different structural CSV than the one passed here,
    the names can drift out of sync and every row will get skipped at lookup
    time (experiments.temporal_decay.fit_source_window warns per row, doesn't
    raise -- check its output if the shortlist here looks non-empty but every
    config gets skipped). Shared by run_temporal_decay.py,
    run_rolling_walk_forward.py, and run_monitor_drift.py so their
    --structural-configs behavior (and the shortlist it produces) stays
    identical across all three experiments."""
    structural = pd.read_csv(structural_configs_path)
    rows = []
    for _, r in structural.iterrows():
        name = f"gr{r['growth_rate']:g}_md{int(r['max_depth'])}_mda{int(r['max_depth_attack'])}"
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
