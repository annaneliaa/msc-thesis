from __future__ import annotations

import json
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent


def load_scenarios(dataset: str) -> list[str]:
    with (_CONFIGS_DIR / "scenarios.json").open() as f:
        mapping = json.load(f)
    if dataset not in mapping:
        valid = list(mapping.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}")
    return mapping[dataset]


def all_datasets() -> list[str]:
    with (_CONFIGS_DIR / "scenarios.json").open() as f:
        return list(json.load(f).keys())


def dataset_for_scenario(scenario: str) -> str | None:
    """Reverse lookup: which dataset (e.g. 'ait-ads', 'cscas') a scenario belongs to.

    Returns None if the scenario isn't listed under any dataset in scenarios.json.
    """
    with (_CONFIGS_DIR / "scenarios.json").open() as f:
        mapping = json.load(f)
    for dataset, scenarios in mapping.items():
        if scenario in scenarios:
            return dataset
    return None


def load_base_features(dataset: str) -> list[str]:
    """Load the base feature list for a dataset (e.g. 'ait-ads', 'cscas')."""
    with (_CONFIGS_DIR / "baseline_features.json").open() as f:
        mapping = json.load(f)
    if dataset not in mapping:
        raise KeyError(
            f"No baseline features defined for dataset '{dataset}'. "
            f"Available: {list(mapping)}"
        )
    return mapping[dataset]


def load_dynamic_features() -> list[str]:
    """Load the dataset-agnostic dynamic feature list."""
    with (_CONFIGS_DIR / "dynamic_features.json").open() as f:
        return json.load(f)
