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
