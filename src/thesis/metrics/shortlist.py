"""Shortlist file format for handing an Experiment-1 (screening sweep) shortlist
to Experiment 2 (temporal decay) and onward.

The file is a plain CSV with (at least) the columns `feature_set`,
`mining_setting`, `granularity`, `model` -- a literal subset of the columns
already produced by `thesis.metrics.config_selection.select_top_k`, so a
notebook can write
`shortlist[["feature_set", "mining_setting", "granularity", "model"]].to_csv(path, index=False)`
and a user can hand-edit the result (trim/add rows) without needing to know
any other columns. Extra columns are ignored, not rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thesis.training.model_factory import MODEL_FACTORIES

REQUIRED_COLUMNS = ["feature_set", "mining_setting", "granularity", "model"]


@dataclass(frozen=True, slots=True)
class ShortlistedConfig:
    feature_set: str  # "baseline" | "symbolic"
    mining_setting: str | None
    granularity: float
    model: str


def load_shortlist(path: Path) -> list[ShortlistedConfig]:
    """Parse and validate a shortlist CSV, failing fast (before any
    mining/training work starts) on anything that would otherwise blow up
    mid-run."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Shortlist file {path} is missing required column(s): {missing}"
        )

    configs: list[ShortlistedConfig] = []
    seen: set[tuple] = set()
    for i, row in df.iterrows():
        feature_set = str(row["feature_set"]).strip()
        mining_setting = row["mining_setting"]
        mining_setting = (
            None
            if pd.isna(mining_setting) or str(mining_setting).strip() == ""
            else str(mining_setting).strip()
        )
        model = str(row["model"]).strip()

        if feature_set == "baseline":
            if mining_setting is not None:
                raise ValueError(
                    f"Shortlist row {i}: feature_set='baseline' must not carry a "
                    f"mining_setting (got {mining_setting!r})"
                )
        elif feature_set == "symbolic":
            if mining_setting is None:
                raise ValueError(
                    f"Shortlist row {i}: feature_set='symbolic' requires a "
                    "mining_setting"
                )
        else:
            raise ValueError(
                f"Shortlist row {i}: unrecognized feature_set {feature_set!r} "
                "(expected 'baseline' or 'symbolic')"
            )

        if model not in MODEL_FACTORIES:
            raise ValueError(
                f"Shortlist row {i}: unknown model {model!r} "
                f"(known models: {sorted(MODEL_FACTORIES)})"
            )

        cfg = ShortlistedConfig(
            feature_set=feature_set,
            mining_setting=mining_setting,
            granularity=float(row["granularity"]),
            model=model,
        )
        key = (cfg.feature_set, cfg.mining_setting, cfg.granularity, cfg.model)
        if key in seen:
            print(f"  [shortlist] duplicate row {i} ({cfg}) -- skipping")
            continue
        seen.add(key)
        configs.append(cfg)

    return configs
