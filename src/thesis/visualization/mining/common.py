"""Shared plotting helpers for mining-sweep visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCENARIO_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#332288",
]
SAVE_KW = dict(dpi=150, bbox_inches="tight")


def scenario_colour_map(scenarios: list[str]) -> dict[str, str]:
    return {
        s: SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        for i, s in enumerate(sorted(scenarios))
    }


def grans(df: pd.DataFrame) -> list[float]:
    return sorted(df["gran"].unique())


def savefig(fig: plt.Figure, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  saved {path.name}")
