"""Shared plotting helpers for mining-sweep visualisations."""

from __future__ import annotations

import re
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
ATTACK_COLOR = "#CC3311"  # warm -> attack-leaning / attack class
BENIGN_COLOR = "#4477AA"  # cool -> benign-leaning / benign class
NEUTRAL_COLOR = "#BBBBBB"


def scenario_colour_map(scenarios: list[str]) -> dict[str, str]:
    return {
        s: SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        for i, s in enumerate(sorted(scenarios))
    }


def ordered_value_color_map(values: list) -> dict:
    """Fixed colours assigned once by sorted value (e.g. granularity, growth_rate,
    max_depth) and reused across every figure, so the same value always gets the same
    colour everywhere it appears."""
    return {
        v: SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        for i, v in enumerate(sorted(values))
    }


def grans(df: pd.DataFrame) -> list[float]:
    return sorted(df["gran"].unique())


def strip_axes(ax) -> None:
    """Light y-grid, no top/right spines -- the default axis style for thesis figures."""
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def parse_win_start(win_range: str) -> float:
    """'30%–40%' -> 0.30 (also tolerates a plain '-' separator).

    Used as a shared, granularity-independent x-axis: a window's raw index means
    different things at different granularities (window 3 covers 30-40% at gran=0.1 but
    doesn't exist at gran=0.33), so overlaying multiple granularities on the same chart
    requires aligning on actual position in the timeline instead.
    """
    first = re.split(r"[–-]", str(win_range))[0]
    return float(first.strip().rstrip("%")) / 100.0


def base_feature(token: str) -> str:
    """Strip NOT_/threshold suffix to get the underlying attribute name, e.g.
    'NOT_multi_target' -> 'multi_target',
    'signature_matches_per_day_le_143.7' -> 'signature_matches_per_day'.
    """
    t = token.strip()
    if t.startswith("NOT_"):
        t = t[4:]
    m = re.match(r"^(.*)_(le|gt)_[-0-9.eE]+$", t)
    return m.group(1) if m else t


def parse_summary_params(text: str) -> dict[str, str]:
    """Parse the `key=value  key2=value2` lines written to a sweep run's summary.txt
    into a dict of raw string values."""
    return dict(re.findall(r"(\w+)=(\S+)", text))


def savefig(fig: plt.Figure, out: Path, name: str, dpi: int = 150) -> None:
    path = out / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
