"""
Visualisation companion for run_mining_window_sweep.py.

Usage:
    # Explicit run directory
    python src/thesis/scripts/run_mining_window_sweep_plots.py \
        --run-dir artifacts/experiments/mining_window_sweep/<dataset>/<timestamp>/

    # Or auto-pick the most recent run for a dataset
    python src/thesis/scripts/run_mining_window_sweep_plots.py --dataset cscas
    python src/thesis/scripts/run_mining_window_sweep_plots.py --dataset ait-ads

Reads the 7 analysis CSVs produced by the sweep script and saves PNG plots
under <run_dir>/plots/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_mining_window_sweep"

# ---------------------------------------------------------------------------
# Colour conventions
# ---------------------------------------------------------------------------
_SCENARIO_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#332288",
]
_MODE_ORDER = ["benign", "mixed", "smart"]
_CHURN_COLOURS = {
    "n_unchanged": "#4477AA",
    "n_added": "#228833",
    "n_removed": "#CC3311",
}
_DELTA_COLOURS = {
    "n_benign_only": "#4477AA",
    "n_shared": "#AAAAAA",
    "n_mixed_only": "#CC3311",
}
_SAVE_KW = dict(dpi=150, bbox_inches="tight")


def _scenario_colour_map(scenarios: list[str]) -> dict[str, str]:
    return {
        s: _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        for i, s in enumerate(sorted(scenarios))
    }


def _modes_present(df: pd.DataFrame) -> list[str]:
    present = set(df["mode"].unique())
    return [m for m in _MODE_ORDER if m in present]


def _grans(df: pd.DataFrame) -> list[float]:
    return sorted(df["gran"].unique())


def _savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, **_SAVE_KW)
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 1 — Jaccard stability (T3)
# ---------------------------------------------------------------------------
def plot_jaccard_stability(df: pd.DataFrame, out: Path) -> None:
    modes = _modes_present(df)
    grans = _grans(df)
    scenarios = sorted(df["scenario"].unique())
    cmap = _scenario_colour_map(scenarios)

    nrows, ncols = len(grans), len(modes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )

    for ri, gran in enumerate(grans):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci]
            sub = df[(df["gran"] == gran) & (df["mode"] == mode)]
            for scenario in scenarios:
                s = sub[sub["scenario"] == scenario].sort_values("transition")
                if s.empty:
                    continue
                # Extract destination window index from "w0→w1" → 1
                xs = s["transition"].str.extract(r"w(\d+)$")[0].astype(int)
                ax.plot(
                    xs,
                    s["jaccard"].values,
                    marker="o",
                    ms=4,
                    color=cmap[scenario],
                    label=scenario,
                )
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
            ax.set_title(f"{mode}  |  gran={gran:.0%}", fontsize=9)
            ax.set_xlabel("window" if ri == nrows - 1 else "")
            ax.set_ylabel("Jaccard" if ci == 0 else "")
            ax.grid(alpha=0.3)

    handles = [mpatches.Patch(color=cmap[s], label=s) for s in scenarios]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(scenarios), 4),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle("Jaccard similarity between consecutive windows", y=1.01)
    fig.tight_layout()
    _savefig(fig, out, "jaccard_stability.png")


# ---------------------------------------------------------------------------
# Plot 2 — Feature count per window (T1)
# ---------------------------------------------------------------------------
def plot_feature_count(df: pd.DataFrame, out: Path) -> None:
    modes = _modes_present(df)
    grans = _grans(df)
    scenarios = sorted(df["scenario"].unique())
    cmap = _scenario_colour_map(scenarios)

    nrows, ncols = len(grans), len(modes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )

    for ri, gran in enumerate(grans):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci]
            sub = df[(df["gran"] == gran) & (df["mode"] == mode)]
            for scenario in scenarios:
                s = sub[sub["scenario"] == scenario].sort_values("window")
                if s.empty:
                    continue
                ax.plot(
                    s["window"],
                    s["n_features"],
                    marker="o",
                    ms=4,
                    color=cmap[scenario],
                    label=scenario,
                )
            ax.set_title(f"{mode}  |  gran={gran:.0%}", fontsize=9)
            ax.set_xlabel("window" if ri == nrows - 1 else "")
            ax.set_ylabel("# features" if ci == 0 else "")
            ax.grid(alpha=0.3)

    handles = [mpatches.Patch(color=cmap[s], label=s) for s in scenarios]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(scenarios), 4),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle("Feature count per window", y=1.01)
    fig.tight_layout()
    _savefig(fig, out, "feature_count_per_window.png")


# ---------------------------------------------------------------------------
# Plot 3 — Feature set changes per window (T1)
# ---------------------------------------------------------------------------
def plot_churn(df: pd.DataFrame, out: Path) -> None:
    """
    Per window transition: green bars upward = features gained,
    red bars downward = features lost.
    Overlap (Jaccard) overlaid as a line on a secondary axis.
    Layout: rows = scenarios, columns = modes.  One figure per granularity.
    """
    modes = _modes_present(df)
    grans = _grans(df)
    scenarios = sorted(df["scenario"].unique())

    for gran in grans:
        nrows, ncols = len(scenarios), len(modes)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows), squeeze=False
        )

        for ri, scenario in enumerate(scenarios):
            for ci, mode in enumerate(modes):
                ax = axes[ri][ci]
                sub = df[
                    (df["gran"] == gran)
                    & (df["mode"] == mode)
                    & (df["scenario"] == scenario)
                ].sort_values("window")
                # Only transitions (window > 0) have churn data
                tr = sub[sub["n_added"].notna()]
                if tr.empty:
                    ax.set_visible(False)
                    continue

                ws = tr["window"].values
                added = tr["n_added"].values
                removed = tr["n_removed"].values
                jaccard = tr["jaccard"].values

                ax.bar(
                    ws, added, color="#228833", width=0.5, alpha=0.85, label="gained"
                )
                ax.bar(
                    ws, -removed, color="#CC3311", width=0.5, alpha=0.85, label="lost"
                )
                ax.axhline(0, color="black", lw=0.6)

                ax2 = ax.twinx()
                ax2.plot(
                    ws,
                    jaccard,
                    color="#4477AA",
                    lw=1.6,
                    marker="o",
                    ms=4,
                    label="overlap",
                )
                ax2.set_ylim(0, 1.1)
                ax2.set_yticks([0, 0.5, 1.0])
                ax2.tick_params(axis="y", labelsize=7, colors="#4477AA")
                if ci == ncols - 1:
                    ax2.set_ylabel("overlap", fontsize=7, color="#4477AA")
                else:
                    ax2.set_yticklabels([])

                ax.set_title(f"{scenario} | {mode}", fontsize=8)
                ax.set_xticks(ws)
                ax.set_xlabel("window" if ri == nrows - 1 else "")
                ax.set_ylabel("# features" if ci == 0 else "")
                ax.grid(axis="y", alpha=0.3)

        handles = [
            mpatches.Patch(color="#228833", alpha=0.85, label="gained"),
            mpatches.Patch(color="#CC3311", alpha=0.85, label="lost"),
            plt.Line2D(
                [0], [0], color="#4477AA", lw=2, marker="o", ms=5, label="overlap"
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
        )
        fig.suptitle(f"Feature set changes per window  (gran={gran:.0%})", y=1.01)
        fig.tight_layout()
        gran_tag = f"{int(gran * 100):02d}"
        _savefig(fig, out, f"feature_set_changes_gran{gran_tag}.png")


# ---------------------------------------------------------------------------
# Plot 4 — Persistence histogram (T6)
# ---------------------------------------------------------------------------
def plot_persistence(df: pd.DataFrame, out: Path) -> None:
    modes = _modes_present(df)
    grans = _grans(df)
    scenarios = sorted(df["scenario"].unique())
    cmap = _scenario_colour_map(scenarios)

    nrows, ncols = len(grans), len(modes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )

    bar_w = 0.8 / max(len(scenarios), 1)

    for ri, gran in enumerate(grans):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci]
            sub = df[(df["gran"] == gran) & (df["mode"] == mode)]
            if sub.empty:
                ax.set_visible(False)
                continue
            max_win = int(sub["n_windows_present"].max())
            xs = np.arange(1, max_win + 1)
            for k, scenario in enumerate(scenarios):
                s = sub[sub["scenario"] == scenario]
                counts = s.set_index("n_windows_present")[
                    "n_features_with_this_persistence"
                ]
                ys = [counts.get(x, 0) for x in xs]
                offset = (k - len(scenarios) / 2 + 0.5) * bar_w
                ax.bar(
                    xs + offset,
                    ys,
                    width=bar_w,
                    color=cmap[scenario],
                    label=scenario,
                    alpha=0.85,
                )
            ax.set_xticks(xs)
            ax.set_title(f"{mode}  |  gran={gran:.0%}", fontsize=9)
            ax.set_xlabel("windows present" if ri == nrows - 1 else "")
            ax.set_ylabel("# features" if ci == 0 else "")
            ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=cmap[s], label=s) for s in scenarios]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(scenarios), 4),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle(
        "Feature persistence (how many windows each feature appears in)", y=1.01
    )
    fig.tight_layout()
    _savefig(fig, out, "persistence_histogram.png")


# ---------------------------------------------------------------------------
# Plot 5 — Attack contamination cost (T5 + T1)
# ---------------------------------------------------------------------------
def plot_attack_contamination(t5: pd.DataFrame, t1: pd.DataFrame, out: Path) -> None:
    """
    For each window: blue line = features mined on benign-only traffic,
    orange line = features surviving when attack traffic is included.
    Red shading between them = features lost to attack contamination.
    Grey bars (secondary axis) = number of attack alert_groups in that window.
    """
    grans = _grans(t5)
    scenarios = sorted(t5["scenario"].unique())

    # Use per-window attack tx counts from mixed mode (benign mode strips attack tx)
    attack_counts = (
        t1[t1["mode"] == "mixed"][["scenario", "gran", "window", "n_attack_tx"]]
        if "n_attack_tx" in t1.columns
        else None
    )

    for gran in grans:
        sub5 = t5[t5["gran"] == gran]
        sub_atk = (
            attack_counts[attack_counts["gran"] == gran]
            if attack_counts is not None
            else None
        )

        nrows = len(scenarios)
        fig, axes = plt.subplots(nrows, 1, figsize=(7, 2.8 * nrows), squeeze=False)

        for ri, scenario in enumerate(scenarios):
            ax = axes[ri][0]
            s = sub5[sub5["scenario"] == scenario].sort_values("window")

            if s.empty:
                ax.set_visible(False)
                continue

            ws = s["window"].values
            n_benign = s["n_benign_total"].values
            n_full = s["n_mixed_total"].values

            ax.plot(
                ws,
                n_benign,
                color="#4477AA",
                lw=1.8,
                marker="o",
                ms=4,
                label="benign-only traffic",
            )
            ax.plot(
                ws,
                n_full,
                color="#EE6677",
                lw=1.8,
                marker="o",
                ms=4,
                label="all traffic",
            )
            ax.fill_between(
                ws,
                n_full,
                n_benign,
                color="#CC3311",
                alpha=0.18,
                label="features lost to attack",
            )

            ax.set_xlim(-0.5, ws.max() + 0.5)
            ax.set_xticks(ws)
            ax.set_ylabel("# features")
            ax.set_title(scenario, fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            # Identify windows with attack alert_groups and mark their x-ticks
            atk = (
                sub_atk[sub_atk["scenario"] == scenario].sort_values("window")
                if sub_atk is not None
                else pd.DataFrame()
            )
            attack_wins = set()
            if not atk.empty:
                atk_idx = atk.set_index("window").reindex(ws)["n_attack_tx"].fillna(0)
                attack_wins = set(ws[atk_idx.values > 0])

            # Color tick labels red and bold, with a circle character below
            from matplotlib.transforms import blended_transform_factory

            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.set_xticklabels([str(int(w)) for w in ws])
            for tick, wi in zip(ax.xaxis.get_major_ticks(), ws):
                if wi in attack_wins:
                    tick.label1.set_color("#CC3311")
                    tick.label1.set_fontweight("bold")
            for wi in attack_wins:
                ax.text(
                    wi,
                    0,
                    "◯",
                    transform=trans,
                    ha="center",
                    va="top",
                    color="#CC3311",
                    fontsize=15,
                    clip_on=False,
                    zorder=10,
                )

            if ri == nrows - 1:
                ax.set_xlabel("window")

        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#4477AA",
                lw=2,
                marker="o",
                ms=5,
                label="benign-only traffic",
            ),
            plt.Line2D(
                [0], [0], color="#EE6677", lw=2, marker="o", ms=5, label="all traffic"
            ),
            mpatches.Patch(color="#CC3311", alpha=0.3, label="features lost to attack"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="#CC3311",
                markeredgewidth=1.5,
                ms=8,
                label="window contains attacks",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
        )
        gran_tag = f"{int(gran * 100):02d}"
        fig.suptitle(
            f"Cost of attack traffic on mined features  (gran={gran:.0%})", y=1.01
        )
        fig.tight_layout()
        _savefig(fig, out, f"attack_contamination_cost_gran{gran_tag}.png")


# ---------------------------------------------------------------------------
# Plot 6 — Cross-scenario stable-core overlap heatmap (T2 + T6)
# ---------------------------------------------------------------------------
def plot_cross_scenario_sharing(t2: pd.DataFrame, t6: pd.DataFrame, out: Path) -> None:
    """
    Pairwise Jaccard of the *stable core* per scenario — features that appear
    in every window (pct_of_total_windows == 1.0).  This answers: do scenarios
    share the same robust patterns, not just any transiently mined feature.
    """
    modes = _modes_present(t2)
    grans = _grans(t2)
    scenarios = sorted(t2["scenario"].unique())
    if len(scenarios) < 2:
        print("  [skip] cross_scenario_sharing_heatmap — only 1 scenario")
        return

    # Stable core: features whose n_windows equals the maximum possible for that group.
    # T2 stores n_windows = how many windows that feature appeared in for that scenario/mode/gran.
    max_wins = t2.groupby(["scenario", "mode", "gran"])["n_windows"].transform("max")
    stable = t2[t2["n_windows"] == max_wins][["scenario", "mode", "gran", "feature"]]

    nrows, ncols = len(grans), len(modes)
    cell = max(2.5, 0.6 * len(scenarios))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell * ncols + 1, cell * nrows), squeeze=False
    )

    for ri, gran in enumerate(grans):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci]
            sub = stable[(stable["gran"] == gran) & (stable["mode"] == mode)]
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no stable core",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="grey",
                    fontsize=8,
                )
                ax.set_title(f"{mode}  |  gran={gran:.0%}", fontsize=9)
                continue

            core = {s: set(sub[sub["scenario"] == s]["feature"]) for s in scenarios}
            n = len(scenarios)
            mat = np.zeros((n, n))
            for i, sa in enumerate(scenarios):
                for j, sb in enumerate(scenarios):
                    a, b = core[sa], core[sb]
                    union = len(a | b)
                    mat[i, j] = len(a & b) / union if union else 0.0

            im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(scenarios, fontsize=7)
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if mat[i, j] < 0.7 else "white",
                    )
            core_sizes = [len(core[s]) for s in scenarios]
            ax.set_title(
                f"{mode}  |  gran={gran:.0%}\n"
                f"core sizes: {dict(zip(scenarios, core_sizes))}",
                fontsize=7,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Pairwise Jaccard of stable feature cores across scenarios\n"
        "(stable = present in every window)",
        y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, out, "cross_scenario_stable_core_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 7 — Dropped feature diagnosis (T4)
# ---------------------------------------------------------------------------
def plot_dropped_diagnosis(df: pd.DataFrame, out: Path) -> None:
    """
    For each feature dropped by the support_diff filter, shows the ratio of
    actual attack support to expected attack support (if attack merely mirrored
    background benign traffic).  Ratio ≤ 1.5 = uniform_noise (drop may be
    unfair); ratio >> 1.5 = attack_concentrated (filter was correct).
    A healthy filter shows almost all dropped features well above the threshold.
    """
    scenarios = sorted(df["scenario"].unique())
    verdicts = sorted(df["verdict"].unique())
    verdict_colours = {"uniform_noise": "#4477AA", "attack_concentrated": "#CC3311"}

    n = len(scenarios)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )

    for idx, scenario in enumerate(scenarios):
        ri, ci = divmod(idx, ncols)
        ax = axes[ri][ci]
        sub = df[df["scenario"] == scenario]
        for verdict in verdicts:
            vals = sub[sub["verdict"] == verdict]["ratio_actual_over_expected"].dropna()
            if vals.empty:
                continue
            colour = verdict_colours.get(verdict, "#BBBBBB")
            ax.hist(
                vals, bins=40, alpha=0.7, color=colour, label=verdict.replace("_", " ")
            )
        ax.axvline(1.5, color="black", lw=1.2, ls="--", label="rescue threshold (1.5)")
        ax.set_title(scenario, fontsize=9)
        ax.set_xlabel("actual / expected attack support")
        ax.set_ylabel("# dropped features" if ci == 0 else "")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        ri, ci = divmod(idx, ncols)
        axes[ri][ci].set_visible(False)

    fig.suptitle(
        "Dropped feature diagnosis\n"
        "Is the support_diff filter dropping features because of genuine attack concentration,\n"
        "or just background noise?",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, out, "dropped_feature_diagnosis.png")


# ---------------------------------------------------------------------------
# Plot 8 — Feature lifecycle rasterplot (window_features CSVs)
# ---------------------------------------------------------------------------


def _load_window_features(
    run_dir: Path, scenario: str, mode: str, gran: float
) -> dict[int, set[str]]:
    """Return {win_idx: set_of_features} from the per-window CSV files."""
    wf_dir = run_dir / "window_features"
    # gran_tag = f"{gran:.6f}".rstrip("0").rstrip(".")
    result: dict[int, set[str]] = {}
    if not wf_dir.exists():
        return result
    for p in sorted(wf_dir.glob(f"window_features_{scenario}_{mode}_*_*.csv")):
        # filename: window_features_{scenario}_{mode}_{gran}_{win_idx}.csv
        stem = p.stem  # e.g. window_features_fox_benign_0.1_0
        parts = stem.split("_")
        # gran and win_idx are the last two parts; mode may contain underscores
        try:
            win_idx = int(parts[-1])
            file_gran = float(parts[-2])
        except (ValueError, IndexError):
            continue
        if abs(file_gran - gran) > 1e-6:
            continue
        df = pd.read_csv(p)
        if "feature" not in df.columns:
            continue
        if "source" in df.columns:
            df = df[df["source"] == "filtered"]
        result[win_idx] = set(df["feature"].dropna().astype(str))
    return result


def _parse_wf_combos(
    wf_dir: Path, gran_filter: list[float] | None
) -> set[tuple[str, str, float]]:
    """Discover (scenario, mode, gran) from window_features filenames."""
    combos: set[tuple[str, str, float]] = set()
    known_modes = {"benign", "mixed", "smart"}
    for p in wf_dir.glob("window_features_*_*_*_*.csv"):
        parts = p.stem.split("_")
        try:
            int(parts[-1])  # win_idx
            gran = float(parts[-2])
        except (ValueError, IndexError):
            continue
        if gran_filter and not any(abs(gran - g) < 1e-6 for g in gran_filter):
            continue
        inner = parts[2:-2]  # drop "window_features" prefix and gran+win_idx suffix
        mode = None
        mode_idx = None
        for i in range(len(inner) - 1, -1, -1):
            if inner[i] in known_modes:
                mode = inner[i]
                mode_idx = i
                break
        if mode is None:
            continue
        scenario = "_".join(inner[:mode_idx])
        combos.add((scenario, mode, gran))
    return combos


def _render_lifecycle_panel(
    ax: plt.Axes,
    wf: dict[int, set[str]],
    sorted_features: list[str],
    feat_idx: dict[str, int],
    n_wins: int,
    n_feats: int,
    attack_windows: set[int] | None = None,
) -> None:
    """Draw one lifecycle raster panel using pcolormesh shading + scatter markers."""
    # Build presence matrix: 0=absent, 1=first run (green), 2=re-entry (grey)
    presence = np.zeros((n_feats, n_wins))
    xs_new: list[float] = []
    ys_new: list[float] = []
    xs_drop: list[float] = []
    ys_drop: list[float] = []

    for feat in sorted_features:
        yi = feat_idx[feat]
        prev_present = False
        ever_dropped = False
        for wi in range(n_wins):
            cur_present = feat in wf.get(wi, set())
            if cur_present:
                presence[yi, wi] = 2.0 if ever_dropped else 1.0
                if not prev_present:
                    xs_new.append(wi)
                    ys_new.append(yi)
            elif prev_present:
                xs_drop.append(wi)
                ys_drop.append(yi)
                ever_dropped = True
            prev_present = cur_present

    # Cell shading: white=absent, green=first run, grey=re-entry
    from matplotlib.colors import ListedColormap

    cmap_presence = ListedColormap(["white", "#AADDBB", "#CCCCCC"])
    x_edges = np.arange(0, n_wins + 1)  # cells: [0,1), [1,2), ...
    y_edges = np.arange(-0.5, n_feats + 0.5)
    ax.pcolormesh(
        x_edges, y_edges, presence, cmap=cmap_presence, vmin=0, vmax=2, zorder=1
    )

    # Attack-window overlay
    for wi in attack_windows or set():
        ax.axvspan(wi, wi + 1, color="#EE6677", alpha=0.15, lw=0, zorder=2)

    # Transition markers on top
    dot_size = max(6, min(25, 1000 / max(n_feats, 1)))
    if xs_new:
        ax.scatter(xs_new, ys_new, s=dot_size, color="#228833", zorder=4, linewidths=0)
    if xs_drop:
        ax.scatter(
            xs_drop,
            ys_drop,
            s=dot_size * 1.2,
            color="#CC3311",
            marker="x",
            zorder=4,
            linewidths=0.8,
        )

    # Grid raster: lines at cell boundaries = integer x positions
    ax.vlines(
        np.arange(0, n_wins + 1),
        -0.5,
        n_feats - 0.5,
        color="grey",
        lw=0.4,
        alpha=0.4,
        zorder=3,
    )
    ax.hlines(
        np.arange(-0.5, n_feats + 0.5),
        0,
        n_wins,
        color="grey",
        lw=0.4,
        alpha=0.4,
        zorder=3,
    )

    # Left margin so window-0 tick sits away from the y-axis,
    # leaving room for first-appearance dots at x=0.
    margin = 1.2
    ax.set_xlim(-margin, n_wins)
    ax.set_ylim(-0.5, n_feats - 0.5)
    ax.set_xticks(range(n_wins))


def plot_feature_lifecycle(
    run_dir: Path, out: Path, gran_filter: list[float] | None = None
) -> None:
    """
    One figure per (scenario, gran): 1 row × 3 cols (modes).
    Y-axis: features present in any mode for this gran, shared across panels.
    Features sorted by first appearance in the reference mode (benign), then by
    total presence count (most persistent first).
    """
    wf_dir = run_dir / "window_features"
    if not wf_dir.exists():
        print("  [skip] feature_lifecycle — window_features/ directory not found")
        return

    combos = _parse_wf_combos(wf_dir, gran_filter)
    scenarios = sorted({s for s, _, _ in combos})
    grans = sorted({g for _, _, g in combos})
    modes = [m for m in _MODE_ORDER if any(m == md for _, md, _ in combos)]

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#228833",
            ms=7,
            label="first / re-entry",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#222222",
            ms=7,
            label="consecutive",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="#CC3311",
            ms=7,
            markeredgewidth=1.2,
            label="dropped",
        ),
    ]

    for scenario in scenarios:
        for gran in grans:
            # Load per-mode data for this (scenario, gran)
            gran_data: dict[str, dict[int, set[str]]] = {}
            for mode in modes:
                if (scenario, mode, gran) not in combos:
                    continue
                gran_data[mode] = _load_window_features(run_dir, scenario, mode, gran)

            if not gran_data:
                continue

            n_wins = max(
                (max(wf.keys()) + 1 for wf in gran_data.values() if wf),
                default=1,
            )

            # Each panel gets its own independent feature set and y-axis.
            def _panel_features(wf: dict[int, set[str]]) -> list[str]:
                feats: set[str] = set()
                for fs in wf.values():
                    feats |= fs

                def _sk(feat: str) -> tuple[int, int]:
                    first = next(
                        (w for w in range(n_wins) if feat in wf.get(w, set())), n_wins
                    )
                    total = sum(1 for w in range(n_wins) if feat in wf.get(w, set()))
                    return (first, -total)

                return sorted(feats, key=_sk)

            max_n_feats = max(
                (len(_panel_features(wf)) for wf in gran_data.values()),
                default=1,
            )

            # Figure height driven by the tallest panel.
            row_h = max(0.10, min(0.20, 20 / max(max_n_feats, 1)))
            fig_h = max(4, min(max_n_feats * row_h + 2, 50))
            ncols = len(gran_data)
            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(max(5, n_wins * 0.9 + 1) * ncols, fig_h),
                squeeze=False,
            )

            # Detect attack windows (any window containing attack alert_groups)
            # using the mixed mode data (most complete alert_group set)
            # ref_wf_any = next(iter(gran_data.values()))
            attack_wins: set[int] = set()
            for mode in ("mixed", "smart", "benign"):
                if mode in gran_data:
                    # We don't have n_attack_tx here — approximate from feature
                    # counts shifting dramatically. Instead just leave empty;
                    # caller can pass externally if needed.
                    break

            for ci, mode in enumerate(modes):
                if mode not in gran_data:
                    continue
                ax = axes[0][ci]
                wf = gran_data[mode]

                sorted_features = _panel_features(wf)
                feat_idx = {f: i for i, f in enumerate(sorted_features)}
                n_feats = len(sorted_features)

                _render_lifecycle_panel(
                    ax,
                    wf,
                    sorted_features,
                    feat_idx,
                    n_wins,
                    n_feats,
                    attack_windows=attack_wins,
                )

                ax.set_xlabel("window", fontsize=9)
                ax.set_title(f"{mode}  (n={n_feats})", fontsize=10)

                fs = max(4, min(8, 220 / max(n_feats, 1)))
                ax.set_yticks(range(n_feats))
                ax.set_yticklabels(
                    sorted_features, fontsize=fs, rotation=45, ha="right", va="top"
                )

                if not wf:
                    ax.text(
                        0.5,
                        0.5,
                        "no data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="grey",
                        fontsize=8,
                    )

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, -0.04),
                fontsize=9,
            )
            fig.suptitle(
                f"{scenario}  —  feature lifecycle  |  gran={gran:.0%}", y=1.01
            )
            fig.tight_layout()
            _savefig(fig, out, f"feature_lifecycle_{scenario}_gran{gran:.0%}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _latest_run_dir(dataset: str) -> Path:
    """Most recent timestamped run directory under mining_window_sweep/<dataset>/.

    Run directories are named %Y%m%d_%H%M%S, so lexicographic sort == chronological.
    """
    dataset_dir = _EXPERIMENTS_DIR / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"No runs found for dataset '{dataset}' at {dataset_dir}\n"
            "Run run_mining_window_sweep.py for this dataset first."
        )
    run_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {dataset_dir}")
    return run_dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mining window sweep results")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific mining_window_sweep run directory. "
        "Mutually exclusive with --dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. 'ait-ads', 'cscas') — auto-selects the most "
        "recent run under artifacts/experiments/mining_window_sweep/<dataset>/. "
        "Mutually exclusive with --run-dir.",
    )
    parser.add_argument(
        "--gran",
        type=float,
        nargs="*",
        default=None,
        help="Restrict to specific granularity values (e.g. --gran 0.1 0.2)",
    )
    args = parser.parse_args()

    if args.run_dir is None and args.dataset is None:
        parser.error("Provide --run-dir or --dataset.")
    if args.run_dir is not None and args.dataset is not None:
        parser.error("Provide only one of --run-dir or --dataset, not both.")

    if args.dataset is not None:
        run_dir = _latest_run_dir(args.dataset)
        print(f"[dataset={args.dataset}] Using latest run: {run_dir}")
    else:
        run_dir = args.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    def _load(name: str) -> pd.DataFrame | None:
        p = run_dir / name
        if not p.exists():
            print(f"  [skip] {name} not found")
            return None
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f"  [skip] {name} is empty (no rows to diagnose)")
            return None
        if args.gran is not None and "gran" in df.columns:
            df = df[df["gran"].isin(args.gran)]
        return df

    print(f"Reading CSVs from {run_dir}")
    t1 = _load("table1_stability.csv")
    t2 = _load("table2_sharing.csv")
    t3 = _load("table3_convergence.csv")
    t4 = _load("table4_dropped_diagnosis.csv")
    t5 = _load("table5_benign_vs_mixed.csv")
    t6 = _load("table6_persistence.csv")

    print(f"\nGenerating plots → {plots_dir}")

    if t3 is not None and not t3.empty:
        plot_jaccard_stability(t3, plots_dir)

    if t1 is not None and not t1.empty:
        plot_feature_count(t1, plots_dir)
        plot_churn(t1, plots_dir)

    if t6 is not None and not t6.empty:
        plot_persistence(t6, plots_dir)

    if t5 is not None and not t5.empty and t1 is not None and not t1.empty:
        plot_attack_contamination(t5, t1, plots_dir)

    if t2 is not None and not t2.empty and t6 is not None and not t6.empty:
        plot_cross_scenario_sharing(t2, t6, plots_dir)

    if t4 is not None and not t4.empty:
        plot_dropped_diagnosis(t4, plots_dir)

    plot_feature_lifecycle(run_dir, plots_dir, gran_filter=args.gran)

    print("\nDone.")


if __name__ == "__main__":
    main()
