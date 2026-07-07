"""
Plots for run_attribute_mining_window_sweep.py (the two-stage per-alert-group
attribute mining sweep: contrast-set filtering + decision-tree rule mining).

Unlike the older mode-based sweep (see window_sweep.py), this pipeline mines
both classes jointly in a single pass per window, so there is no
benign/mixed/smart axis here — every plot is faceted by granularity only,
with one line/bar per scenario.

Functions take the in-memory analysis DataFrames (t1..t6, as produced by
run_attribute_mining_window_sweep.py) directly, so plotting can run right
after mining without a CSV round-trip.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from thesis.visualization.mining.common import (
    ATTACK_COLOR,
    BENIGN_COLOR,
    NEUTRAL_COLOR,
    base_feature,
    grans,
    ordered_value_color_map,
    parse_summary_params,
    parse_win_start,
    savefig,
    scenario_colour_map,
    strip_axes,
)

_REASON_COLOURS = {
    "growth_rate_too_close_to_1": "#4477AA",
    "insufficient_attack_coverage": "#CC3311",
    "insufficient_benign_coverage": "#EE6677",
    "other": "#BBBBBB",
}


# ---------------------------------------------------------------------------
# Plot 1 — Feature count per window (T1)
# ---------------------------------------------------------------------------
def plot_feature_count(t1: pd.DataFrame, out: Path) -> None:
    gran_vals = grans(t1)
    scenarios = sorted(t1["scenario"].unique())
    cmap = scenario_colour_map(scenarios)

    fig, axes = plt.subplots(
        len(gran_vals), 1, figsize=(6, 3 * len(gran_vals)), squeeze=False
    )

    for ri, gran in enumerate(gran_vals):
        ax = axes[ri][0]
        sub = t1[t1["gran"] == gran]
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
        ax.set_title(f"gran={gran:.0%}", fontsize=9)
        ax.set_ylabel("# features")
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("window")

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
    savefig(fig, out, "feature_count_per_window.png")


# ---------------------------------------------------------------------------
# Plot 2 — Feature set changes per window (T1)
# ---------------------------------------------------------------------------
def plot_churn(t1: pd.DataFrame, out: Path) -> None:
    """Green bars up = features gained, red bars down = features lost.
    Jaccard overlap overlaid as a line on a secondary axis.
    Layout: one row per scenario, one figure per granularity."""
    gran_vals = grans(t1)
    scenarios = sorted(t1["scenario"].unique())

    for gran in gran_vals:
        nrows = len(scenarios)
        fig, axes = plt.subplots(nrows, 1, figsize=(6, 2.8 * nrows), squeeze=False)

        for ri, scenario in enumerate(scenarios):
            ax = axes[ri][0]
            sub = t1[(t1["gran"] == gran) & (t1["scenario"] == scenario)].sort_values(
                "window"
            )
            tr = sub[sub["n_added"].notna()]
            if tr.empty:
                ax.set_visible(False)
                continue

            ws = tr["window"].values
            added = tr["n_added"].values
            removed = tr["n_removed"].values
            jaccard = tr["jaccard"].values

            ax.bar(ws, added, color="#228833", width=0.5, alpha=0.85, label="gained")
            ax.bar(ws, -removed, color="#CC3311", width=0.5, alpha=0.85, label="lost")
            ax.axhline(0, color="black", lw=0.6)

            ax2 = ax.twinx()
            ax2.plot(
                ws, jaccard, color="#4477AA", lw=1.6, marker="o", ms=4, label="overlap"
            )
            ax2.set_ylim(0, 1.1)
            ax2.set_yticks([0, 0.5, 1.0])
            ax2.tick_params(axis="y", labelsize=7, colors="#4477AA")
            ax2.set_ylabel("overlap", fontsize=7, color="#4477AA")

            ax.set_title(scenario, fontsize=9)
            ax.set_xticks(ws)
            ax.set_ylabel("# features")
            ax.grid(axis="y", alpha=0.3)

        axes[-1][0].set_xlabel("window")

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
        gran_tag = f"{int(gran * 100):02d}"
        fig.suptitle(f"Feature set changes per window  (gran={gran:.0%})", y=1.01)
        fig.tight_layout()
        savefig(fig, out, f"feature_set_changes_gran{gran_tag}.png")


# ---------------------------------------------------------------------------
# Plot 3 — Jaccard convergence (T2)
# ---------------------------------------------------------------------------
def plot_convergence(t2: pd.DataFrame, out: Path) -> None:
    gran_vals = grans(t2)
    scenarios = sorted(t2["scenario"].unique())
    cmap = scenario_colour_map(scenarios)

    fig, axes = plt.subplots(
        len(gran_vals), 1, figsize=(6, 3 * len(gran_vals)), squeeze=False
    )

    for ri, gran in enumerate(gran_vals):
        ax = axes[ri][0]
        sub = t2[t2["gran"] == gran]
        for scenario in scenarios:
            s = sub[sub["scenario"] == scenario].sort_values("transition")
            if s.empty:
                continue
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
        ax.set_title(f"gran={gran:.0%}", fontsize=9)
        ax.set_ylabel("Jaccard")
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("window")

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
    savefig(fig, out, "jaccard_convergence.png")


# ---------------------------------------------------------------------------
# Plot 4 — Contrast-set drop diagnosis (T3)
# ---------------------------------------------------------------------------
def plot_drop_diagnosis(t3: pd.DataFrame, out: Path) -> None:
    """Stacked bar per scenario of why Step 1 dropped a categorical candidate,
    one panel per granularity."""
    gran_vals = grans(t3)
    scenarios = sorted(t3["scenario"].unique())
    reasons = [r for r in _REASON_COLOURS if r in set(t3["reason"].unique())]
    reasons += sorted(set(t3["reason"].unique()) - set(reasons))

    fig, axes = plt.subplots(
        len(gran_vals),
        1,
        figsize=(max(5, 1.2 * len(scenarios)), 3.2 * len(gran_vals)),
        squeeze=False,
    )

    for ri, gran in enumerate(gran_vals):
        ax = axes[ri][0]
        sub = t3[t3["gran"] == gran]
        counts = sub.groupby(["scenario", "reason"]).size().unstack(fill_value=0)
        counts = counts.reindex(index=scenarios, columns=reasons, fill_value=0)

        bottom = np.zeros(len(scenarios))
        xs = np.arange(len(scenarios))
        for reason in reasons:
            vals = counts[reason].values
            ax.bar(
                xs,
                vals,
                bottom=bottom,
                color=_REASON_COLOURS.get(reason, "#BBBBBB"),
                label=reason.replace("_", " "),
            )
            bottom += vals

        ax.set_xticks(xs)
        ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"gran={gran:.0%}", fontsize=9)
        ax.set_ylabel("# dropped candidates")
        ax.grid(axis="y", alpha=0.3)

    handles = [
        mpatches.Patch(
            color=_REASON_COLOURS.get(r, "#BBBBBB"), label=r.replace("_", " ")
        )
        for r in reasons
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(reasons), 3),
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
    )
    fig.suptitle("Why Step 1 dropped a categorical candidate", y=1.01)
    fig.tight_layout()
    savefig(fig, out, "contrast_drop_diagnosis.png")


# ---------------------------------------------------------------------------
# Plot 5 — Persistence histogram (T4)
# ---------------------------------------------------------------------------
def plot_persistence(t4: pd.DataFrame, out: Path) -> None:
    gran_vals = grans(t4)
    scenarios = sorted(t4["scenario"].unique())
    cmap = scenario_colour_map(scenarios)
    bar_w = 0.8 / max(len(scenarios), 1)

    fig, axes = plt.subplots(
        len(gran_vals), 1, figsize=(6, 3 * len(gran_vals)), squeeze=False
    )

    for ri, gran in enumerate(gran_vals):
        ax = axes[ri][0]
        sub = t4[t4["gran"] == gran]
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
        ax.set_title(f"gran={gran:.0%}", fontsize=9)
        ax.set_ylabel("# features")
        ax.grid(axis="y", alpha=0.3)
    axes[-1][0].set_xlabel("windows present")

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
    savefig(fig, out, "persistence_histogram.png")


# ---------------------------------------------------------------------------
# Plot 6 — Cross-granularity consistency (T5)
# ---------------------------------------------------------------------------
def plot_cross_granularity(t5: pd.DataFrame, out: Path) -> None:
    """Solid line = frac of coarse features also seen in *any* overlapping fine
    window; dashed line = frac seen in *all* of them. One panel per
    (fine_gran, coarse_gran) pair, one line per scenario."""
    if t5.empty:
        print("  [skip] cross_granularity — only one granularity was swept")
        return

    pairs = sorted(
        t5[["fine_gran", "coarse_gran"]].drop_duplicates().itertuples(index=False)
    )
    scenarios = sorted(t5["scenario"].unique())
    cmap = scenario_colour_map(scenarios)

    fig, axes = plt.subplots(len(pairs), 1, figsize=(6, 3 * len(pairs)), squeeze=False)

    for ri, (fine_gran, coarse_gran) in enumerate(pairs):
        ax = axes[ri][0]
        sub = t5[(t5["fine_gran"] == fine_gran) & (t5["coarse_gran"] == coarse_gran)]
        for scenario in scenarios:
            s = sub[sub["scenario"] == scenario].sort_values("coarse_win")
            if s.empty:
                continue
            colour = cmap[scenario]
            ax.plot(
                s["coarse_win"],
                s["frac_in_any"],
                color=colour,
                marker="o",
                ms=4,
                lw=1.6,
                label=f"{scenario} (any)",
            )
            ax.plot(
                s["coarse_win"],
                s["frac_in_all"],
                color=colour,
                marker="x",
                ms=4,
                lw=1.2,
                ls="--",
                label=f"{scenario} (all)",
            )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"fine={fine_gran:.0%}  vs  coarse={coarse_gran:.0%}", fontsize=9)
        ax.set_ylabel("frac of coarse features")
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("coarse window")

    handles = [mpatches.Patch(color=cmap[s], label=s) for s in scenarios]
    handles += [
        plt.Line2D(
            [0], [0], color="grey", marker="o", lw=1.6, label="in any fine window"
        ),
        plt.Line2D(
            [0],
            [0],
            color="grey",
            marker="x",
            lw=1.2,
            ls="--",
            label="in all fine windows",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(scenarios) + 2, 4),
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
    )
    fig.suptitle("Cross-granularity consistency", y=1.01)
    fig.tight_layout()
    savefig(fig, out, "cross_granularity_consistency.png")


# ---------------------------------------------------------------------------
# Plot 7 — Decision-tree top-rule drift (T6)
# ---------------------------------------------------------------------------
def plot_rule_drift(t6: pd.DataFrame, out: Path) -> None:
    """Line = confidence_attack of the top rule per window. A marker is drawn
    whenever the top rule's text changed from the previous window, so drift in
    the dominant discriminative rule is visible at a glance."""
    gran_vals = grans(t6)
    scenarios = sorted(t6["scenario"].unique())
    cmap = scenario_colour_map(scenarios)

    fig, axes = plt.subplots(
        len(gran_vals), 1, figsize=(6, 3 * len(gran_vals)), squeeze=False
    )

    for ri, gran in enumerate(gran_vals):
        ax = axes[ri][0]
        sub = t6[t6["gran"] == gran]
        for scenario in scenarios:
            s = sub[sub["scenario"] == scenario].sort_values("window")
            if s.empty:
                continue
            colour = cmap[scenario]
            ax.plot(
                s["window"],
                s["confidence_attack"],
                color=colour,
                lw=1.4,
                alpha=0.6,
            )
            changed = s["top_rule"] != s["top_rule"].shift()
            changed_pts = s[changed]
            ax.scatter(
                changed_pts["window"],
                changed_pts["confidence_attack"],
                color=colour,
                s=30,
                zorder=3,
                label=scenario,
            )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"gran={gran:.0%}", fontsize=9)
        ax.set_ylabel("confidence_attack\nof top rule")
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("window")

    handles = [mpatches.Patch(color=cmap[s], label=s) for s in scenarios]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            ms=7,
            label="top rule changed",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(scenarios) + 1, 4),
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
    )
    fig.suptitle("Decision-tree top-rule drift", y=1.01)
    fig.tight_layout()
    savefig(fig, out, "rule_drift.png")


# ---------------------------------------------------------------------------
# Plot 8 — Pattern lifecycle rasterplot (window_features CSVs)
# ---------------------------------------------------------------------------
_STEP_SUFFIXES = ("_step1_raw", "_step1_survivors", "_step2_survivors")


def _strip_step_suffix(stem: str) -> str:
    """window_features_{scenario}_{gran}_{win_idx}_step{1,2}_{raw,survivors}
    -> window_features_{scenario}_{gran}_{win_idx}, so the existing
    scenario/gran/win_idx parsing (by position from the end) still works
    regardless of which pipeline-step file this is."""
    for suffix in _STEP_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_window_patterns(
    run_dir: Path, scenario: str, gran: float
) -> dict[int, set[str]]:
    """Return {win_idx: set_of_patterns}, unioning the step1_survivors and
    step2_survivors per-window CSVs (the two 'made it into a rule' outputs;
    step1_raw is every candidate, filtered or not, so it's excluded here)."""
    wf_dir = run_dir / "window_features"
    result: dict[int, set[str]] = {}
    if not wf_dir.exists():
        return result
    for suffix in ("step1_survivors", "step2_survivors"):
        for p in sorted(wf_dir.glob(f"window_features_{scenario}_*_*_{suffix}.csv")):
            stem = _strip_step_suffix(p.stem)
            parts = stem.split("_")
            try:
                win_idx = int(parts[-1])
                file_gran = float(parts[-2])
            except (ValueError, IndexError):
                continue
            if abs(file_gran - gran) > 1e-6:
                continue
            df = pd.read_csv(p)
            if "pattern" not in df.columns:
                continue
            result.setdefault(win_idx, set()).update(df["pattern"].dropna().astype(str))
    return result


def _parse_pattern_combos(
    wf_dir: Path, gran_filter: list[float] | None
) -> set[tuple[str, float]]:
    """Discover (scenario, gran) from window_features filenames (using the
    step1_survivors files as the canonical set -- every mined window writes
    one, unlike step2_survivors which can be empty for a tiny window)."""
    combos: set[tuple[str, float]] = set()
    for p in wf_dir.glob("window_features_*_*_*_step1_survivors.csv"):
        stem = _strip_step_suffix(p.stem)
        parts = stem.split("_")
        try:
            int(parts[-1])  # win_idx
            gran = float(parts[-2])
        except (ValueError, IndexError):
            continue
        if gran_filter and not any(abs(gran - g) < 1e-6 for g in gran_filter):
            continue
        scenario = "_".join(
            parts[2:-2]
        )  # drop "window_features" prefix + gran/win suffix
        if not scenario:
            continue
        combos.add((scenario, gran))
    return combos


def plot_feature_lifecycle(
    run_dir: Path, out: Path, gran_filter: list[float] | None = None
) -> None:
    """One figure per (scenario, gran): a single raster panel showing which
    mined patterns (contrast-set + decision-tree) are present in each window."""
    wf_dir = run_dir / "window_features"
    if not wf_dir.exists():
        print("  [skip] feature_lifecycle — window_features/ directory not found")
        return

    combos = _parse_pattern_combos(wf_dir, gran_filter)
    if not combos:
        print("  [skip] feature_lifecycle — no window_features files found")
        return

    for scenario, gran in sorted(combos):
        wf = _load_window_patterns(run_dir, scenario, gran)
        if not wf:
            continue
        n_wins = max(wf.keys()) + 1

        patterns: set[str] = set()
        for fs in wf.values():
            patterns |= fs

        def _sk(pat: str) -> tuple[int, int]:
            first = next((w for w in range(n_wins) if pat in wf.get(w, set())), n_wins)
            total = sum(1 for w in range(n_wins) if pat in wf.get(w, set()))
            return (first, -total)

        sorted_patterns = sorted(patterns, key=_sk)
        n_pats = len(sorted_patterns)
        pat_idx = {p: i for i, p in enumerate(sorted_patterns)}

        presence = np.zeros((n_pats, n_wins))
        xs_new, ys_new, xs_drop, ys_drop = [], [], [], []
        for pat in sorted_patterns:
            yi = pat_idx[pat]
            prev_present = False
            ever_dropped = False
            for wi in range(n_wins):
                cur_present = pat in wf.get(wi, set())
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

        from matplotlib.colors import ListedColormap

        cmap_presence = ListedColormap(["white", "#AADDBB", "#CCCCCC"])
        row_h = max(0.10, min(0.20, 20 / max(n_pats, 1)))
        fig_h = max(4, min(n_pats * row_h + 2, 50))
        fig, ax = plt.subplots(figsize=(max(5, n_wins * 0.9 + 1), fig_h))

        x_edges = np.arange(0, n_wins + 1)
        y_edges = np.arange(-0.5, n_pats + 0.5)
        ax.pcolormesh(
            x_edges, y_edges, presence, cmap=cmap_presence, vmin=0, vmax=2, zorder=1
        )

        dot_size = max(6, min(25, 1000 / max(n_pats, 1)))
        if xs_new:
            ax.scatter(
                xs_new, ys_new, s=dot_size, color="#228833", zorder=4, linewidths=0
            )
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

        ax.vlines(
            np.arange(0, n_wins + 1),
            -0.5,
            n_pats - 0.5,
            color="grey",
            lw=0.4,
            alpha=0.4,
            zorder=3,
        )
        ax.hlines(
            np.arange(-0.5, n_pats + 0.5),
            0,
            n_wins,
            color="grey",
            lw=0.4,
            alpha=0.4,
            zorder=3,
        )

        margin = 1.2
        ax.set_xlim(-margin, n_wins)
        ax.set_ylim(-0.5, n_pats - 0.5)
        ax.set_xticks(range(n_wins))
        ax.set_xlabel("window")

        fs = max(4, min(8, 220 / max(n_pats, 1)))
        ax.set_yticks(range(n_pats))
        ax.set_yticklabels(
            sorted_patterns, fontsize=fs, rotation=45, ha="right", va="top"
        )

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
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.04),
            fontsize=9,
        )
        fig.suptitle(f"{scenario}  —  pattern lifecycle  |  gran={gran:.0%}", y=1.02)
        fig.tight_layout()
        savefig(fig, out, f"feature_lifecycle_{scenario}_gran{gran:.0%}.png")


# ---------------------------------------------------------------------------
# Thesis report — stability / discriminativeness / feature-content figures for
# one run_attribute_mining_window_sweep.py output directory (single scenario).
#
# These are a separate, more polished figure set from plot_* above (which are
# multi-scenario, generated automatically by the sweep CLI itself): every
# report_* function below assumes a single scenario per run_dir and produces
# 300dpi figures meant to go straight into the thesis. generate_thesis_report
# is the single entry point notebooks should call.
# ---------------------------------------------------------------------------

REQUIRED_REPORT_FILES = [
    "table1_stability.csv",
    "table3_contrast_drop_diagnosis.csv",
    "table4_persistence.csv",
    "table5_cross_granularity.csv",
    "table6_rule_drift.csv",
    "mined_features_overview_step1_raw.csv",
    "mined_features_overview_step1_survivors.csv",
    "mined_features_overview_step2_survivors.csv",
]


def run_dir_is_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in REQUIRED_REPORT_FILES)


def discover_run_dirs(base_dir: Path) -> list[Path]:
    """All complete run_attribute_mining_window_sweep.py output dirs directly under
    base_dir, sorted by name (i.e. chronologically, since dirs are timestamp-prefixed).
    Incomplete dirs (crashed/partial runs) are silently skipped."""
    if not base_dir.exists():
        return []
    return sorted(
        p for p in base_dir.iterdir() if p.is_dir() and run_dir_is_complete(p)
    )


def report_stability_jaccard(
    t1: pd.DataFrame,
    gran_vals: list[float],
    gran_colors: dict[float, str],
    out: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in gran_vals:
        sub = t1[(t1["gran"] == g) & t1["jaccard"].notna()].sort_values("win_start")
        ax.plot(
            sub["win_start"] * 100,
            sub["jaccard"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=gran_colors[g],
            label=f"gran={g:.0%}",
        )
    ax.set_xlabel("position in timeline (%)")
    ax.set_ylabel("Jaccard similarity (vs. previous window)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Survivor-set stability across consecutive windows")
    ax.legend(title="Granularity", frameon=False)
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "stability_jaccard_trajectory.png", dpi=dpi)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """table5_cross_granularity.csv is written blank (no header, sometimes just a
    trailing newline) by single-granularity sweep runs, since there's nothing to
    compute cross-granularity agreement against -- pd.read_csv errors on that rather
    than returning an empty frame, so guard it explicitly."""
    if not path.read_text().strip():
        return pd.DataFrame()
    return pd.read_csv(path)


def report_cross_granularity(t5: pd.DataFrame, out: Path, dpi: int) -> None:
    if t5.empty:
        print("  [skip] stability_cross_granularity — only one granularity was swept")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(t5))
    labels = [
        f"g={r.coarse_gran:.0%} w{r.coarse_win}\n({r.coarse_range})"
        for r in t5.itertuples()
    ]
    ax.bar(
        x,
        t5["frac_in_all"],
        color=BENIGN_COLOR,
        alpha=0.85,
        width=0.6,
        label="in ALL overlapping fine windows",
    )
    ax.bar(
        x,
        t5["frac_in_any"] - t5["frac_in_all"],
        bottom=t5["frac_in_all"],
        color=NEUTRAL_COLOR,
        alpha=0.6,
        width=0.6,
        label="in ANY overlapping fine window (extra)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of coarse-window features")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Cross-granularity consistency: do coarse-window features\n"
        "also show up in the matching fine sub-windows?"
    )
    ax.legend(frameon=False, fontsize=8)
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "stability_cross_granularity.png", dpi=dpi)


def report_feature_count_trajectory(
    t1: pd.DataFrame,
    gran_vals: list[float],
    gran_colors: dict[float, str],
    out: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in gran_vals:
        sub = t1[t1["gran"] == g].sort_values("win_start")
        ax.plot(
            sub["win_start"] * 100,
            sub["n_features"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=gran_colors[g],
            label=f"gran={g:.0%}",
        )
    ax.set_xlabel("position in timeline (%)")
    ax.set_ylabel("# surviving predicates")
    ax.set_title("Survivor-set size over time")
    ax.legend(title="Granularity", frameon=False)
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "stability_feature_count.png", dpi=dpi)


def report_drop_reasons(t3: pd.DataFrame, out: Path, dpi: int) -> None:
    reason_order = [
        "growth_rate_too_close_to_1",
        "insufficient_attack_coverage",
        "insufficient_benign_coverage",
        "other",
    ]
    counts = t3["reason"].value_counts().reindex(reason_order).dropna()
    colors = [NEUTRAL_COLOR, ATTACK_COLOR, BENIGN_COLOR, "#DDCC77"][: len(counts)]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, color=colors, alpha=0.85, width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(
            i,
            v + counts.values.max() * 0.01,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("# Step 1 candidates dropped (summed over all windows)")
    ax.set_title("Why Step 1 candidates fail the contrast-set filter")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "discriminativeness_drop_reasons.png", dpi=dpi)


def report_rule_drift(
    t6: pd.DataFrame,
    gran_vals: list[float],
    gran_colors: dict[float, str],
    has_eval_col: bool,
    out: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        2 if has_eval_col else 1, 1, figsize=(9, 6.5 if has_eval_col else 4)
    )
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for g in gran_vals:
        sub = t6[(t6["gran"] == g) & t6["confidence_attack"].notna()].sort_values(
            "win_start"
        )
        ax.plot(
            sub["win_start"] * 100,
            sub["confidence_attack"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=gran_colors[g],
            label=f"gran={g:.0%}",
        )
    ax.set_ylabel(
        "confidence_attack\n(top rule" + (", holdout)" if has_eval_col else ")")
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Top decision-tree rule per window: attack recall"
        + (" on chronological holdout" if has_eval_col else "")
    )
    ax.legend(title="Granularity", frameon=False, fontsize=8)
    ax.set_xlabel("position in timeline (%)")
    strip_axes(ax)

    if has_eval_col:
        ax2 = axes[1]
        for g in gran_vals:
            sub = t6[(t6["gran"] == g) & t6["n_attack_eval"].notna()].sort_values(
                "win_start"
            )
            ax2.bar(
                sub["win_start"] * 100,
                sub["n_attack_eval"],
                width=1.5,
                color=gran_colors[g],
                alpha=0.7,
            )
        ax2.set_ylabel("# attack alert_groups\nin holdout eval")
        ax2.set_xlabel("position in timeline (%)")
        ax2.set_title(
            "Holdout sample size backing each confidence_attack estimate above"
        )
        strip_axes(ax2)

    fig.tight_layout()
    savefig(fig, out, "discriminativeness_rule_drift.png", dpi=dpi)


def _leaf_precision_table(step2_survivors: pd.DataFrame) -> pd.DataFrame:
    """confidence_attack/confidence_benign are recall, not precision -- with a large
    class imbalance a leaf can have confidence_attack > confidence_benign (labelled
    "attack-leaning") while still being majority benign in absolute composition. This
    recomputes true precision P(class | leaf) from confidence_attack and
    n_attack_eval."""
    leaves = step2_survivors.copy()
    leaves["n_attack_leaf"] = (
        (leaves["confidence_attack"] * leaves["n_attack_eval"]).round().astype(int)
    )
    leaves["precision_attack"] = leaves["n_attack_leaf"] / leaves["support_count"]
    leaves["precision_benign"] = 1 - leaves["precision_attack"]
    leaves["leaning"] = np.where(
        leaves["confidence_attack"] > leaves["confidence_benign"],
        "attack-leaning",
        "benign-leaning",
    )
    leaves["true_precision"] = np.where(
        leaves["leaning"] == "attack-leaning",
        leaves["precision_attack"],
        leaves["precision_benign"],
    )
    return leaves


def report_precision_vs_leaning(leaves: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [
        leaves.loc[leaves["leaning"] == "attack-leaning", "true_precision"],
        leaves.loc[leaves["leaning"] == "benign-leaning", "true_precision"],
    ]
    bp = ax.boxplot(
        data,
        tick_labels=[
            "attack-leaning\n(by relative recall)",
            "benign-leaning\n(by relative recall)",
        ],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bp["boxes"], [ATTACK_COLOR, BENIGN_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("true precision  P(class | leaf)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        'Recall-based "leaning" label vs. actual precision\n'
        "(a leaf can be labelled attack-leaning while still being majority-benign)"
    )
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "discriminativeness_precision_vs_leaning.png", dpi=dpi)


def report_support_vs_precision(leaves: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in [
        ("attack-leaning", ATTACK_COLOR),
        ("benign-leaning", BENIGN_COLOR),
    ]:
        sub = leaves[leaves["leaning"] == label]
        ax.scatter(
            sub["support"],
            sub["true_precision"],
            s=28,
            alpha=0.7,
            color=color,
            label=label,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("support (fraction of window, log scale)")
    ax.set_ylabel("true precision")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Compactness vs. quality: small, high-precision leaves are the\n"
        "most valuable workload-reduction candidates"
    )
    ax.legend(frameon=False)
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "discriminativeness_support_vs_precision.png", dpi=dpi)


def report_feature_reuse(
    step2_survivors: pd.DataFrame, out: Path, dpi: int
) -> pd.Series:
    """Counts how many decision-tree leaves use each underlying attribute (stripping
    thresholds/NOT_), across the whole sweep. Returns the full reuse Series (used by
    the caller to report the distinct-feature count)."""
    feature_counts: dict[str, int] = {}
    for pat in step2_survivors["pattern"]:
        for clause in pat.split(" AND "):
            feat = base_feature(clause)
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    reuse = pd.Series(feature_counts).sort_values(ascending=False)
    top = reuse.head(15)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=BENIGN_COLOR, alpha=0.85)
    for i, v in enumerate(top.values):
        ax.text(v + top.values.max() * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_xlabel(
        f"# decision-tree leaves using this feature (of {len(step2_survivors)} total)"
    )
    ax.set_title(
        f"Feature reuse across all mined rules\n({len(reuse)} distinct features used in total)"
    )
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    savefig(fig, out, "feature_content_reuse.png", dpi=dpi)
    return reuse


def report_rule_complexity(step2_survivors: pd.DataFrame, out: Path, dpi: int) -> None:
    clause_lengths = step2_survivors["pattern"].apply(lambda p: len(p.split(" AND ")))
    counts = clause_lengths.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=NEUTRAL_COLOR,
        alpha=0.9,
        width=0.6,
    )
    for i, v in enumerate(counts.values):
        ax.text(
            i,
            v + counts.values.max() * 0.01,
            str(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("# AND-clauses in the rule")
    ax.set_ylabel("# leaves")
    ax.set_title("Decision-tree rule complexity")
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "feature_content_rule_complexity.png", dpi=dpi)


def report_benign_attack_ratio(
    step1_survivors: pd.DataFrame,
    step2_survivors: pd.DataFrame,
    min_growth_rate: float,
    out: Path,
    dpi: int,
) -> None:
    inv = 1.0 / min_growth_rate
    s1_attack = int((step1_survivors["growth_rate"] >= min_growth_rate).sum())
    s1_benign = int(
        (
            (step1_survivors["confidence_benign"] > 0)
            & (step1_survivors["growth_rate"] <= inv)
        ).sum()
    )
    s2_attack = int(
        (
            step2_survivors["confidence_attack"] > step2_survivors["confidence_benign"]
        ).sum()
    )
    s2_benign = int(
        (
            step2_survivors["confidence_benign"] > step2_survivors["confidence_attack"]
        ).sum()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    w = 0.32
    attack_vals = [s1_attack, s2_attack]
    benign_vals = [s1_benign, s2_benign]
    ax.bar(
        x - w / 2,
        attack_vals,
        width=w,
        color=ATTACK_COLOR,
        alpha=0.85,
        label="attack-leaning",
    )
    ax.bar(
        x + w / 2,
        benign_vals,
        width=w,
        color=BENIGN_COLOR,
        alpha=0.85,
        label="benign-leaning",
    )
    y_max = max(attack_vals + benign_vals)
    for i, (a, b) in enumerate(zip(attack_vals, benign_vals)):
        ax.text(i - w / 2, a + y_max * 0.01, str(a), ha="center", fontsize=9)
        ax.text(i + w / 2, b + y_max * 0.01, str(b), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Step 1\ncontrast-set predicates\n(n={len(step1_survivors)})",
            f"Step 2\ndecision-tree leaves\n(n={len(step2_survivors)})",
        ]
    )
    ax.set_ylabel("# mined features")
    ax.set_title("Attack- vs. benign-leaning features, by pipeline step")
    ax.legend(frameon=False)
    strip_axes(ax)
    fig.tight_layout()
    savefig(fig, out, "feature_content_benign_attack_ratio.png", dpi=dpi)


def report_persistence(
    t4: pd.DataFrame,
    gran_vals: list[float],
    gran_colors: dict[float, str],
    out: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1, len(gran_vals), figsize=(4.2 * len(gran_vals), 4), sharey=False
    )
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, gran_vals):
        sub = t4[t4["gran"] == g].sort_values("n_windows_present")
        ax.bar(
            sub["n_windows_present"].astype(str),
            sub["n_features_with_this_persistence"],
            color=gran_colors[g],
            alpha=0.85,
            width=0.6,
        )
        ax.set_title(f"gran={g:.0%}")
        ax.set_xlabel("# windows the feature persists in")
        strip_axes(ax)
    axes[0].set_ylabel("# features")
    fig.suptitle(
        "Feature persistence: how many windows does each surviving feature keep showing up in?"
    )
    fig.tight_layout()
    savefig(fig, out, "feature_content_persistence.png", dpi=dpi)


def generate_thesis_report(run_dir: Path, dpi: int = 300) -> dict:
    """Load one run_attribute_mining_window_sweep.py output directory (single
    scenario), generate the full stability / discriminativeness / feature-content
    figure set under run_dir/thesis_figures/, and return a one-row summary dict for
    comparing this mining-parameter configuration against others."""
    figures_dir = run_dir / "thesis_figures"
    figures_dir.mkdir(exist_ok=True)

    t1 = pd.read_csv(run_dir / "table1_stability.csv")
    t3 = pd.read_csv(run_dir / "table3_contrast_drop_diagnosis.csv")
    t4 = pd.read_csv(run_dir / "table4_persistence.csv")
    t5 = _read_csv_or_empty(run_dir / "table5_cross_granularity.csv")
    t6 = pd.read_csv(run_dir / "table6_rule_drift.csv")
    step1_raw = pd.read_csv(run_dir / "mined_features_overview_step1_raw.csv")
    step1_survivors = pd.read_csv(
        run_dir / "mined_features_overview_step1_survivors.csv"
    )
    step2_survivors = pd.read_csv(
        run_dir / "mined_features_overview_step2_survivors.csv"
    )

    t1 = t1.assign(win_start=t1["win_range"].apply(parse_win_start))
    t6 = t6.assign(win_start=t6["win_range"].apply(parse_win_start))

    gran_vals = grans(t1)
    gran_colors = ordered_value_color_map(gran_vals)
    has_eval_col = "n_attack_eval" in step2_survivors.columns

    summary_path = run_dir / "summary.txt"
    params = (
        parse_summary_params(summary_path.read_text()) if summary_path.exists() else {}
    )
    min_growth_rate = float(params.get("min_growth_rate", 3.0))

    print(
        f"[{run_dir.name}] {len(gran_vals)} granularities, {len(t1)} windows, has_eval_col={has_eval_col}"
    )

    report_stability_jaccard(t1, gran_vals, gran_colors, figures_dir, dpi)
    report_cross_granularity(t5, figures_dir, dpi)
    report_feature_count_trajectory(t1, gran_vals, gran_colors, figures_dir, dpi)
    report_drop_reasons(t3, figures_dir, dpi)
    report_rule_drift(t6, gran_vals, gran_colors, has_eval_col, figures_dir, dpi)

    if has_eval_col:
        leaves = _leaf_precision_table(step2_survivors)
        report_precision_vs_leaning(leaves, figures_dir, dpi)
        report_support_vs_precision(leaves, figures_dir, dpi)

    reuse = report_feature_reuse(step2_survivors, figures_dir, dpi)
    report_rule_complexity(step2_survivors, figures_dir, dpi)
    report_benign_attack_ratio(
        step1_survivors, step2_survivors, min_growth_rate, figures_dir, dpi
    )
    report_persistence(t4, gran_vals, gran_colors, figures_dir, dpi)

    return {
        "run_dir": run_dir.name,
        "min_growth_rate": min_growth_rate,
        "max_depth": int(params.get("max_depth", 4)),
        "granularities": ",".join(f"{g:.0%}" for g in gran_vals),
        "n_windows": len(t1),
        "mean_jaccard": round(float(t1["jaccard"].mean()), 4),
        "min_jaccard": round(float(t1["jaccard"].min()), 4),
        "step1_raw": len(step1_raw),
        "step1_survivors": len(step1_survivors),
        "step1_survival_rate": round(len(step1_survivors) / len(step1_raw), 4)
        if len(step1_raw)
        else None,
        "step2_leaves": len(step2_survivors),
        "n_distinct_base_features": len(reuse),
        "mean_top_rule_confidence_attack": round(
            float(t6["confidence_attack"].mean()), 4
        ),
        "min_top_rule_confidence_attack": round(
            float(t6["confidence_attack"].min()), 4
        ),
        "has_holdout_eval": has_eval_col,
    }
