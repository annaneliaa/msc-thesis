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

from thesis.visualization.mining.common import grans, savefig, scenario_colour_map

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
