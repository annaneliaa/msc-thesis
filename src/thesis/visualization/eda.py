"""EDA visualizations for the alert dataset (thesis dataset introduction chapter)."""

import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from thesis.configs import load_scenarios

# Backward-compat alias for scripts that import SCENARIOS from this module.
# Always the ait-ads list; use load_scenarios() for other datasets.
SCENARIOS = load_scenarios("ait-ads")
BENIGN_LABEL = "false_positive"
ATTACK_TYPES = [
    "dirb",
    "wpscan",
    "dnsteal",
    "cracking",
    "service_scans",
    "network_scans",
    "privilege_escalation",
    "reverse_shell",
    "webshell",
    "service_stop",
]

_C_BENIGN = "#4477AA"
_C_ATTACK = "#CC3311"
_SCENARIO_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#332288",
]


# ─── helpers ─────────────────────────────────────────────────────────────────


def _ensure_dir(path: str | None) -> str | None:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _save(fig, out_path: str | None):
    if out_path:
        _ensure_dir(os.path.dirname(out_path) or ".")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _step_xy(edges: np.ndarray, values: np.ndarray):
    """Convert histogram edges + bin values into x, y for fill_between (step style)."""
    x = np.empty(2 * len(values))
    y = np.empty(2 * len(values))
    x[0::2] = edges[:-1]
    x[1::2] = edges[1:]
    y[0::2] = values
    y[1::2] = values
    return x, y


# ─── data loading ─────────────────────────────────────────────────────────────


def _ordered_scenarios(df: pd.DataFrame) -> list[str]:
    """Return scenarios present in df, preserving canonical ait-ads order where possible."""
    actual = set(df["scenario"].unique())
    ordered = [s for s in SCENARIOS if s in actual]
    return ordered if ordered else sorted(actual)


def load_alerts(
    data_dir: str,
    scenarios: list[str] | None = None,
    dataset: str = "ait-ads",
) -> pd.DataFrame:
    """
    Load alert CSVs for the given dataset.

    Returns a DataFrame with common columns:
      scenario, is_attack, timestamp (UTC pd.Timestamp), time (unix int),
      signature (alert text), ip, label
    """
    if scenarios is None:
        scenarios = load_scenarios(dataset)

    frames = []

    if dataset == "ait-ads":
        for sc in scenarios:
            path = os.path.join(data_dir, f"{sc}_alerts.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Alert CSV not found: {path}")
            df = pd.read_csv(path, dtype=str)
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["scenario"] = sc
            df["label"] = df["time_label"]
            df["signature"] = df["name"]
            df["is_attack"] = df["label"].ne(BENIGN_LABEL) & df["label"].notna()
            frames.append(df)

    elif dataset == "cscas":
        for sc in scenarios:
            path = os.path.join(data_dir, "dataset-labeled-anon-ip.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSCAS CSV not found: {path}")
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
            df["time"] = (df["timestamp"].astype("int64") // 10**9).astype(int)
            df["scenario"] = sc
            df["is_attack"] = df["Label"] == 1
            df["label"] = df["Label"].map({0: BENIGN_LABEL, 1: "attack"})
            df["signature"] = df["SignatureText"]
            df["ip"] = df["ExtIP"]
            frames.append(df)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid choices: {load_scenarios.__module__}"
        )

    return pd.concat(frames, ignore_index=True)


# ─── plots ────────────────────────────────────────────────────────────────────


def plot_alert_volume_concatenated(
    df: pd.DataFrame,
    bin_hours: float = 1.0,
    ncols: int = 4,
    figsize: tuple = (16, 6),
    out_path: str | None = None,
) -> tuple:
    """
    Alert volume over time — one subplot per scenario (2 × 4 grid).

    Each subplot shows elapsed hours from the scenario's own start on the
    x-axis, so the timelines are independent and comparable in shape without
    implying any temporal relationship between scenarios.  Benign alerts are
    stacked at the bottom (blue), attack alerts on top (red).  Y-axis is log
    scale to keep low-volume benign periods visible alongside attack bursts.

    Parameters
    ----------
    df        : DataFrame returned by load_alerts()
    bin_hours : histogram bin width in hours
    ncols     : number of subplot columns (default 4 → 2 × 4 for 8 scenarios)
    out_path  : if set, saves the figure there (PNG/PDF)
    """
    scenarios = _ordered_scenarios(df)
    nrows = (len(scenarios) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes_flat = axes.flat

    for i, sc in enumerate(scenarios):
        ax = axes_flat[i]
        sc_df = df[df["scenario"] == sc]
        t0 = sc_df["time"].min()
        elapsed = (sc_df["time"] - t0) / 3600.0
        duration = float(elapsed.max())

        bin_edges = np.arange(0, duration + bin_hours, bin_hours)
        benign_h, _ = np.histogram(elapsed[~sc_df["is_attack"]], bins=bin_edges)
        attack_h, _ = np.histogram(elapsed[sc_df["is_attack"]], bins=bin_edges)

        xe, ye_b = _step_xy(bin_edges, benign_h)
        _, ye_a = _step_xy(bin_edges, benign_h + attack_h)

        ax.fill_between(
            xe, 1, np.maximum(ye_b, 1), color=_C_BENIGN, alpha=0.75, linewidth=0
        )
        ax.fill_between(
            xe,
            np.maximum(ye_b, 1),
            np.maximum(ye_a, 1),
            color=_C_ATTACK,
            alpha=0.80,
            linewidth=0,
        )

        ax.set_yscale("log")
        ax.set_xlim(0, duration)
        ax.set_title(sc, fontsize=10, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel("Elapsed time (h)", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"Alerts / {bin_hours:.0f}h", fontsize=8)

    # hide any unused axes (if scenario count isn't a multiple of ncols)
    for j in range(len(scenarios), nrows * ncols):
        axes_flat[j].set_visible(False)

    # shared legend in the figure
    fig.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.75, label="Benign"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.80, label="Attack"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        fontsize=9,
        framealpha=0.9,
    )
    fig.suptitle("Alert volume timeline (all scenarios)", fontsize=12, y=1.01)
    plt.tight_layout()

    _save(fig, out_path)
    return fig, axes


def _get_attack_phases(
    attack_elapsed: np.ndarray, gap_hours: float
) -> list[tuple[float, float]]:
    """Group contiguous attack timestamps into (start_h, end_h) phases."""
    if len(attack_elapsed) == 0:
        return []
    t = np.sort(attack_elapsed)
    phases: list[tuple[float, float]] = []
    p_start = p_end = float(t[0])
    for ti in t[1:]:
        if float(ti) - p_end > gap_hours:
            phases.append((p_start, p_end))
            p_start = float(ti)
        p_end = float(ti)
    phases.append((p_start, p_end))
    return phases


def _draw_break_marks(ax: plt.Axes, side: str) -> None:
    """Draw two diagonal slash marks on the specified spine to signal a time break."""
    d_x, d_y = 0.022, 0.065
    x0 = 1.0 if side == "right" else 0.0
    kw = dict(
        transform=ax.transAxes, color="0.40", linewidth=1.1, clip_on=False, zorder=5
    )
    for yc in (0.18, 0.33):
        ax.plot([x0 - d_x, x0 + d_x], [yc - d_y, yc + d_y], **kw)


def plot_attack_phase_zoom(
    df: pd.DataFrame,
    context_hours: float = 0.5,  # default h before/after phase
    phase_gap_hours: float = 3.0,
    bin_hours: float = 0.01,  # default 0.01 h = 36 s for fine-grained view
    out_path: str | None = None,
) -> tuple:
    """
    Alert-volume histogram zoomed into each attack phase — one row per scenario.

    Each attack phase window is expanded by *context_hours* on both sides.
    When a scenario has multiple attack phases separated by at least
    *phase_gap_hours*, the zoomed windows are placed side-by-side with widths
    proportional to their duration; broken-axis marks (diagonal slashes) on the
    shared boundary indicate the discontinuity in elapsed time.

    Parameters
    ----------
    df               : DataFrame returned by load_alerts()
    context_hours    : hours of context shown before / after each phase
    phase_gap_hours  : minimum inter-phase gap (hours) to treat as separate
    bin_hours        : histogram bin width (default 0.25 h = 15 min)
    out_path         : optional save path
    """
    scenarios = _ordered_scenarios(df)

    # 1. Compute zoom windows per scenario
    scenario_info: list[tuple] = []
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        t0 = float(sc_df["time"].min())
        duration_h = (float(sc_df["time"].max()) - t0) / 3600.0

        elapsed_all = (sc_df["time"].values.astype(float) - t0) / 3600.0
        attack_elapsed = elapsed_all[sc_df["is_attack"].values]

        phases = _get_attack_phases(attack_elapsed, phase_gap_hours)
        windows: list[tuple[float, float]] = []
        for p_start, p_end in phases:
            ws = max(0.0, p_start - context_hours)
            we = min(duration_h, p_end + context_hours)
            if windows and ws <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], we))
            else:
                windows.append((ws, we))

        scenario_info.append((sc, sc_df, t0, elapsed_all, windows))

    # 2. Figure layout: one row per scenario, columns per phase window
    n_sc = len(scenario_info)
    fig = plt.figure(figsize=(16, n_sc * 2.0 + 1.2))
    from matplotlib.gridspec import GridSpec

    outer_gs = GridSpec(
        n_sc,
        1,
        figure=fig,
        hspace=0.75,
        top=0.93,
        bottom=0.06,
        left=0.07,
        right=0.97,
    )

    all_axes: list[list] = []

    for i, (sc, sc_df, t0, elapsed_all, windows) in enumerate(scenario_info):
        if not windows:
            ax = fig.add_subplot(outer_gs[i])
            ax.text(
                0.5,
                0.5,
                "no attack data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="0.5",
            )
            ax.set_title(sc, fontsize=9, loc="left", fontweight="bold", pad=2)
            all_axes.append([ax])
            continue

        n_panels = len(windows)
        durations = [we - ws for ws, we in windows]

        inner_gs = GridSpecFromSubplotSpec(
            1,
            n_panels,
            subplot_spec=outer_gs[i],
            width_ratios=durations,
            wspace=0.06,
        )

        row_axes: list = []
        for j, (ws, we) in enumerate(windows):
            ax = fig.add_subplot(inner_gs[0, j])
            row_axes.append(ax)

            mask = (elapsed_all >= ws) & (elapsed_all <= we)
            e_win = elapsed_all[mask]
            is_att = sc_df["is_attack"].values[mask]

            bin_edges = np.arange(ws, we + bin_hours, bin_hours)
            benign_h, _ = np.histogram(e_win[~is_att], bins=bin_edges)
            attack_h, _ = np.histogram(e_win[is_att], bins=bin_edges)

            xe, ye_b = _step_xy(bin_edges, benign_h)
            _, ye_a = _step_xy(bin_edges, benign_h + attack_h)

            ax.fill_between(
                xe, 1, np.maximum(ye_b, 1), color=_C_BENIGN, alpha=0.75, linewidth=0
            )
            ax.fill_between(
                xe,
                np.maximum(ye_b, 1),
                np.maximum(ye_a, 1),
                color=_C_ATTACK,
                alpha=0.80,
                linewidth=0,
            )

            ax.set_yscale("log")
            ax.set_xlim(ws, we)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=False))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            if j > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False, labelleft=False)
                _draw_break_marks(ax, "left")
            if j < n_panels - 1:
                ax.spines["right"].set_visible(False)
                _draw_break_marks(ax, "right")

            ax.set_xlabel("Elapsed time (h)", fontsize=7)

        row_axes[0].set_title(sc, fontsize=9, loc="left", fontweight="bold", pad=2)
        row_axes[0].set_ylabel(f"Alerts / {bin_hours:.2g}h", fontsize=7)
        all_axes.append(row_axes)

    fig.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.75, label="Benign"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.80, label="Attack"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        fontsize=9,
        framealpha=0.9,
    )
    fig.suptitle(
        f"Alert volume timeline - attack phase zoom"
        f"  (all scenarios, context={context_hours}h, phase_gap={phase_gap_hours}h, bin={bin_hours}h)",
        fontsize=11,
    )
    _save(fig, out_path)
    return fig, all_axes


def plot_class_balance(
    df: pd.DataFrame,
    figsize: tuple = (10, 4),
    out_path: str | None = None,
) -> tuple:
    """
    Per-scenario alert counts split by benign / attack.

    Left panel: absolute counts (log scale).
    Right panel: percentage breakdown (shows class imbalance clearly).
    """
    scenarios = _ordered_scenarios(df)

    rows = []
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        n_total = len(sc_df)
        n_benign = (~sc_df["is_attack"]).sum()
        n_attack = sc_df["is_attack"].sum()
        rows.append(
            dict(
                scenario=sc,
                n_total=n_total,
                n_benign=n_benign,
                n_attack=n_attack,
                pct_benign=100 * n_benign / n_total,
                pct_attack=100 * n_attack / n_total,
            )
        )
    stats = pd.DataFrame(rows)

    x = np.arange(len(scenarios))
    w = 0.35

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=figsize)

    # — absolute counts —
    ax_abs.bar(x, stats["n_benign"], w, color=_C_BENIGN, alpha=0.8, label="Benign")
    ax_abs.bar(
        x,
        stats["n_attack"],
        w,
        bottom=stats["n_benign"],
        color=_C_ATTACK,
        alpha=0.8,
        label="Attack",
    )
    ax_abs.set_yscale("log")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax_abs.set_ylabel("Alert count (log scale)", fontsize=10)
    ax_abs.set_title("Alert counts per scenario", fontsize=11)
    ax_abs.legend(fontsize=9)
    ax_abs.grid(axis="y", alpha=0.3, linewidth=0.5)

    # — percentage breakdown —
    ax_pct.bar(x, stats["pct_benign"], w, color=_C_BENIGN, alpha=0.8, label="Benign")
    ax_pct.bar(
        x,
        stats["pct_attack"],
        w,
        bottom=stats["pct_benign"],
        color=_C_ATTACK,
        alpha=0.8,
        label="Attack",
    )
    ax_pct.axhline(50, color="0.5", linestyle="--", linewidth=0.8)
    ax_pct.set_ylim(0, 100)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax_pct.set_ylabel("Percentage (%)", fontsize=10)
    ax_pct.set_title("Class balance per scenario", fontsize=11)
    ax_pct.legend(fontsize=9)
    ax_pct.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    _save(fig, out_path)
    return fig, (ax_abs, ax_pct)


def plot_attack_type_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    out_path: str | None = None,
) -> tuple:
    """
    Heatmap of attack type presence across scenarios.

    Rows = attack types, columns = scenarios.
    Cell colour encodes log10(count + 1) so rare types are visible.
    """
    scenarios = _ordered_scenarios(df)

    counts = pd.DataFrame(index=ATTACK_TYPES, columns=scenarios, dtype=float).fillna(0)
    for sc in scenarios:
        sc_df = df[(df["scenario"] == sc) & df["is_attack"]]
        vc = sc_df["label"].value_counts()
        for at in ATTACK_TYPES:
            counts.loc[at, sc] = vc.get(at, 0)

    # drop attack types absent in all scenarios
    counts = counts.loc[counts.sum(axis=1) > 0]

    if counts.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No named attack types in this dataset",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        _save(fig, out_path)
        return fig, ax

    log_counts = np.log10(counts.astype(float) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(log_counts.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels(counts.index, fontsize=9)

    # annotate raw counts in each cell
    for r, at in enumerate(counts.index):
        for c, sc in enumerate(scenarios):
            n = int(counts.loc[at, sc])
            if n > 0:
                ax.text(
                    c,
                    r,
                    f"{n:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if log_counts.loc[at, sc] < 3.5 else "white",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log₁₀(count + 1)", fontsize=9)

    ax.set_title("Attack type distribution across scenarios", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_top_alert_signatures(
    df: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple = (10, 6),
    out_path: str | None = None,
) -> tuple:
    """
    Top-K most frequent alert signatures (names), coloured by majority label.

    Useful for showing which IDS rules fire most often in the dataset.
    """
    vc = df["signature"].value_counts().head(top_k)
    names = vc.index.tolist()

    # determine majority label for each name
    colors = []
    for name in names:
        sub = df[df["signature"] == name]
        pct_attack = sub["is_attack"].mean()
        colors.append(_C_ATTACK if pct_attack >= 0.5 else _C_BENIGN)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))
    ax.barh(y, vc.values, color=colors, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Total occurrences (log scale)", fontsize=10)
    ax.set_title(f"Top {top_k} most frequent alert signatures", fontsize=12)
    ax.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.82, label="Majority benign"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.82, label="Majority attack"),
        ],
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_inter_arrival_time_cdf(
    df: pd.DataFrame,
    max_seconds: float = 3600.0,
    figsize: tuple = (8, 5),
    out_path: str | None = None,
) -> tuple:
    """
    CDF of inter-arrival times between consecutive alerts, per scenario.

    Log-scale x-axis reveals the heavy-tailed burst structure.
    The vertical dashed line marks 2 seconds — the fixed grouping window used
    in this work — to show what fraction of alert pairs are captured by it.
    """
    scenarios = _ordered_scenarios(df)

    fig, ax = plt.subplots(figsize=figsize)

    for sc, color in zip(scenarios, _SCENARIO_COLORS):
        sc_df = df[df["scenario"] == sc].sort_values("time")
        diffs = sc_df["time"].diff().dropna().astype(float)
        diffs = diffs[(diffs > 0) & (diffs <= max_seconds)]
        if diffs.empty:
            continue
        xs = np.sort(diffs.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, linewidth=1.4, color=color, label=sc)

    ax.axvline(2, color="black", linestyle="--", linewidth=1.0, label="2 s window")
    ax.set_xscale("log")
    ax.set_xlabel("Inter-arrival time (seconds)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Inter-arrival time distribution per scenario", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_group_size_distribution(
    df: pd.DataFrame,
    window_seconds: float = 2.0,
    max_size: int = 30,
    figsize: tuple = (9, 5),
    out_path: str | None = None,
) -> tuple:
    """
    Distribution of alert group sizes under the fixed time-window grouper.

    Groups alerts that fall within the same `window_seconds` bucket
    (ts // window_seconds) per scenario, then plots the count distribution.
    Gives the reader an intuition for how many alerts end up in one alert_group.
    """
    scenarios = _ordered_scenarios(df)

    all_sizes: list[int] = []
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc].copy()
        sc_df["bucket"] = (sc_df["time"] // window_seconds).astype(int)
        sizes = sc_df.groupby("bucket").size().values
        all_sizes.extend(sizes.tolist())

    sizes_arr = np.array(all_sizes)
    sizes_clipped = np.clip(sizes_arr, 1, max_size)

    bins = np.arange(0.5, max_size + 1.5, 1)
    counts, edges = np.histogram(sizes_clipped, bins=bins)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, max_size + 1)
    ax.bar(x, counts, width=0.8, color=_C_BENIGN, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(f"Group size (alerts per {window_seconds:.0f}s window)", fontsize=11)
    ax.set_ylabel("Number of groups (log scale)", fontsize=11)
    ax.set_title(
        f"Alert group size distribution ({window_seconds:.0f}s fixed window, all scenarios)",
        fontsize=12,
    )
    ax.set_xlim(0.5, max_size + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def load_alert_groups(
    alert_groups_dir: str, scenarios: list[str] | None = None, suffix: str = ""
) -> pd.DataFrame:
    """
    Load pre-computed alert_group CSVs produced by run_eda.py.

    Expects one file per scenario: <alert_groups_dir>/<scenario>_alert_groups<suffix>.csv
    with at least columns: window_start, window_end, n_alerts, group_label.

    Returns a combined DataFrame with an added 'scenario' column.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    frames = []
    for sc in scenarios:
        path = os.path.join(alert_groups_dir, f"{sc}_alert_groups{suffix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"AlertGroups CSV not found: {path}")
        df = pd.read_csv(
            path,
            usecols=["window_start", "window_end", "n_alerts", "group_label"],
        )
        df["scenario"] = sc
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def plot_alert_group_volume_concatenated(
    groups_df: pd.DataFrame,
    bin_hours: float = 1.0,
    ncols: int = 4,
    figsize: tuple = (16, 6),
    out_path: str | None = None,
) -> tuple:
    """
    AlertGroup volume over time — one subplot per scenario (2 × 4 grid).

    Mirrors plot_alert_volume_concatenated but counts 2-second alert_group
    windows rather than individual alerts. Benign alert_groups are blue,
    attack + mixed alert_groups are red. Y-axis is log scale.
    """
    scenarios = _ordered_scenarios(groups_df)
    ncols = min(ncols, len(scenarios))
    nrows = (len(scenarios) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes_flat = np.array(axes).flat

    for i, sc in enumerate(scenarios):
        ax = axes_flat[i]
        sc_tx = groups_df[groups_df["scenario"] == sc].copy()
        t0 = float(sc_tx["window_start"].min())
        elapsed = (sc_tx["window_start"].astype(float) - t0) / 3600.0
        duration = float(elapsed.max())

        is_attack = sc_tx["group_label"].isin(["attack", "mixed"]).values
        is_benign = (sc_tx["group_label"] == "benign").values

        bin_edges = np.arange(0, duration + bin_hours, bin_hours)
        benign_h, _ = np.histogram(elapsed[is_benign], bins=bin_edges)
        attack_h, _ = np.histogram(elapsed[is_attack], bins=bin_edges)

        xe, ye_b = _step_xy(bin_edges, benign_h)
        _, ye_a = _step_xy(bin_edges, benign_h + attack_h)

        ax.fill_between(
            xe, 1, np.maximum(ye_b, 1), color=_C_BENIGN, alpha=0.75, linewidth=0
        )
        ax.fill_between(
            xe,
            np.maximum(ye_b, 1),
            np.maximum(ye_a, 1),
            color=_C_ATTACK,
            alpha=0.80,
            linewidth=0,
        )

        ax.set_yscale("log")
        ax.set_xlim(0, duration)
        ax.set_title(sc, fontsize=10, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel("Elapsed time (h)", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"Tx / {bin_hours:.0f}h", fontsize=8)

    for j in range(len(scenarios), nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.75, label="Benign"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.80, label="Attack / Mixed"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        fontsize=9,
        framealpha=0.9,
    )
    fig.suptitle("AlertGroup volume timeline (all scenarios)", fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_alert_group_volume_attack_zoom(
    groups_df: pd.DataFrame,
    context_hours: float = 0.5,
    phase_gap_hours: float = 3.0,
    bin_hours: float = (1 / 60),  # 2sec bins
    out_path: str | None = None,
) -> tuple:
    """
    AlertGroup volume zoomed into each attack phase — one row per scenario.

    Attack phases are derived from alert_groups labelled 'attack' or 'mixed'.
    Mirrors plot_attack_phase_zoom but operates on alert_group windows rather
    than individual alerts.
    """
    scenarios = _ordered_scenarios(groups_df)

    scenario_info: list[tuple] = []
    for sc in scenarios:
        sc_tx = groups_df[groups_df["scenario"] == sc].copy()
        t0 = float(sc_tx["window_start"].min())
        duration_h = (float(sc_tx["window_start"].max()) - t0) / 3600.0

        elapsed_all = (sc_tx["window_start"].values.astype(float) - t0) / 3600.0
        attack_mask = sc_tx["group_label"].isin(["attack", "mixed"]).values
        attack_elapsed = elapsed_all[attack_mask]

        phases = _get_attack_phases(attack_elapsed, phase_gap_hours)
        windows: list[tuple[float, float]] = []
        for p_start, p_end in phases:
            ws = max(0.0, p_start - context_hours)
            we = min(duration_h, p_end + context_hours)
            if windows and ws <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], we))
            else:
                windows.append((ws, we))

        scenario_info.append((sc, elapsed_all, attack_mask, windows))

    n_sc = len(scenario_info)
    fig = plt.figure(figsize=(16, n_sc * 2.0 + 1.2))
    from matplotlib.gridspec import GridSpec

    outer_gs = GridSpec(
        n_sc,
        1,
        figure=fig,
        hspace=0.75,
        top=0.93,
        bottom=0.06,
        left=0.07,
        right=0.97,
    )

    all_axes: list[list] = []

    for i, (sc, elapsed_all, attack_mask, windows) in enumerate(scenario_info):
        if not windows:
            ax = fig.add_subplot(outer_gs[i])
            ax.text(
                0.5,
                0.5,
                "no attack data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="0.5",
            )
            ax.set_title(sc, fontsize=9, loc="left", fontweight="bold", pad=2)
            all_axes.append([ax])
            continue

        n_panels = len(windows)
        durations = [we - ws for ws, we in windows]
        inner_gs = GridSpecFromSubplotSpec(
            1,
            n_panels,
            subplot_spec=outer_gs[i],
            width_ratios=durations,
            wspace=0.06,
        )

        row_axes: list = []
        for j, (ws, we) in enumerate(windows):
            ax = fig.add_subplot(inner_gs[0, j])
            row_axes.append(ax)

            mask = (elapsed_all >= ws) & (elapsed_all <= we)
            e_win = elapsed_all[mask]
            is_att = attack_mask[mask]

            bin_edges = np.arange(ws, we + bin_hours, bin_hours)
            benign_h, _ = np.histogram(e_win[~is_att], bins=bin_edges)
            attack_h, _ = np.histogram(e_win[is_att], bins=bin_edges)

            xe, ye_b = _step_xy(bin_edges, benign_h)
            _, ye_a = _step_xy(bin_edges, benign_h + attack_h)

            ax.fill_between(
                xe, 1, np.maximum(ye_b, 1), color=_C_BENIGN, alpha=0.75, linewidth=0
            )
            ax.fill_between(
                xe,
                np.maximum(ye_b, 1),
                np.maximum(ye_a, 1),
                color=_C_ATTACK,
                alpha=0.80,
                linewidth=0,
            )

            ax.set_yscale("log")
            ax.set_xlim(ws, we)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=False))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            if j > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False, labelleft=False)
                _draw_break_marks(ax, "left")
            if j < n_panels - 1:
                ax.spines["right"].set_visible(False)
                _draw_break_marks(ax, "right")

            ax.set_xlabel("Elapsed time (h)", fontsize=7)

        row_axes[0].set_title(sc, fontsize=9, loc="left", fontweight="bold", pad=2)
        row_axes[0].set_ylabel(f"Tx / {bin_hours:.2g}h", fontsize=7)
        all_axes.append(row_axes)

    fig.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.75, label="Benign"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.80, label="Attack / Mixed"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        fontsize=9,
        framealpha=0.9,
    )
    fig.suptitle(
        f"AlertGroup volume timeline — attack phase zoom"
        f"  (context={context_hours}h, phase_gap={phase_gap_hours}h, bin={bin_hours}h)",
        fontsize=11,
    )
    _save(fig, out_path)
    return fig, all_axes


def plot_temporal_attack_overview(
    df: pd.DataFrame,
    bin_hours: float = 1.0,
    out_path: str | None = None,
) -> tuple:
    """
    Three-panel temporal overview for datasets with sparse attacks.

    One column per scenario, three rows (share x-axis within each column):
      Row 1 — Benign volume per bin
      Row 2 — Attack volume per bin (independent y-scale from row 1)
      Row 3 — Attack ratio per bin (0–1)
    """
    scenarios = _ordered_scenarios(df)
    n_sc = len(scenarios)

    fig, axes = plt.subplots(
        3,
        n_sc,
        figsize=(max(14, 7 * n_sc), 9),
        sharex="col",
    )
    if n_sc == 1:
        axes = np.array(axes).reshape(3, 1)

    for col, sc in enumerate(scenarios):
        sc_df = df[df["scenario"] == sc].copy()
        t0 = float(sc_df["time"].min())
        elapsed = (sc_df["time"].values.astype(float) - t0) / 3600.0
        duration = float(elapsed.max())

        is_attack = sc_df["is_attack"].values
        bin_edges = np.arange(0, duration + bin_hours, bin_hours)
        benign_h, _ = np.histogram(elapsed[~is_attack], bins=bin_edges)
        attack_h, _ = np.histogram(elapsed[is_attack], bins=bin_edges)
        total_h = benign_h + attack_h
        ratio_h = np.where(total_h > 0, attack_h / total_h.clip(1), np.nan)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax0, ax1, ax2 = axes[0, col], axes[1, col], axes[2, col]

        ax0.fill_between(centers, 0, benign_h, color=_C_BENIGN, alpha=0.75, linewidth=0)
        ax0.set_title(sc, fontsize=10, fontweight="bold", pad=4)
        ax0.set_ylabel(f"Benign / {bin_hours:.0f}h", fontsize=8)

        ax1.fill_between(centers, 0, attack_h, color=_C_ATTACK, alpha=0.80, linewidth=0)
        ax1.set_ylabel(f"Attack / {bin_hours:.0f}h", fontsize=8)

        ax2.plot(centers, ratio_h, color=_C_ATTACK, linewidth=0.8, alpha=0.9)
        ax2.set_ylim(0, None)
        ax2.set_ylabel("Attack ratio", fontsize=8)
        ax2.set_xlabel("Elapsed time (h)", fontsize=8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

        for ax in axes[:, col]:
            ax.tick_params(labelsize=7)
            ax.grid(axis="x", alpha=0.2, linewidth=0.5)
            ax.set_xlim(0, duration)

    fig.suptitle(
        f"Temporal attack overview  (bin={bin_hours:.0f}h)", fontsize=12, y=1.01
    )
    plt.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_signature_event_raster(
    df: pd.DataFrame,
    group_col: str = "signature",
    time_unit: str = "days",
    marker_size: float = 3.0,
    alpha: float = 0.5,
    figsize: tuple | None = None,
    out_path: str | None = None,
) -> tuple:
    """
    Event raster of every unique value of `group_col`'s occurrences over time.

    One row per unique value (all values shown, not just top-K); each
    occurrence is drawn as a dot at its elapsed time, coloured by whether that
    specific hit was benign or attack. Rows are grouped into a majority-attack
    block and a majority-benign block (split by which class each value fires
    in more often), each block ordered by descending occurrence count. This
    makes it possible to see, at a glance, which values are mostly benign
    noise, and whether their hits are spread at a constant rate or arrive in
    recurring bursts.

    Assumes df represents a single continuous timeline (as with CSCAS);
    elapsed time is measured from the global minimum timestamp in df.

    Parameters
    ----------
    df          : DataFrame returned by load_alerts() (or with the same
                  time/is_attack columns and a `group_col` column present)
    group_col   : column to group occurrences by, e.g. "signature" (full alert
                  text, CSCAS) or "short" (AIT-ADS's coarser descriptor code)
    time_unit   : "hours" or "days" — unit for the x-axis
    marker_size : scatter marker size (points^2)
    alpha       : marker transparency (lower helps reveal density in bursts)
    figsize     : optional override; default scales height with value count
    out_path    : optional save path
    """
    divisor = 3600.0 if time_unit == "hours" else 86400.0
    t0 = float(df["time"].min())
    elapsed = (df["time"].astype(float) - t0) / divisor

    stats = (
        df.groupby(group_col)["is_attack"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_attack", "count": "n_total"})
    )
    stats["majority_attack"] = 2 * stats["n_attack"] >= stats["n_total"]

    attack_block = stats[stats["majority_attack"]].sort_values(
        "n_total", ascending=False
    )
    benign_block = stats[~stats["majority_attack"]].sort_values(
        "n_total", ascending=False
    )
    ordered_vals = attack_block.index.tolist() + benign_block.index.tolist()
    val_to_row = {val: i for i, val in enumerate(ordered_vals)}
    n_vals = len(ordered_vals)

    if figsize is None:
        figsize = (16, max(8, 0.11 * n_vals))

    fig, ax = plt.subplots(figsize=figsize)

    y = df[group_col].map(val_to_row).values
    colors = np.where(df["is_attack"].values, _C_ATTACK, _C_BENIGN)

    ax.scatter(
        elapsed, y, s=marker_size, c=colors, alpha=alpha, linewidths=0, rasterized=True
    )

    if len(attack_block) and len(benign_block):
        ax.axhline(len(attack_block) - 0.5, color="0.3", linewidth=0.8, linestyle="--")

    ax.set_yticks(range(n_vals))
    ax.set_yticklabels(ordered_vals, fontsize=4)
    ax.set_ylim(n_vals - 0.5, -0.5)
    ax.set_xlim(0, float(elapsed.max()))
    ax.set_xlabel(f"Elapsed time ({time_unit})", fontsize=11)
    ax.set_title(
        f"{group_col.capitalize()} occurrence raster ({n_vals} {group_col} values)",
        fontsize=12,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.9, label="Benign hit"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.9, label="Attack hit"),
        ],
        loc="upper right",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_occurrence_burst_raster(
    df: pd.DataFrame,
    group_col: str = "signature",
    time_unit: str = "days",
    min_occurrences: int = 5,
    top_n: int | None = None,
    size_by_count: bool = True,
    marker_size: float = 6.0,
    alpha: float = 0.5,
    sort_by: str = "count",  # "count" or "first_seen"
    figsize: tuple | None = None,
    out_path: str | None = None,
) -> tuple:
    """
    Two-panel event raster of `group_col`'s occurrences over time, split into
    a majority-attack panel and a majority-benign panel (by which class each
    value fires in more often). Each occurrence is drawn as a dot at its
    elapsed time; dots are bucketed per day/hour and sized by how many hits
    landed in that bucket, so bursts are visually distinguishable from
    single hits rather than all rendering as the same-size dot.

    Rows below `min_occurrences` (or outside the top `top_n` by total count)
    are dropped before plotting -- with hundreds of values, the long tail of
    1-2-hit signatures adds height without adding readable signal.

    Assumes df represents a single continuous timeline (as with CSCAS);
    elapsed time is measured from the global minimum timestamp in df.

    Parameters
    ----------
    df              : DataFrame returned by load_alerts() (or with the same
                      time/is_attack columns and a `group_col` column present)
    group_col       : column to group occurrences by, e.g. "signature" (full
                      alert text, CSCAS) or "short" (AIT-ADS's coarser
                      descriptor code)
    time_unit       : "hours" or "days" -- unit for the x-axis and for the
                      daily/hourly bucketing used to size markers
    min_occurrences : drop values with fewer than this many total hits
    top_n           : if set, keep only the top-N values by total count
                      (applied after min_occurrences)
    size_by_count   : if True, scale marker area by log(bucket count) so
                      bursts read as bigger dots; if False, all dots are
                      `marker_size` (old behaviour)
    marker_size     : base scatter marker size (points^2)
    alpha           : marker transparency (lower helps reveal density in bursts)
    sort_by         : "count" sorts each block by descending total hits;
                      "first_seen" sorts by earliest elapsed occurrence,
                      which lines up rows that start together and makes
                      simultaneous bursts easier to read as a block
    figsize         : optional override; default scales height with value count
    out_path        : optional save path
    """
    divisor = 3600.0 if time_unit == "hours" else 86400.0
    t0 = float(df["time"].min())
    df = df.copy()
    df["_elapsed"] = (df["time"].astype(float) - t0) / divisor

    stats = (
        df.groupby(group_col)["is_attack"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_attack", "count": "n_total"})
    )
    stats["majority_attack"] = 2 * stats["n_attack"] >= stats["n_total"]
    stats["first_seen"] = df.groupby(group_col)["_elapsed"].min()

    n_before = len(stats)
    stats = stats[stats["n_total"] >= min_occurrences]
    if top_n is not None:
        stats = stats.sort_values("n_total", ascending=False).head(top_n)
    n_dropped = n_before - len(stats)
    df = df[df[group_col].isin(stats.index)]

    sort_col = "first_seen" if sort_by == "first_seen" else "n_total"
    sort_asc = sort_by == "first_seen"

    attack_block = stats[stats["majority_attack"]].sort_values(
        sort_col, ascending=sort_asc
    )
    benign_block = stats[~stats["majority_attack"]].sort_values(
        sort_col, ascending=sort_asc
    )
    n_attack_rows, n_benign_rows = len(attack_block), len(benign_block)
    n_vals = n_attack_rows + n_benign_rows

    if n_vals == 0:
        raise ValueError(
            f"No {group_col} values left after filtering (min_occurrences={min_occurrences}, "
            f"top_n={top_n}) -- loosen the filter."
        )

    if figsize is None:
        figsize = (16, max(6, 0.16 * n_vals))

    height_ratios = [max(n_attack_rows, 1), max(n_benign_rows, 1)]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
    )

    x_max = float(df["_elapsed"].max())

    def _draw_panel(ax, block, title):
        val_to_row = {val: i for i, val in enumerate(block.index)}
        sub = df[df[group_col].isin(block.index)].copy()
        sub["_row"] = sub[group_col].map(val_to_row)

        if size_by_count:
            bucket = np.floor(sub["_elapsed"]).astype(int)
            counts = sub.groupby([group_col, bucket, "is_attack"])[
                "_elapsed"
            ].transform("size")
            sizes = marker_size * (1 + np.log1p(counts))
        else:
            sizes = marker_size

        colors = np.where(sub["is_attack"].values, _C_ATTACK, _C_BENIGN)
        ax.scatter(
            sub["_elapsed"],
            sub["_row"],
            s=sizes,
            c=colors,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels(block.index, fontsize=5)
        ax.set_ylim(len(block) - 0.5, -0.5)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)

    _draw_panel(axes[0], attack_block, f"Majority-attack ({n_attack_rows})")
    _draw_panel(axes[1], benign_block, f"Majority-benign ({n_benign_rows})")

    axes[1].set_xlim(0, x_max)
    axes[1].set_xlabel(f"Elapsed time ({time_unit})", fontsize=11)

    subtitle = f"{n_vals} shown"
    if n_dropped:
        subtitle += f", {n_dropped} dropped below {min_occurrences} occurrences"
    fig.suptitle(
        f"{group_col.capitalize()} occurrence burst raster ({subtitle})",
        fontsize=12,
    )
    fig.legend(
        handles=[
            mpatches.Patch(color=_C_BENIGN, alpha=0.9, label="Benign hit"),
            mpatches.Patch(color=_C_ATTACK, alpha=0.9, label="Attack hit"),
        ],
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.99, 0.99),
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_path)
    return fig, axes


def plot_signature_purity_pie(
    df: pd.DataFrame,
    figsize: tuple = (7, 7),
    out_path: str | None = None,
) -> tuple:
    """
    Pie chart classifying each unique signature as always-benign,
    always-attack, or mixed (fires in both classes at least once).

    Slice size = number of distinct signatures in each category, so this
    answers "how much of the signature space is each type", as opposed to
    "how much alert volume" (which plot_top_alert_signatures covers).
    """
    stats = df.groupby("signature")["is_attack"].agg(["sum", "count"])
    stats.columns = ["n_attack", "n_total"]
    n_benign = stats["n_total"] - stats["n_attack"]

    always_benign = int((stats["n_attack"] == 0).sum())
    always_attack = int((n_benign == 0).sum())
    mixed = len(stats) - always_benign - always_attack

    labels = ["Always benign", "Always attack", "Mixed"]
    counts = [always_benign, always_attack, mixed]
    colors = [_C_BENIGN, _C_ATTACK, "#CCBB44"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        counts,
        labels=[f"{lbl}\n(n={c})" for lbl, c in zip(labels, counts)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(alpha=0.85, linewidth=0.5, edgecolor="white"),
        textprops=dict(fontsize=10),
    )
    ax.set_title(f"Signature purity ({len(stats)} unique signatures)", fontsize=12)
    ax.axis("equal")
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def classify_signatures(df: pd.DataFrame, sig_col: str = "signature") -> pd.Series:
    """
    Per-value verdict for every unique `sig_col` value: 'benign' (never seen
    with is_attack), 'attack' (never seen without), or 'mixed' (both).
    Index is unique `sig_col` values. Shared by plot_signature_purity_pie's
    inline version and the vocabulary-churn / activity plots below, and by
    thesis.data.eda's signature-behaviour tables.
    """
    stats = df.groupby(sig_col)["is_attack"].agg(["sum", "count"])
    n_benign = stats["count"] - stats["sum"]
    return pd.Series(
        np.select(
            [stats["sum"] == 0, n_benign == 0],
            ["benign", "attack"],
            default="mixed",
        ),
        index=stats.index,
    )


def plot_signature_activity_bins(
    df: pd.DataFrame,
    scenario: str,
    sig_col: str = "signature",
    bin_freq: str = "1h",
    figsize: tuple = (14, 4),
    out_path: str | None = None,
) -> tuple:
    """
    Alert volume over time (line, log scale) with the count of distinct
    active `sig_col` values per bin stacked underneath, split into
    benign-only / attack-only / mixed classes.

    Companion to plot_signature_vocabulary_churn: this shows *how many*
    signatures are active per bin and how much volume they produce; the
    churn plot shows whether that active set is the *same* signatures
    recurring or a constantly-replaced one.
    """
    d = df.copy()
    d["_bin"] = d["timestamp"].dt.floor(bin_freq)
    vol = d.groupby("_bin").agg(
        total=("is_attack", "size"), attacks=("is_attack", "sum")
    )

    sig_class = classify_signatures(d, sig_col)
    d["_sig_class"] = d[sig_col].map(sig_class)
    sig_ct = (
        d.groupby(["_bin", "_sig_class"])[sig_col]
        .nunique()
        .unstack("_sig_class", fill_value=0)
        .reindex(columns=["benign", "attack", "mixed"], fill_value=0)
        .reindex(vol.index, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    width_days = pd.Timedelta(bin_freq).total_seconds() / 86400 * 0.9
    bar_kwargs = dict(width=width_days, alpha=0.35, linewidth=0)
    ax2.bar(
        sig_ct.index,
        sig_ct["benign"],
        label="Benign-only signatures",
        color=_C_BENIGN,
        **bar_kwargs,
    )
    ax2.bar(
        sig_ct.index,
        sig_ct["attack"],
        bottom=sig_ct["benign"],
        label="Attack-only signatures",
        color=_C_ATTACK,
        **bar_kwargs,
    )
    ax2.bar(
        sig_ct.index,
        sig_ct["mixed"],
        bottom=sig_ct["benign"] + sig_ct["attack"],
        label="Mixed signatures",
        color="#CCBB44",
        **bar_kwargs,
    )
    ax2.set_ylabel(f"# distinct {sig_col} values active / bin")

    # keep the alert-volume lines drawn on top of the bars
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(
        vol.index, vol["total"], label="Total alerts", linewidth=0.8, color="dimgray"
    )
    ax.plot(
        vol.index, vol["attacks"], label="Attack alerts", linewidth=0.8, color=_C_ATTACK
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.set_title(
        f"{scenario}: alert volume ({bin_freq} bins, log scale) with active "
        f"{sig_col} count per bin, stacked by class"
    )
    ax.set_ylabel("# alerts")
    ax.set_yscale("log")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_signature_vocabulary_churn(
    df: pd.DataFrame,
    scenario: str,
    sig_col: str = "signature",
    bin_freq: str = "1h",
    last_frac: float = 0.25,
    figsize: tuple = (13, 4.5),
    out_path: str | None = None,
) -> tuple:
    """
    Two-panel view of whether the active `sig_col` vocabulary keeps turning
    over across the scenario or saturates early:
    (a) cumulative distinct values ever seen vs elapsed time, split by class;
    (b) per-bin split of active values into first-ever appearances ("new")
        vs previously-seen ("returning").

    A cumulative curve that keeps climbing (rather than flattening) means
    new signatures keep arriving throughout the window -- exactly the
    condition that would make a schema mined on an early slice miss
    signatures seen later.
    """
    d = df.copy()
    sig_class = classify_signatures(d, sig_col)
    fs = d.groupby(sig_col)["timestamp"].min().to_frame("first_seen")
    fs["sig_class"] = sig_class.reindex(fs.index)

    alls = fs["first_seen"].sort_values()
    n_total = len(alls)
    t0, t1 = alls.iloc[0], alls.iloc[-1]
    half_date = alls.iloc[n_total // 2]
    last_start = t0 + (1 - last_frac) * (t1 - t0)
    n_last = int((alls > last_start).sum())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

    colors = {"benign": _C_BENIGN, "attack": _C_ATTACK, "mixed": "#CCBB44"}
    for cls, color in colors.items():
        s = fs.loc[fs.sig_class == cls, "first_seen"].sort_values()
        if len(s):
            axA.step(
                s.values,
                np.arange(1, len(s) + 1),
                where="post",
                color=color,
                label=f"{cls} (n={len(s)})",
            )
    axA.step(
        alls.values,
        np.arange(1, n_total + 1),
        where="post",
        color="0.35",
        lw=1.3,
        label=f"all (n={n_total})",
    )

    axA.axvspan(last_start, t1, color="0.9", zorder=0)
    axA.annotate(
        f"{n_last} {sig_col}s first seen\nin the last {last_frac:.0%} of the timeline",
        xy=(last_start, n_total * 0.55),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6),
    )
    axA.plot(
        [t0, half_date, half_date],
        [n_total / 2, n_total / 2, 0],
        ls=":",
        color="0.5",
        lw=0.8,
    )
    axA.plot(half_date, n_total / 2, "o", color="0.3", ms=5)
    axA.annotate(
        f"half of all {n_total} seen by\n{half_date:%m-%d %Hh}",
        xy=(half_date, n_total / 2),
        xytext=(12, -30),
        textcoords="offset points",
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6),
    )
    axA.set_ylim(0, n_total * 1.05)
    axA.set_ylabel(f"cumulative # distinct {sig_col}s seen")
    axA.set_title("(a) Cumulative distinct signatures over time")
    axA.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    axA.legend(fontsize=8, loc="lower right")

    bs = d[["timestamp", sig_col]].copy()
    bs["bucket"] = bs["timestamp"].dt.floor(bin_freq)
    bs = bs[["bucket", sig_col]].drop_duplicates()
    bs["first_bucket"] = bs.groupby(sig_col)["bucket"].transform("min")
    bs["status"] = np.where(bs["bucket"] == bs["first_bucket"], "new", "returning")
    pb = (
        bs.groupby(["bucket", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["returning", "new"], fill_value=0)
    )
    width_days = pd.Timedelta(bin_freq).total_seconds() / 86400 * 0.9
    axB.bar(
        pb.index,
        pb["returning"],
        width=width_days,
        color="0.7",
        label="previously seen",
    )
    axB.bar(
        pb.index,
        pb["new"],
        bottom=pb["returning"],
        width=width_days,
        color="tab:orange",
        label="first appearance",
    )
    y_top = float((pb["returning"] + pb["new"]).max()) if len(pb) else 1.0
    axB.axvspan(last_start, t1, color="0.9", zorder=0)
    axB.annotate(
        f"{n_last} first appearances\nin the last {last_frac:.0%}",
        xy=(last_start, y_top * 0.95),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="0.25",
    )
    axB.set_ylabel(f"# distinct {sig_col}s active / bin")
    axB.set_title(f"(b) New vs returning {sig_col}s per {bin_freq} bin")
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    axB.legend(fontsize=8, loc="upper right")

    for ax in (axA, axB):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"{scenario}: {sig_col} vocabulary churn", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, (axA, axB)


def plot_signature_activity_heatmap(
    df: pd.DataFrame,
    scenario: str,
    sig_col: str = "signature",
    n_tbins: int = 120,
    figsize: tuple = (13, 5),
    out_path: str | None = None,
) -> tuple:
    """
    Time x `sig_col` activity heatmap, attack and benign side by side,
    values ordered by first appearance (bottom = earliest). Cell colour
    (log) = alert count in that time/signature bin. A vertical stripe
    spanning many rows is many signatures co-firing (a campaign); a value
    confined to a narrow x-range near the bottom is an early-appearing,
    short-lived signature. Quantifies "active whole window vs briefly" --
    the event-raster scatter (plot_signature_event_raster) only shows this
    visually.
    """
    import matplotlib.colors as mcolors

    g0 = float(df["time"].min())
    span_days = max((float(df["time"].max()) - g0) / 86400.0, 1e-6)

    def _panel(sub, ax, title, cmap):
        order = sub.groupby(sig_col)["time"].min().sort_values().index
        n_order = len(order)
        if n_order == 0:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            return
        row = {s: i for i, s in enumerate(order)}
        x = (sub["time"].values.astype(float) - g0) / 86400.0
        y = sub[sig_col].map(row).values
        H, xe, ye = np.histogram2d(
            x, y, bins=[n_tbins, n_order], range=[[0, span_days], [0, n_order]]
        )
        pcm = ax.pcolormesh(
            xe,
            ye,
            H.T,
            cmap=cmap,
            rasterized=True,
            norm=mcolors.LogNorm(vmin=1, vmax=max(H.max(), 1)),
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("elapsed time (days)")
        ax.set_ylabel(f"{n_order} {sig_col}s (ordered by first appearance)")
        ax.set_ylim(0, n_order)
        fig.colorbar(pcm, ax=ax, pad=0.02, label="alerts / bin")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)
    _panel(
        df[df["is_attack"]],
        axA,
        f"Attack {sig_col}s (n={df.loc[df['is_attack'], sig_col].nunique()})",
        "Reds",
    )
    _panel(
        df[~df["is_attack"]],
        axB,
        f"Benign {sig_col}s (n={df.loc[~df['is_attack'], sig_col].nunique()})",
        "Blues",
    )
    fig.suptitle(f"{scenario}: {sig_col} activity over time", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, (axA, axB)


def plot_attack_temporal_concentration(
    df: pd.DataFrame,
    scenario: str,
    sig_col: str = "signature",
    figsize: tuple = (16, 4.3),
    out_path: str | None = None,
) -> tuple:
    """
    Three complementary views of whether attacks cluster in time rather
    than arriving at a steady rate:
    (a) cumulative attack count vs elapsed time, against the
        uniform-arrival diagonal;
    (b) share of attacks falling in the busiest slices of the timeline;
    (c) CDF of the gap between consecutive attacks (any `sig_col` value) vs
        between consecutive occurrences of the SAME value, against an
        exponential of the same mean (what a memoryless, non-clustered
        arrival process would give). If the pooled curve clusters at short
        gaps but the per-signature curve doesn't, the clustering is
        different signatures co-firing, not any one signature repeating.
    """
    att = df.loc[df["is_attack"]].sort_values("time").reset_index(drop=True)
    if len(att) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Not enough attack alerts to assess concentration",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        _save(fig, out_path)
        return fig, ax

    t0, t1 = att["time"].min(), df["time"].max()
    frac_time = (att["time"].values.astype(float) - t0) / (t1 - t0)
    frac_att = np.arange(1, len(att) + 1) / len(att)

    edges = np.linspace(0, 1, 1001)
    binned = np.histogram(frac_time, bins=edges)[0]
    busiest = np.sort(binned)[::-1].cumsum() / binned.sum()

    def _time_for(pp):
        return float(frac_time[min(np.searchsorted(frac_att, pp), len(att) - 1)])

    q_facts = [(pp, _time_for(pp)) for pp in (0.25, 0.5, 0.8)]
    b_facts = [(q, float(busiest[int(q * 1000) - 1])) for q in (0.01, 0.05, 0.10)]
    max_dev = float(np.max(np.abs(frac_att - frac_time)))

    fig, (axL, axR, axG) = plt.subplots(1, 3, figsize=figsize)

    axL.fill_between(
        frac_time,
        frac_time,
        frac_att,
        where=frac_att >= frac_time,
        color=_C_ATTACK,
        alpha=0.12,
    )
    axL.fill_between(
        frac_time,
        frac_time,
        frac_att,
        where=frac_att < frac_time,
        color=_C_BENIGN,
        alpha=0.12,
    )
    axL.plot(frac_time, frac_att, lw=1.8, color=_C_ATTACK, label="attack alerts")
    axL.plot([0, 1], [0, 1], "k--", lw=1, label="uniform arrival")
    for pp, tf in q_facts:
        axL.plot([tf, tf, 0], [0, pp, pp], color="0.5", lw=0.8, ls=":")
        axL.plot(tf, pp, "o", color=_C_ATTACK, ms=5)
        axL.annotate(
            f"{pp:.0%} of attacks in {tf:.0%} of time",
            xy=(tf, pp),
            xytext=(min(tf + 0.03, 0.55), max(pp - 0.16, 0.03)),
            fontsize=8,
            color="0.2",
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6),
        )
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.set_xlabel("fraction of elapsed time")
    axL.set_ylabel("fraction of attack alerts")
    axL.set_title(
        f"(a) Cumulative attacks over time\n(max deviation from uniform: {max_dev:.0%})",
        fontsize=9,
    )
    axL.legend(loc="upper left", fontsize=8)

    axR.plot(
        edges[1:] * 100, busiest * 100, lw=1.8, color=_C_ATTACK, label="attack alerts"
    )
    axR.plot([0, 100], [0, 100], "k--", lw=1, label="uniform")
    # One consolidated text block (like panel (c)'s) rather than a label
    # pinned to each point's own (x, y) -- when concentration is extreme (a
    # single bursty signature dominates, common in AIT-ADS) all three
    # busiest-% points land within a few % of 100 and per-point labels
    # collide regardless of offset.
    for q, sh in b_facts:
        axR.plot(q * 100, sh * 100, "o", color=_C_ATTACK, ms=5)
    facts_txt = "\n".join(
        f"busiest {q:.0%} -> {sh:.0%}  ({sh / q:.0f}x)" for q, sh in b_facts
    )
    axR.text(
        0.97,
        0.97,
        facts_txt,
        transform=axR.transAxes,
        fontsize=7.5,
        color="0.2",
        ha="right",
        va="top",
        family="monospace",
    )
    axR.set_xlim(0, 20)
    axR.set_ylim(0, max(40, b_facts[-1][1] * 100 + 12))
    axR.set_xlabel("% of timeline (busiest 1h bins first)")
    axR.set_ylabel("% of attack alerts")
    axR.set_title("(b) Concentration in the busiest time slices", fontsize=9)
    axR.legend(loc="lower right", fontsize=8)

    gap = np.diff(att["time"].values.astype(float))
    n_zero = int((gap == 0).sum())
    gs = np.sort(gap)
    mean_gap = gap.mean() if gap.mean() > 0 else 1.0

    _gsig = [
        np.diff(np.sort(t.values.astype(float)))
        for _, t in att.groupby(sig_col)["time"]
        if len(t) > 1
    ]
    gsig = np.sort(np.concatenate(_gsig)) if _gsig else np.array([1.0])

    def _cdf(a):
        return np.clip(a, 1, None), np.arange(1, len(a) + 1) / len(a)

    x_all, y_all = _cdf(gs)
    x_sig, y_sig = _cdf(gsig)
    grid = np.logspace(0, np.log10(max(gs.max(), 1) + 1), 200)
    axG.plot(x_all, y_all, lw=1.8, color=_C_ATTACK, label="any signature")
    axG.plot(x_sig, y_sig, lw=1.8, color="tab:orange", label="same signature")
    axG.plot(grid, 1 - np.exp(-grid / mean_gap), "k--", lw=1, label="exponential")
    for sec, lbl in [(60, "1 min"), (3600, "1 h"), (86400, "1 d")]:
        if sec <= gs.max():
            axG.axvline(sec, ls=":", color="0.6", lw=0.8)
            axG.text(
                sec,
                0.02,
                lbl,
                rotation=90,
                fontsize=7,
                va="bottom",
                ha="right",
                color="0.4",
            )

    m_all, m_sig = float(np.median(gs)), float(np.median(gsig))
    f_all, f_sig = (gs <= 60).mean(), (gsig <= 60).mean()
    axG.text(
        0.03,
        0.97,
        f"gaps <= 1 min:  any sig {f_all:.0%},  same sig {f_sig:.0%}\n"
        f"median gap:   any sig {m_all:,.0f} s,  same sig {m_sig:,.0f} s\n"
        f"{n_zero:,} coincident (0 s)",
        transform=axG.transAxes,
        fontsize=6.5,
        va="top",
        color="0.3",
        family="monospace",
    )

    axG.set_xscale("log")
    axG.set_ylim(0, 1)
    axG.set_xlabel("gap between consecutive attacks (s, log)")
    axG.set_ylabel("CDF")
    axG.set_title(
        "(c) Arrival-gap distribution\n(pooled clusters more than any one signature)",
        fontsize=9,
    )
    axG.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(f"{scenario}: temporal concentration of attacks", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, (axL, axR, axG)


def plot_scenario_overview(
    df: pd.DataFrame,
    groups_df: pd.DataFrame | None = None,
    figsize: tuple | None = None,
    out_path: str | None = None,
) -> tuple:
    """
    Summary table rendered as a figure — one row per scenario.

    Alert columns: start/end date, duration, total/benign/attack counts, attack %,
    unique alert  , attack types present.
    When groups_df is provided, four alert_group columns are appended:
    Tx Total, Tx Benign, Tx Attack (+mixed), Tx Att %.
    """
    scenarios = _ordered_scenarios(df)
    has_tx = groups_df is not None

    rows = []
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        n_total = len(sc_df)
        n_benign = (~sc_df["is_attack"]).sum()
        n_attack = sc_df["is_attack"].sum()
        t_start = pd.to_datetime(sc_df["time"].min(), unit="s", utc=True)
        t_end = pd.to_datetime(sc_df["time"].max(), unit="s", utc=True)
        duration_days = (sc_df["time"].max() - sc_df["time"].min()) / 86400
        n_sig = sc_df["signature"].nunique()
        n_attack_types = sc_df.loc[sc_df["is_attack"], "label"].nunique()
        row = [
            sc,
            t_start.strftime("%Y-%m-%d"),
            t_end.strftime("%Y-%m-%d"),
            f"{duration_days:.1f}",
            f"{n_total:,}",
            f"{n_benign:,}",
            f"{n_attack:,}",
            f"{100 * n_attack / n_total:.1f}%",
            str(n_sig),
            str(n_attack_types),
        ]
        if has_tx:
            sc_tx = groups_df[groups_df["scenario"] == sc]
            tx_total = len(sc_tx)
            tx_benign = (sc_tx["group_label"] == "benign").sum()
            tx_attack = sc_tx["group_label"].isin(["attack", "mixed"]).sum()
            row += [
                f"{tx_total:,}",
                f"{tx_benign:,}",
                f"{tx_attack:,}",
                f"{100 * tx_attack / tx_total:.1f}%" if tx_total else "—",
            ]
        rows.append(row)

    col_labels = [
        "Scenario",
        "Start",
        "End",
        "Days",
        "Total",
        "Benign",
        "Attack",
        "Attack %",
        "Alert\nsigs",
        "Attack\ntypes",
    ]
    if has_tx:
        col_labels += ["Tx\nTotal", "Tx\nBenign", "Tx\nAttack", "Tx\nAtt %"]

    if figsize is None:
        figsize = (18, 4) if has_tx else (13, 4)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#DDDDDD")
        tbl[0, j].set_text_props(fontweight="bold")

    ax.set_title("Dataset overview — per-scenario statistics", fontsize=12, pad=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


# ─── per-scenario alert_group plots (used by data.eda.run_scenario_eda) ──────


def plot_alert_group_size_distribution(
    alert_groups: pd.DataFrame,
    scenario: str,
    figsize: tuple = (10, 6),
    out_path: str | None = None,
) -> tuple:
    """Histogram of alert_group size (# items), split benign vs attack."""
    groups = alert_groups.copy()
    groups["group_size"] = groups["items"].apply(len)

    fig, ax = plt.subplots(figsize=figsize)
    for label, color in [("benign", _C_BENIGN), ("attack", _C_ATTACK)]:
        subset = groups.loc[groups["group_label"] == label, "group_size"]
        ax.hist(subset, bins=20, alpha=0.7, color=color, label=label)
    ax.set_title(f"Distribution of alert_group size (scenario={scenario})")
    ax.set_xlabel("Number of items in alert_group")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_pair_support_scatter(
    pair_metrics_df: pd.DataFrame,
    scenario: str,
    figsize: tuple = (7, 6),
    out_path: str | None = None,
) -> tuple:
    """Scatter of item-pair support: attack vs benign (log-log)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        pair_metrics_df["support_benign"], pair_metrics_df["support_attack"], alpha=0.5
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Support (benign)")
    ax.set_ylabel("Support (attack)")
    ax.set_title(f"Pair support: attack vs benign (scenario={scenario})")
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_label_distribution_table(
    label_dist_df: pd.DataFrame,
    out_path: str | None = None,
) -> tuple:
    """
    Table of per-scenario label breakdown (count, % of data, % of attacks).

    `label_dist_df` is the tidy frame from data.eda.compute_label_distribution_table:
    columns scenario/label/count/pct_of_data/pct_of_attacks, attack labels before
    'false_positive' within each scenario (row order as given is preserved).
    """
    col_labels = ["Scenario", "Label", "Count", "% of data", "% of attacks"]
    scenarios = label_dist_df["scenario"].drop_duplicates().tolist()
    sc_bg = ["#EEF3FF", "#FFF8EE"]
    sc_color = {sc: sc_bg[i % 2] for i, sc in enumerate(scenarios)}

    cell_text = []
    cell_colors = []
    seen_scenario = set()
    for _, row in label_dist_df.iterrows():
        first = row["scenario"] not in seen_scenario
        seen_scenario.add(row["scenario"])
        pct_attacks = (
            f"{row['pct_of_attacks']:.1f}%" if pd.notna(row["pct_of_attacks"]) else "—"
        )
        cell_text.append(
            [
                row["scenario"] if first else "",
                row["label"],
                f"{int(row['count']):,}",
                f"{row['pct_of_data']:.1f}%",
                pct_attacks,
            ]
        )
        cell_colors.append([sc_color[row["scenario"]]] * len(col_labels))

    n_rows = len(cell_text)
    fig_height = max(5, 0.35 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2D2D2D")
        tbl[0, j].set_text_props(fontweight="bold", color="white")

    ax.set_title("Alert label distribution per scenario", fontsize=12, pad=10)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


# ─── per-host plots (used by data.eda.run_host_scenario_eda) ────────────────


def plot_host_alert_type_heatmap(
    df: pd.DataFrame,
    scenario: str,
    out_path: str | None = None,
) -> tuple:
    """Alert-type x host heatmap (count, log-scale)."""
    pivot = df.groupby(["host", "short"]).size().unstack(fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(
        figsize=(max(12, len(pivot.columns) * 0.55), max(4, len(pivot) * 0.55 + 1.5))
    )
    data = pivot.values.astype(float)
    data_masked = np.ma.masked_where(data == 0, data)

    vmin = max(1, data_masked.min()) if data_masked.count() else 1
    im = ax.imshow(
        data_masked,
        aspect="auto",
        cmap="YlOrRd",
        norm=LogNorm(vmin=vmin, vmax=data.max() + 1),
    )
    plt.colorbar(im, ax=ax, label="Alert count (log scale)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f"{scenario} — Alert types per host", fontsize=12)
    ax.set_xlabel("Alert type (short code)")
    ax.set_ylabel("Host")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            if v > 0:
                text_color = "white" if v > data.max() / 4 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(v):,}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_host_timeline(
    df: pd.DataFrame,
    scenario: str,
    bin_minutes: int = 5,
    out_path: str | None = None,
) -> tuple:
    """Per-host alert rate over time (stacked subplots, benign vs attack)."""
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    n = len(hosts)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t_start = df["time"].min()
    t_end = df["time"].max()
    bin_s = bin_minutes * 60
    edges = np.arange(t_start, t_end + bin_s, bin_s)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    hours_elapsed = (bin_centers - t_start) / 3600

    for ax, host in zip(axes, hosts):
        hdf = df[df["host"] == host]
        benign = hdf[~hdf["is_attack"]]["time"].values
        attack = hdf[hdf["is_attack"]]["time"].values

        b_counts, _ = np.histogram(benign, bins=edges)
        a_counts, _ = np.histogram(attack, bins=edges)

        b_rate = np.where(b_counts > 0, b_counts / (bin_s / 60), np.nan)
        a_rate = np.where(a_counts > 0, a_counts / (bin_s / 60), np.nan)

        ax.fill_between(
            hours_elapsed,
            b_rate,
            alpha=0.6,
            color=_C_BENIGN,
            label="benign",
            step="mid",
        )
        ax.fill_between(
            hours_elapsed,
            a_rate,
            alpha=0.8,
            color=_C_ATTACK,
            label="attack",
            step="mid",
        )

        ax.set_ylabel(host, rotation=0, labelpad=6, ha="right", fontsize=8, va="center")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(axis="y", labelsize=7)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    axes[-1].set_xlabel("Elapsed time (h)")
    fig.suptitle(
        f"{scenario} — Alert rate per host ({bin_minutes}-min bins, alerts/min)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_host_interarrival_cdf(
    df: pd.DataFrame,
    scenario: str,
    delta: float,
    figsize: tuple = (9, 5),
    out_path: str | None = None,
) -> tuple:
    """CDF of inter-arrival times per host (log x-axis)."""
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab10")
    for i, host in enumerate(hosts):
        hdf = df[df["host"] == host].sort_values("time")
        times = hdf["time"].values
        iat = np.diff(np.sort(times)).astype(float) if len(times) >= 2 else np.array([])
        if len(iat) == 0:
            continue
        iat_sorted = np.sort(iat)
        cdf = np.arange(1, len(iat_sorted) + 1) / len(iat_sorted)
        ax.plot(iat_sorted, cdf, label=host, color=cmap(i % 10), linewidth=1.5)

    ax.axvline(
        delta,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"δ={delta}s",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Inter-arrival time (seconds, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{scenario} — Inter-arrival time CDF per host")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_host_burst_profile(
    host_summaries: list[dict],
    scenario: str,
    delta: float,
    figsize: tuple = (10, 6),
    out_path: str | None = None,
) -> tuple:
    """Bar chart: max stream size and % of IATs below delta, per host."""
    hosts = [s["host"] for s in host_summaries]
    max_stream = [s["max_stream_alerts"] for s in host_summaries]
    pct_sub_delta = [s["pct_iat_sub_delta"] for s in host_summaries]

    x = np.arange(len(hosts))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    bars = ax1.bar(x, max_stream, color=_C_ATTACK, alpha=0.8)
    ax1.set_ylabel("Max stream size (alerts)")
    ax1.set_title(f"{scenario} — Per-host burst profile")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, v in zip(bars, max_stream):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2.bar(x, pct_sub_delta, color=_C_BENIGN, alpha=0.8)
    ax2.axhline(50, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(f"% inter-arrival times < δ={delta}s")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hosts, rotation=25, ha="right", fontsize=8)
    ax2.set_xlabel("Host")
    for i, v in enumerate(pct_sub_delta):
        ax2.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _save(fig, out_path)
    return fig, (ax1, ax2)


def plot_host_type_overlap(
    df: pd.DataFrame,
    scenario: str,
    out_path: str | None = None,
) -> tuple:
    """
    Host x host Jaccard similarity of alert-type sets.
    Low Jaccard = hosts see different alert types -> per-host grouping
    separates semantically distinct streams.
    """
    hosts = df.groupby("host").size().sort_values(ascending=False).index.tolist()
    type_sets = {h: set(df[df["host"] == h]["short"].unique()) for h in hosts}

    n = len(hosts)
    matrix = np.zeros((n, n))
    for i, h1 in enumerate(hosts):
        for j, h2 in enumerate(hosts):
            s1, s2 = type_sets[h1], type_sets[h2]
            union = s1 | s2
            matrix[i, j] = len(s1 & s2) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.8)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Jaccard similarity of alert-type sets")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(hosts, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(hosts, fontsize=8)
    ax.set_title(
        f"{scenario} — Alert-type vocabulary overlap between hosts\n"
        "(low Jaccard → hosts speak different alert languages)"
    )
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > 0.6 else "black",
            )
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_host_exclusive_types(
    df: pd.DataFrame,
    scenario: str,
    out_path: str | None = None,
) -> tuple:
    """
    Stacked bar: for each alert type, how many hosts see it?
    Types seen at exactly 1 host are 'exclusive' — grouping by host keeps them pure.
    """
    host_per_type = df.groupby("short")["host"].nunique().sort_values(ascending=False)
    short_totals = df["short"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bucket_labels = ["1 host (exclusive)", "2–3 hosts", "4+ hosts"]
    buckets = [
        host_per_type[host_per_type == 1],
        host_per_type[(host_per_type >= 2) & (host_per_type <= 3)],
        host_per_type[host_per_type >= 4],
    ]
    colors = ["#228833", "#CCBB44", "#CC3311"]
    sizes = [len(b) for b in buckets]
    alert_volumes = [short_totals[b.index].sum() for b in buckets]

    ax1.bar(bucket_labels, sizes, color=colors, alpha=0.85)
    ax1.set_ylabel("Number of distinct alert types")
    ax1.set_title("Alert type host exclusivity\n(by # distinct types)")
    for i, (v, av) in enumerate(zip(sizes, alert_volumes)):
        ax1.text(
            i,
            v + 0.3,
            f"{v} types\n({av:,} alerts)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylim(0, max(sizes) * 1.25)

    exclusive = host_per_type[host_per_type == 1]
    exc_df = pd.DataFrame(
        {
            "short": exclusive.index,
            "host": [df[df["short"] == s]["host"].iloc[0] for s in exclusive.index],
            "n_alerts": [short_totals.get(s, 0) for s in exclusive.index],
        }
    ).sort_values(["host", "n_alerts"], ascending=[True, False])

    host_colors = {
        h: plt.get_cmap("tab10")(i % 10) for i, h in enumerate(exc_df["host"].unique())
    }
    bar_colors = [host_colors[h] for h in exc_df["host"]]

    ax2.barh(range(len(exc_df)), exc_df["n_alerts"], color=bar_colors, alpha=0.85)
    ax2.set_yticks(range(len(exc_df)))
    ax2.set_yticklabels(exc_df["short"], fontsize=7)
    ax2.set_xlabel("Alert count (log scale)")
    ax2.set_xscale("log")
    ax2.set_title("Host-exclusive alert types\n(color = host)")
    patches = [plt.Rectangle((0, 0), 1, 1, color=host_colors[h]) for h in host_colors]
    ax2.legend(patches, list(host_colors.keys()), fontsize=7, loc="lower right")

    fig.suptitle(f"{scenario} — Alert type host exclusivity", fontsize=11)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, (ax1, ax2)
