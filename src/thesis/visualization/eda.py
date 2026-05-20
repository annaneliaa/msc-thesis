"""EDA visualizations for the alert dataset (thesis dataset introduction chapter)."""

import os

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCENARIOS = [
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
]
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

# Paul Tol muted palette — colorblind-safe
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


def load_alerts(data_dir: str, scenarios: list[str] | None = None) -> pd.DataFrame:
    """
    Load all scenario alert CSVs.

    Returns a DataFrame with extra columns:
      scenario, is_attack, timestamp (UTC pd.Timestamp)
    """
    if scenarios is None:
        scenarios = SCENARIOS

    frames = []
    for sc in scenarios:
        path = os.path.join(data_dir, f"{sc}_alerts.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Alert CSV not found: {path}")
        df = pd.read_csv(path, dtype=str)
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["scenario"] = sc
        df["is_attack"] = df["time_label"].ne(BENIGN_LABEL) & df["time_label"].notna()
        frames.append(df)

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
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]
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
    fig.suptitle("Alert volume over time — all scenarios", fontsize=12, y=1.01)
    plt.tight_layout()

    _save(fig, out_path)
    return fig, axes


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
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

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
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

    counts = pd.DataFrame(index=ATTACK_TYPES, columns=scenarios, dtype=float).fillna(0)
    for sc in scenarios:
        sc_df = df[(df["scenario"] == sc) & df["is_attack"]]
        vc = sc_df["time_label"].value_counts()
        for at in ATTACK_TYPES:
            counts.loc[at, sc] = vc.get(at, 0)

    # drop attack types absent in all scenarios
    counts = counts.loc[counts.sum(axis=1) > 0]

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


def plot_top_alert_names(
    df: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple = (10, 6),
    out_path: str | None = None,
) -> tuple:
    """
    Top-K most frequent alert signatures (names), coloured by majority label.

    Useful for showing which IDS rules fire most often in the dataset.
    """
    vc = df["name"].value_counts().head(top_k)
    names = vc.index.tolist()

    # determine majority label for each name
    colors = []
    for name in names:
        sub = df[df["name"] == name]
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
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

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
    Gives the reader an intuition for how many alerts end up in one transaction.
    """
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

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


def plot_scenario_overview(
    df: pd.DataFrame,
    figsize: tuple = (13, 4),
    out_path: str | None = None,
) -> tuple:
    """
    Summary table rendered as a figure — one row per scenario.

    Columns: total alerts, benign count, attack count, attack %, duration (days),
             number of distinct alert types, number of attack types present.
    Suitable as Table N in the thesis dataset chapter.
    """
    scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

    rows = []
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        n_total = len(sc_df)
        n_benign = (~sc_df["is_attack"]).sum()
        n_attack = sc_df["is_attack"].sum()
        t_start = pd.to_datetime(sc_df["time"].min(), unit="s", utc=True)
        t_end = pd.to_datetime(sc_df["time"].max(), unit="s", utc=True)
        duration_days = (sc_df["time"].max() - sc_df["time"].min()) / 86400
        n_sig = sc_df["name"].nunique()
        n_attack_types = sc_df.loc[sc_df["is_attack"], "time_label"].nunique()
        rows.append(
            [
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
        )

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

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#DDDDDD")
        tbl[0, j].set_text_props(fontweight="bold")

    ax.set_title("Dataset overview — per-scenario statistics", fontsize=12, pad=12)
    plt.tight_layout()
    _save(fig, out_path)
    return fig, ax
