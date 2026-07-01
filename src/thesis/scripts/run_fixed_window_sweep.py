"""
Sweep over fixed-window sizes for fixed_window and fixed_window_host grouping.

For each (method, window_size) combination reports:
  - AlertGroup counts: total, benign, attack, mixed, purity
  - Alert-size distribution: mean, p50, p95, max, % single-alert
  - Token diversity: mean/max unique tokens, unique alert types per alert_group
  - Recurring alerts: % alert_groups with repeated alert IDs, mean repetition ratio
  - Window duration: mean and max seconds spanned by a alert_group
  - Train/test split quality: label distribution for a temporal split (default 70/30),
    flagging window sizes that produce a single-class train or test set.

This helps identify a suitable window size for each method. The reference value of
2 s used in earlier runs came from the Landauer et al. (2022) time-delta method,
not from a fixed-window evaluation.

Usage:
    python src/thesis/scripts/run_fixed_window_sweep.py fox \\
        [--windows 1 2 5 10 30 60] \\
        [--train-frac 0.7] \\
        [--out-dir artifacts/experiments/run_fixed_window_sweep]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.grouping.group_alerts import (
    group_alerts_fixed_window,
    group_alerts_fixed_window_host,
)
from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.schemas.groups import GroupSnapshot
from thesis.schemas.preprocessing import IncomingAlert, TokenizedAlert
from thesis.paths import CACHE_DIR


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))


class _Tee:
    """Mirror sys.stdout to a file for the duration of a with-block."""

    def __init__(self, log_path: Path) -> None:
        self._log = log_path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def __enter__(self) -> "_Tee":
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self._stdout
        self._log.close()

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._log.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()


_DEFAULT_WINDOWS = [1, 2, 5, 10, 30, 60]
_RAW_ALERTS_DIR = _REPO / "data" / "alerts_csv"
_BALANCED_ALERTS_DIR = _REPO / "artifacts" / "alerts" / "balanced"
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_fixed_window_sweep"
_BENIGN_LABEL = "false_positive"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _alerts_source_path(scenario: str, filter_method: str | None) -> Path:
    if filter_method:
        return _BALANCED_ALERTS_DIR / filter_method / f"{scenario}_alerts.json"
    return _RAW_ALERTS_DIR / f"{scenario}_alerts.txt"


def _load_alert_rows(path: Path) -> list[dict]:
    """Read alert rows from JSON (balanced) or CSV/txt (raw)."""
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tok_cache_path(scenario: str, filter_method: str | None) -> Path:
    batch = filter_method or "raw"
    return CACHE_DIR / scenario / "alerts" / f"tokenized_{batch}.json"


def _load_tok_cache(path: Path) -> list[TokenizedAlert]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        TokenizedAlert(
            alert_id=r["alert_id"],
            ts=r["ts"],
            time_norm=None,
            name=r.get("name"),
            ip=r.get("ip"),
            host=r.get("host"),
            short=r.get("short"),
            time_label=r.get("time_label"),
            event_label=r.get("event_label"),
            tokens=set(r.get("tokens", [])),
        )
        for r in rows
    ]


def _save_tok_cache(path: Path, alerts: list[TokenizedAlert]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "alert_id": a.alert_id,
            "ts": a.ts,
            "name": a.name,
            "ip": a.ip,
            "host": a.host,
            "short": a.short,
            "time_label": a.time_label,
            "event_label": a.event_label,
            "tokens": sorted(a.tokens),
        }
        for a in alerts
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f)


def load_and_tokenize(
    scenario: str, filter_method: str | None = None
) -> list[TokenizedAlert]:
    cache_path = _tok_cache_path(scenario, filter_method)
    if cache_path.exists():
        print(f"  [cache] Loading tokenized alerts from {cache_path}")
        return _load_tok_cache(cache_path)

    alerts_path = _alerts_source_path(scenario, filter_method)
    if not alerts_path.exists():
        raise FileNotFoundError(f"Alerts not found: {alerts_path}")

    rows = _load_alert_rows(alerts_path)

    tokenized: list[TokenizedAlert] = []
    skipped = 0
    for row in rows:
        try:
            alert = IncomingAlert.from_row(row)
            parsed = parse_incoming_alert(alert=alert, scenario=scenario)
            tok = tokenize_alert(parsed)
            tokenized.append(tok)
        except Exception:
            skipped += 1

    if skipped:
        print(f"  [warn] Skipped {skipped} alerts during tokenization")

    _save_tok_cache(cache_path, tokenized)
    print(f"  [cache] Saved tokenized alerts → {cache_path}")
    return tokenized


# ---------------------------------------------------------------------------
# In-memory snapshot builder (avoids disk I/O)
# ---------------------------------------------------------------------------


def _group_label_from_labels(alert_labels: set[str]) -> str:
    has_benign = _BENIGN_LABEL in alert_labels
    has_attack = any(lbl != _BENIGN_LABEL for lbl in alert_labels if lbl)
    if has_attack and has_benign:
        return "mixed"
    if has_attack:
        return "attack"
    return "benign"


def build_snapshots_in_memory(
    alerts: list[TokenizedAlert],
    records: list,
) -> list[GroupSnapshot]:
    alerts_by_id: dict[str, TokenizedAlert] = {a.alert_id: a for a in alerts}

    raw: dict[str, dict[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda r: alerts_by_id[r.alert_id].ts if r.alert_id in alerts_by_id else 0,
    ):
        alert = alerts_by_id.get(record.alert_id)
        if alert is None:
            continue

        if record.group_id not in raw:
            raw[record.group_id] = {
                "group_id": record.group_id,
                "method": record.method,
                "start_ts": alert.ts,
                "end_ts": alert.ts,
                "alert_ids": [],
                "items": set(),
                "sorted_items": [],
                "alert_ips": set(),
                "alert_labels": set(),
            }

        g = raw[record.group_id]
        g["alert_ids"].append(alert.alert_id)
        g["sorted_items"].append(set(alert.tokens))
        g["items"] |= set(alert.tokens)
        if alert.ip:
            g["alert_ips"].add(alert.ip)
        if alert.time_label is not None:
            g["alert_labels"].add(str(alert.time_label))
        g["start_ts"] = min(g["start_ts"], alert.ts)
        g["end_ts"] = max(g["end_ts"], alert.ts)

    snapshots = []
    for g in raw.values():
        group_label = (
            _group_label_from_labels(g["alert_labels"])
            if g["alert_labels"]
            else "benign"
        )
        snapshots.append(
            GroupSnapshot(
                group_id=g["group_id"],
                method=g["method"],
                version=1,
                start_ts=g["start_ts"],
                end_ts=g["end_ts"],
                alert_ids=g["alert_ids"],
                n_alerts=len(g["alert_ids"]),
                items=g["items"],
                sorted_items=g["sorted_items"],
                alert_ips=g["alert_ips"],
                alert_labels=g["alert_labels"],
                group_label=group_label,
                status="closed",
            )
        )

    return snapshots


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(snapshots: list[GroupSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {
            k: 0
            for k in [
                "n_tx",
                "n_benign",
                "n_attack",
                "n_mixed",
                "purity_pct",
                "n_single_alert",
                "pct_single_alert",
                "mean_alerts",
                "median_alerts",
                "p95_alerts",
                "max_alerts",
                "mean_unique_ids",
                "max_unique_ids",
                "mean_rep_ratio",
                "pct_repeated",
                "total_extra_alerts",
                "mean_tokens",
                "median_tokens",
                "max_tokens",
                "mean_alert_types",
                "max_alert_types",
                "mean_duration_s",
                "max_duration_s",
            ]
        }

    n_tx = len(snapshots)
    labels = [s.group_label for s in snapshots]
    n_benign = labels.count("benign")
    n_attack = labels.count("attack")
    n_mixed = labels.count("mixed")

    n_alerts_arr = np.array([s.n_alerts for s in snapshots], dtype=float)

    # Distinct alert profiles: how many unique (token-set) combinations appear in a
    # alert_group. Unlike alert_id (which ties to a specific second), this captures
    # whether the same alert TYPE recurs across different seconds in the window.
    profile_counts = np.array(
        [len({frozenset(item_set) for item_set in s.sorted_items}) for s in snapshots],
        dtype=float,
    )
    profile_ratios = n_alerts_arr / np.maximum(profile_counts, 1)
    pct_profile_repeated = float(np.mean(profile_ratios > 1.0) * 100)

    token_counts = np.array([len(s.items) for s in snapshots], dtype=float)

    # Alert-type diversity: distinct "short:<name>" tokens per alert_group.
    # Each unique short: token represents a distinct detector signature type.
    short_counts = np.array(
        [len([t for t in s.items if t.startswith("short:")]) for s in snapshots],
        dtype=float,
    )

    durations = np.array([s.end_ts - s.start_ts for s in snapshots], dtype=float)

    n_single = int(np.sum(n_alerts_arr == 1))

    # Per-label size stats
    attack_sizes = [s.n_alerts for s in snapshots if s.group_label == "attack"]
    benign_sizes = [s.n_alerts for s in snapshots if s.group_label == "benign"]

    return {
        "n_tx": n_tx,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "n_mixed": n_mixed,
        "purity_pct": (n_benign + n_attack) / n_tx * 100,
        "n_single_alert": n_single,
        "pct_single_alert": n_single / n_tx * 100,
        # Alert size
        "mean_alerts": float(n_alerts_arr.mean()),
        "median_alerts": float(np.median(n_alerts_arr)),
        "p95_alerts": float(np.percentile(n_alerts_arr, 95)),
        "max_alerts": int(n_alerts_arr.max()),
        # Per-label alert sizes
        "mean_attack_alerts": float(np.mean(attack_sizes)) if attack_sizes else 0.0,
        "max_attack_alerts": int(max(attack_sizes)) if attack_sizes else 0,
        "mean_benign_alerts": float(np.mean(benign_sizes)) if benign_sizes else 0.0,
        "max_benign_alerts": int(max(benign_sizes)) if benign_sizes else 0,
        # Recurring alert profiles within window.
        # profile_ratio = n_alerts / n_distinct_token_profiles:
        #   ratio > 1 means the same alert type (token combination) fires multiple
        #   times within the window — the signal is diluted by repetition.
        "mean_profile_ratio": float(profile_ratios.mean()),
        "pct_profile_repeated": pct_profile_repeated,
        "mean_profiles": float(profile_counts.mean()),
        "max_profiles": int(profile_counts.max()),
        # Token diversity
        "mean_tokens": float(token_counts.mean()),
        "median_tokens": float(np.median(token_counts)),
        "max_tokens": int(token_counts.max()),
        # Alert-type diversity (distinct detector signatures)
        "mean_alert_types": float(short_counts.mean()),
        "max_alert_types": int(short_counts.max()),
        # Temporal span of alert_groups
        "mean_duration_s": float(durations.mean()),
        "max_duration_s": float(durations.max()),
    }


# ---------------------------------------------------------------------------
# Train/test split quality
# ---------------------------------------------------------------------------


def compute_split_stats(
    snapshots: list[GroupSnapshot], train_frac: float = 0.7
) -> dict[str, Any]:
    """Temporal train/test split: first train_frac of snapshots (by start_ts) → train."""
    if not snapshots:
        return {
            "n_train": 0,
            "n_benign_train": 0,
            "n_attack_train": 0,
            "pct_attack_train": 0.0,
            "warn_train": "WARN",
            "n_test": 0,
            "n_benign_test": 0,
            "n_attack_test": 0,
            "pct_attack_test": 0.0,
            "warn_test": "WARN",
            "split_ok": False,
        }

    sorted_snaps = sorted(snapshots, key=lambda s: s.start_ts)
    split_idx = max(1, int(len(sorted_snaps) * train_frac))
    train = sorted_snaps[:split_idx]
    test = sorted_snaps[split_idx:]

    def _counts(snaps: list[GroupSnapshot]) -> dict[str, Any]:
        labels = [s.group_label for s in snaps]
        n_b = labels.count("benign")
        n_a = labels.count("attack")
        n_m = labels.count("mixed")
        n = len(snaps)
        n_positive = n_a + n_m
        classes = sum([n_b > 0, n_positive > 0])
        return {
            "n": n,
            "n_benign": n_b,
            "n_attack": n_a,
            "pct_attack": n_positive / n * 100 if n else 0.0,
            "single_class": classes <= 1,
        }

    tr = _counts(train)
    te = _counts(test)
    return {
        "n_train": tr["n"],
        "n_benign_train": tr["n_benign"],
        "n_attack_train": tr["n_attack"],
        "pct_attack_train": tr["pct_attack"],
        "warn_train": "WARN" if tr["single_class"] else "",
        "n_test": te["n"],
        "n_benign_test": te["n_benign"],
        "n_attack_test": te["n_attack"],
        "pct_attack_test": te["pct_attack"],
        "warn_test": "WARN" if te["single_class"] else "",
        "split_ok": not tr["single_class"] and not te["single_class"],
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _fmt(v: Any, precision: int = 1) -> str:
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def print_table(
    title: str,
    rows: list[dict],
    columns: list[tuple[str, str, int]],  # (key, header, width)
) -> None:
    print(f"\n{'─' * 4} {title} {'─' * max(0, 80 - len(title) - 6)}")
    header = "  ".join(h.ljust(w) for _, h, w in columns)
    sep = "  ".join("─" * w for _, _, w in columns)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(_fmt(row.get(k, "")).ljust(w) for k, _, w in columns)
        print(line)


def print_split_table(results: list[dict], train_frac: float) -> None:
    pct_tr = int(train_frac * 100)
    pct_te = 100 - pct_tr
    print_table(
        f"Train/test split ({pct_tr}/{pct_te}, temporal) — label distribution",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("n_train", "n_train", 8),
            ("n_benign_train", "ben_tr", 7),
            ("n_attack_train", "atk_tr", 7),
            ("pct_attack_train", "atk%_tr", 8),
            ("warn_train", "train?", 7),
            ("n_test", "n_test", 8),
            ("n_benign_test", "ben_te", 7),
            ("n_attack_test", "atk_te", 7),
            ("pct_attack_test", "atk%_te", 8),
            ("warn_test", "test?", 6),
        ],
    )


def print_all_tables(results: list[dict], train_frac: float = 0.7) -> None:
    print_table(
        "AlertGroup counts and purity",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("n_tx", "n_tx", 8),
            ("n_benign", "benign", 8),
            ("n_attack", "attack", 8),
            ("n_mixed", "mixed", 7),
            ("purity_pct", "purity%", 8),
            ("pct_single_alert", "single%", 8),
        ],
    )

    print_table(
        "Alert-size distribution — all alert_groups (alerts per alert_group)",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("mean_alerts", "mean", 8),
            ("median_alerts", "p50", 8),
            ("p95_alerts", "p95", 8),
            ("max_alerts", "max", 8),
        ],
    )

    print_table(
        "Alert-size distribution — by label (mean alerts; max alerts)",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("mean_attack_alerts", "atk mean", 9),
            ("max_attack_alerts", "atk max", 9),
            ("mean_benign_alerts", "ben mean", 9),
            ("max_benign_alerts", "ben max", 9),
        ],
    )

    print_table(
        "Token diversity (unique tokens and alert signature types per alert_group)",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("mean_tokens", "mean_tok", 9),
            ("median_tokens", "p50_tok", 9),
            ("max_tokens", "max_tok", 9),
            ("mean_alert_types", "mean_types", 11),
            ("max_alert_types", "max_types", 10),
        ],
    )

    print_table(
        "Recurring alert profiles (same type fires multiple times within window)",
        results,
        [
            ("method_label", "Method", 22),
            ("window_s", "Window", 8),
            ("pct_profile_repeated", "pct_recur%", 11),
            ("mean_profile_ratio", "mean_ratio", 11),
            ("mean_profiles", "mean_prof", 10),
            ("max_profiles", "max_prof", 9),
            ("mean_duration_s", "mean_dur_s", 11),
            ("max_duration_s", "max_dur_s", 10),
        ],
    )

    print_split_table(results, train_frac)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_C_METHOD = {
    "fixed_window": "#4C72B0",
    "fixed_window_host": "#55A868",
}
_SCENARIO_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_LS_METHOD = {
    "fixed_window": "-",
    "fixed_window_host": "--",
}
_MK_METHOD = {
    "fixed_window": "o",
    "fixed_window_host": "s",
}
_LBL_METHOD = {
    "fixed_window": "Fixed window",
    "fixed_window_host": "Fixed window (host)",
}


def _method_data(results: list[dict], method: str) -> tuple[list[float], list[dict]]:
    rows = [r for r in results if r["method"] == method]
    rows.sort(key=lambda r: r["window_val"])
    return [r["window_val"] for r in rows], rows


def _set_window_xticks(ax: plt.Axes, windows: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(windows)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def plot_sweep(
    results: list[dict], scenario: str, out_dir: Path, filter_method: str | None = None
) -> None:
    methods = list(dict.fromkeys(r["method"] for r in results))
    windows = sorted({r["window_val"] for r in results})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Fixed-window sweep — {scenario}", fontsize=12)

    # ── (0,0) AlertGroup count vs window size ─────────────────────────────
    ax = axes[0, 0]
    for m in methods:
        ws, rows = _method_data(results, m)
        ax.plot(
            ws,
            [r["n_tx"] for r in rows],
            color=_C_METHOD[m],
            ls=_LS_METHOD[m],
            marker=_MK_METHOD[m],
            markersize=5,
            linewidth=1.5,
            label=_LBL_METHOD[m],
        )
    _set_window_xticks(ax, windows)
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel("Number of alert_groups")
    ax.set_title("(A) Total alert_group count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)

    # ── (0,1) Mean alert_group size by label ───────────────────────────────
    ax = axes[0, 1]
    for m in methods:
        ws, rows = _method_data(results, m)
        c = _C_METHOD[m]
        mk = _MK_METHOD[m]
        lbl = _LBL_METHOD[m]
        ax.plot(
            ws,
            [r["mean_attack_alerts"] for r in rows],
            color=c,
            ls="-",
            marker=mk,
            markersize=5,
            linewidth=1.5,
            label=f"{lbl} – attack",
        )
        ax.plot(
            ws,
            [r["mean_benign_alerts"] for r in rows],
            color=c,
            ls=":",
            marker=mk,
            markersize=4,
            linewidth=1.2,
            alpha=0.7,
            label=f"{lbl} – benign",
        )
    _set_window_xticks(ax, windows)
    ax.set_yscale("log")
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel("Mean alerts per alert_group  (log scale)")
    ax.set_title("(B) Mean alert_group size by label")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5, which="both")

    # ── (1,0) Token diversity ─────────────────────────────────────────────
    ax = axes[1, 0]
    for m in methods:
        ws, rows = _method_data(results, m)
        c = _C_METHOD[m]
        mk = _MK_METHOD[m]
        lbl = _LBL_METHOD[m]
        ax.plot(
            ws,
            [r["mean_tokens"] for r in rows],
            color=c,
            ls="-",
            marker=mk,
            markersize=5,
            linewidth=1.5,
            label=f"{lbl} – unique tokens",
        )
        ax.plot(
            ws,
            [r["mean_alert_types"] for r in rows],
            color=c,
            ls=":",
            marker=mk,
            markersize=4,
            linewidth=1.2,
            alpha=0.7,
            label=f"{lbl} – signature types",
        )
    _set_window_xticks(ax, windows)
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel("Mean count per alert_group")
    ax.set_title("(C) Token diversity (larger window ≠ more unique types)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5)

    # ── (1,1) Profile recurrence ──────────────────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()
    legend_handles: list = []
    for m in methods:
        ws, rows = _method_data(results, m)
        c = _C_METHOD[m]
        mk = _MK_METHOD[m]
        lbl = _LBL_METHOD[m]
        (l1,) = ax.plot(
            ws,
            [r["pct_profile_repeated"] for r in rows],
            color=c,
            ls="-",
            marker=mk,
            markersize=5,
            linewidth=1.5,
            label=f"{lbl} – % recurring",
        )
        (l2,) = ax2.plot(
            ws,
            [r["mean_profile_ratio"] for r in rows],
            color=c,
            ls="--",
            marker=mk,
            markersize=4,
            linewidth=1.2,
            alpha=0.8,
            label=f"{lbl} – mean ratio",
        )
        legend_handles += [l1, l2]
    _set_window_xticks(ax, windows)
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel("% alert_groups with recurring alert profiles", color="k")
    ax2.set_ylabel("Mean ratio (alerts / distinct profiles)", color="grey")
    ax2.tick_params(labelcolor="grey")
    ax.set_title("(D) Alert profile recurrence within window")
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        f"data: filtered ({filter_method})" if filter_method else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path = out_dir / "sweep_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved → {out_path}")


def plot_split_quality(
    results: list[dict],
    scenario: str,
    out_dir: Path,
    train_frac: float,
    filter_method: str | None = None,
) -> None:
    """Plot attack-class % in train and test splits across window sizes."""
    methods = list(dict.fromkeys(r["method"] for r in results))
    windows = sorted({r["window_val"] for r in results})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    pct_tr = int(train_frac * 100)
    pct_te = 100 - pct_tr
    fig.suptitle(
        f"Train/test split quality ({pct_tr}/{pct_te} temporal) — {scenario}",
        fontsize=12,
    )

    for ax_idx, (split_key, split_label) in enumerate(
        [
            ("pct_attack_train", f"Train ({pct_tr}%)"),
            ("pct_attack_test", f"Test ({pct_te}%)"),
        ]
    ):
        ax = axes[ax_idx]
        for m in methods:
            ws, rows = _method_data(results, m)
            pct_vals = [r[split_key] for r in rows]
            # Mark single-class windows with a red X
            warn_key = "warn_train" if ax_idx == 0 else "warn_test"
            warn_flags = [r[warn_key] == "WARN" for r in rows]

            ax.plot(
                ws,
                pct_vals,
                color=_C_METHOD[m],
                ls=_LS_METHOD[m],
                marker=_MK_METHOD[m],
                markersize=5,
                linewidth=1.5,
                label=_LBL_METHOD[m],
            )
            # Highlight single-class windows
            warn_ws = [w for w, f in zip(ws, warn_flags) if f]
            warn_pct = [p for p, f in zip(pct_vals, warn_flags) if f]
            if warn_ws:
                ax.scatter(
                    warn_ws,
                    warn_pct,
                    marker="X",
                    s=80,
                    color=_C_METHOD[m],
                    edgecolors="red",
                    linewidths=1.5,
                    zorder=5,
                    label=f"{_LBL_METHOD[m]} – single-class",
                )

        _set_window_xticks(ax, windows)
        ax.set_xlabel("Window size (s)")
        ax.set_ylabel("Attack-class alert_groups (%)")
        ax.set_title(f"{split_label} — attack %")
        ax.set_ylim(-2, 102)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axhline(100, color="grey", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        f"data: filtered ({filter_method})" if filter_method else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path = out_dir / "split_quality_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Split quality plots saved → {out_path}")


def plot_cross_scenario(
    scenario_results: dict[str, list[dict]],
    out_dir: Path,
    windows: list[float],
    filter_method: str | None = None,
) -> None:
    """
    Four-panel cross-scenario summary that supports window-size and method selection.

      A/B  Heatmaps: attack alert_groups available in the train set for each
           (scenario × window) cell, one panel per method. Red border = split_ok=False
           (single-class split). Green = healthy, red/yellow = too few attack examples.

      C    Line plot: % attack in the train set per scenario (fixed_window).
           Flat low lines reveal structurally late-campaign scenarios where window
           size cannot improve the class balance.

      D    Line plot: mean attack / mean benign alert_group size ratio (log scale,
           fixed_window). A high ratio means the classifier can trivially use size
           rather than alert-type patterns. Higher window → higher ratio.
    """
    scenarios = sorted(scenario_results.keys())
    n_scen = len(scenarios)
    n_win = len(windows)

    # Build (scenario × window) matrices for each method
    matrices: dict[str, np.ndarray] = {}
    split_ok_matrices: dict[str, np.ndarray] = {}
    for m in ("fixed_window", "fixed_window_host"):
        mat = np.zeros((n_scen, n_win))
        ok_mat = np.ones((n_scen, n_win), dtype=bool)
        for i, scen in enumerate(scenarios):
            rows = [r for r in scenario_results[scen] if r["method"] == m]
            rows.sort(key=lambda r: r["window_val"])
            for j, w in enumerate(windows):
                row = next((r for r in rows if r["window_val"] == w), None)
                if row:
                    mat[i, j] = row["n_attack_train"]
                    ok_mat[i, j] = row["split_ok"]
        matrices[m] = mat
        split_ok_matrices[m] = ok_mat

    vmax = max(m.max() for m in matrices.values())

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Cross-scenario summary — fixed-window sweep", fontsize=13)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    # ── A & B: heatmaps ───────────────────────────────────────────────────────
    for ax_col, (method, panel_label) in enumerate(
        [("fixed_window", "A"), ("fixed_window_host", "B")]
    ):
        ax = fig.add_subplot(gs[0, ax_col])
        mat = matrices[method]
        ok_mat = split_ok_matrices[method]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(vmax, 1))
        for i in range(n_scen):
            for j in range(n_win):
                val = int(mat[i, j])
                text_color = "white" if mat[i, j] > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
                if not ok_mat[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none",
                            zorder=5,
                        )
                    )

        ax.set_xticks(range(n_win))
        ax.set_xticklabels([f"{w:g}s" for w in windows])
        ax.set_yticks(range(n_scen))
        ax.set_yticklabels(scenarios)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attack tx in train")
        ax.set_title(
            f"({panel_label}) Attack tx in train — {_LBL_METHOD[method]}", fontsize=10
        )
        ax.set_xlabel("Window size")

    colors = _SCENARIO_PALETTE[:n_scen]

    # ── C: train attack % per scenario (fixed_window) ─────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for i, scen in enumerate(scenarios):
        rows = [r for r in scenario_results[scen] if r["method"] == "fixed_window"]
        rows.sort(key=lambda r: r["window_val"])
        ws = [r["window_val"] for r in rows]
        pcts = [r["pct_attack_train"] for r in rows]
        ax_c.plot(
            ws,
            pcts,
            marker="o",
            markersize=5,
            linewidth=1.5,
            color=colors[i],
            label=scen,
        )
    _set_window_xticks(ax_c, windows)
    ax_c.set_xlabel("Window size (s)")
    ax_c.set_ylabel("Attack % in train set")
    ax_c.set_title("(C) Train attack % by scenario  [fixed_window]")
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.grid(alpha=0.3, linewidth=0.5)

    # ── D: attack / benign size ratio per scenario (fixed_window) ─────────────
    ax_d = fig.add_subplot(gs[1, 1])
    for i, scen in enumerate(scenarios):
        rows = [r for r in scenario_results[scen] if r["method"] == "fixed_window"]
        rows.sort(key=lambda r: r["window_val"])
        ws = [r["window_val"] for r in rows]
        ratios = [
            r["mean_attack_alerts"] / max(r["mean_benign_alerts"], 1.0) for r in rows
        ]
        ax_d.plot(
            ws,
            ratios,
            marker="o",
            markersize=5,
            linewidth=1.5,
            color=colors[i],
            label=scen,
        )
    _set_window_xticks(ax_d, windows)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Window size (s)")
    ax_d.set_ylabel("Mean alerts/tx (attack)  ÷  mean alerts/tx (benign)  [log]")
    ax_d.set_title("(D) Alert-count ratio: attack tx vs benign tx  [fixed_window]")
    ax_d.legend(fontsize=7, ncol=2)
    ax_d.grid(alpha=0.3, linewidth=0.5, which="both")

    fig.text(
        0.99,
        0.01,
        f"data: filtered ({filter_method})" if filter_method else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path = out_dir / "cross_scenario_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCross-scenario plots saved → {out_path}")


def plot_scenario_window_overview(
    scenario_results: dict[str, list[dict]],
    out_dir: Path,
    windows: list[float],
    filter_method: str | None = None,
) -> None:
    """
    Matplotlib table — one row per (scenario, window), grouped by scenario.

    Columns: Scenario | Window | [per method: Total, Benign, Attack, Att%]
    Attack count = n_attack + n_mixed (all non-benign alert_groups).
    """
    scenarios = sorted(scenario_results.keys())
    methods = ["fixed_window", "fixed_window_host"]
    method_labels = [_LBL_METHOD[m] for m in methods]

    col_labels = ["Scenario", "Window"]
    for ml in method_labels:
        col_labels += [f"{ml}\nTotal", f"{ml}\nBenign", f"{ml}\nAttack", f"{ml}\nAtt%"]

    sc_bg = ["#EEF3FF", "#FFF8EE"]
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for i, sc in enumerate(scenarios):
        rows_for_sc = scenario_results[sc]
        bg = sc_bg[i % 2]
        first = True
        for w in sorted(windows):
            row: list[str] = [sc if first else "", f"{w:g}s"]
            first = False
            for m in methods:
                result = next(
                    (
                        r
                        for r in rows_for_sc
                        if r["method"] == m and r["window_val"] == w
                    ),
                    None,
                )
                if result:
                    n_tx = result["n_tx"]
                    n_benign = result["n_benign"]
                    n_att = result["n_attack"] + result["n_mixed"]
                    att_pct = f"{100 * n_att / n_tx:.1f}%" if n_tx else "—"
                    row += [f"{n_tx:,}", f"{n_benign:,}", f"{n_att:,}", att_pct]
                else:
                    row += ["—", "—", "—", "—"]
            cell_text.append(row)
            cell_colors.append([bg] * len(col_labels))

    n_rows = len(cell_text)
    n_cols = len(col_labels)
    fig_height = max(4, 0.32 * (n_rows + 2))
    fig_width = max(10, 1.5 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)

    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2D2D2D")
        tbl[0, j].set_text_props(fontweight="bold", color="white")

    ax.set_title("AlertGroup counts by scenario and window size", fontsize=12, pad=10)
    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        f"data: filtered ({filter_method})" if filter_method else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path = out_dir / "scenario_window_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scenario/window overview saved → {out_path}")


def plot_metrics_heatmap(
    scenario_results: dict[str, list[dict]],
    out_dir: Path,
    windows: list[float],
    method: str = "fixed_window",
    filter_method: str | None = None,
) -> None:
    """
    2×3 grid of (scenario × window) heatmaps — one metric per panel.

      A  Attack %             (attack+mixed)/n_tx — class imbalance
      B  Purity %             (benign+attack)/n_tx — no mixed-label tx
      C  Mean alerts/tx       alert_group richness
      D  Single-alert tx %    degenerate alert_groups (window too small?)
      E  Recurring profiles % same alert type fires multiple times (window too large?)
      F  Split viable         temporal 70/30 split has both classes in train+test
    """
    scenarios = sorted(scenario_results.keys())
    n_scen = len(scenarios)
    n_win = len(windows)
    win_labels = [f"{w:g}s" for w in windows]

    keys = [
        "att_pct",
        "purity_pct",
        "mean_alerts",
        "pct_single_alert",
        "pct_profile_repeated",
        "split_ok",
    ]
    mats: dict[str, np.ndarray] = {k: np.full((n_scen, n_win), np.nan) for k in keys}

    for i, sc in enumerate(scenarios):
        rows = [r for r in scenario_results[sc] if r["method"] == method]
        for j, w in enumerate(windows):
            r = next((x for x in rows if x["window_val"] == w), None)
            if r is None:
                continue
            n_tx = r["n_tx"]
            n_att = r["n_attack"] + r["n_mixed"]
            mats["att_pct"][i, j] = 100 * n_att / n_tx if n_tx else 0.0
            mats["purity_pct"][i, j] = r["purity_pct"]
            mats["mean_alerts"][i, j] = r["mean_alerts"]
            mats["pct_single_alert"][i, j] = r["pct_single_alert"]
            mats["pct_profile_repeated"][i, j] = r["pct_profile_repeated"]
            mats["split_ok"][i, j] = float(r["split_ok"])

    # (key, title, cmap, vmin, vmax, cell_fmt)
    panels = [
        (
            "att_pct",
            "(A) Attack %\n(attack+mixed) / total",
            "RdYlGn",
            0,
            100,
            lambda v: f"{v:.1f}%",
        ),
        (
            "purity_pct",
            "(B) Purity %\npure-label alert_groups",
            "RdYlGn",
            0,
            100,
            lambda v: f"{v:.1f}%",
        ),
        (
            "mean_alerts",
            "(C) Mean alerts / tx\nalert_group richness",
            "Blues",
            None,
            None,
            lambda v: f"{v:.1f}",
        ),
        (
            "pct_single_alert",
            "(D) Single-alert tx %\n(window too small?)",
            "RdYlGn_r",
            0,
            100,
            lambda v: f"{v:.1f}%",
        ),
        (
            "pct_profile_repeated",
            "(E) Recurring profiles %\n(window too large?)",
            "RdYlGn_r",
            0,
            100,
            lambda v: f"{v:.1f}%",
        ),
        (
            "split_ok",
            "(F) Train/test split viable\n(both classes in train & test)",
            "RdYlGn",
            0,
            1,
            lambda v: "OK" if v > 0.5 else "WARN",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(
        f"Cross-scenario metric overview — {_LBL_METHOD.get(method, method)}",
        fontsize=13,
    )

    for ax, (key, title, cmap_name, vmin, vmax, fmt_fn) in zip(axes.flat, panels):
        mat = mats[key]
        _vmin = vmin if vmin is not None else float(np.nanmin(mat))
        _vmax = vmax if vmax is not None else float(np.nanmax(mat))
        if _vmin == _vmax:
            _vmax = _vmin + 1

        im = ax.imshow(mat, aspect="auto", cmap=cmap_name, vmin=_vmin, vmax=_vmax)
        cmap_obj = plt.get_cmap(cmap_name)

        for i in range(n_scen):
            for j in range(n_win):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                norm_v = np.clip((v - _vmin) / max(_vmax - _vmin, 1e-9), 0, 1)
                rgba = cmap_obj(norm_v)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.45 else "white"
                ax.text(
                    j,
                    i,
                    fmt_fn(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                )

        if key != "split_ok":
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_win))
        ax.set_xticklabels(win_labels, fontsize=8)
        ax.set_yticks(range(n_scen))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Window size", fontsize=8)

    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        f"data: filtered ({filter_method})" if filter_method else "data: raw",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out_path = out_dir / f"metrics_heatmap_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics heatmap ({method}) saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_method(
    label: str,
    method_fn,
    alerts: list[TokenizedAlert],
    window_s: float,
    train_frac: float = 0.7,
) -> dict[str, Any]:
    # Pass float directly; Python's // operator handles floats fine (ts // 0.5 works).
    records = method_fn(alerts, window_size=window_s)
    snapshots = build_snapshots_in_memory(alerts, records)
    stats = compute_stats(snapshots)
    split_stats = compute_split_stats(snapshots, train_frac)
    return {
        "method": label,
        "method_label": label,
        "window_s": f"{window_s:g}s",
        "window_val": window_s,
        **stats,
        **split_stats,
    }


def _discover_scenarios(filter_method: str | None) -> list[str]:
    if filter_method:
        search_dir = _BALANCED_ALERTS_DIR / filter_method
        if not search_dir.exists():
            return []
        return sorted(
            p.stem.removesuffix("_alerts") for p in search_dir.glob("*_alerts.json")
        )
    return sorted(
        p.stem.removesuffix("_alerts") for p in _RAW_ALERTS_DIR.glob("*_alerts.txt")
    )


def run_scenario(
    scenario: str,
    windows: list[float],
    train_frac: float,
    filter_method: str | None,
    out_dir: Path,
    run_ts: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    with _Tee(out_dir / "sweep.log"):
        return _run_scenario_inner(
            scenario, windows, train_frac, filter_method, out_dir, run_ts
        )


def _run_scenario_inner(
    scenario: str,
    windows: list[float],
    train_frac: float,
    filter_method: str | None,
    out_dir: Path,
    run_ts: str,
) -> list[dict]:
    print(
        f"\nFixed-window sweep: scenario='{scenario}'  windows={windows}  "
        f"train_frac={train_frac}  filter={filter_method or 'none'}"
    )
    print(f"Output: {out_dir}")

    print(f"\nLoading and tokenizing alerts for '{scenario}'...")
    alerts = load_and_tokenize(scenario, filter_method=filter_method)
    print(f"  Tokenized {len(alerts)} alerts.")

    methods = [
        ("fixed_window", group_alerts_fixed_window),
        ("fixed_window_host", group_alerts_fixed_window_host),
    ]

    results: list[dict] = []
    all_data: dict[str, Any] = {
        "scenario": scenario,
        "windows": windows,
        "train_frac": train_frac,
        "rows": [],
    }

    for method_name, method_fn in methods:
        print(f"\n[{method_name}]")
        for w in windows:
            print(f"  window={w:g}s ...", end=" ", flush=True)
            row = _run_method(method_name, method_fn, alerts, w, train_frac=train_frac)
            results.append(row)
            all_data["rows"].append(row)
            split_status = "ok" if row["split_ok"] else "WARN(single-class)"
            print(
                f"n_tx={row['n_tx']}  purity={row['purity_pct']:.1f}%  "
                f"mean_alerts={row['mean_alerts']:.1f}  "
                f"pct_recur={row['pct_profile_repeated']:.1f}%  "
                f"split={split_status}"
            )

    print("\n" + "=" * 84)
    print(f" Fixed-window sweep results — {scenario}")
    print("=" * 84)
    print_all_tables(results, train_frac=train_frac)
    print()

    json_path = out_dir / f"sweep_{run_ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved → {json_path}")

    print("\nGenerating plots...")
    plot_sweep(results, scenario, out_dir, filter_method=filter_method)
    plot_split_quality(
        results, scenario, out_dir, train_frac=train_frac, filter_method=filter_method
    )

    print(f"\nLog saved → {out_dir / 'sweep.log'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        help="Scenario name (e.g. 'fox'). Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_scenarios",
        help="Run for every scenario found in the alerts source directory.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=_DEFAULT_WINDOWS,
        metavar="W",
        help=f"Window sizes in seconds (default: {_DEFAULT_WINDOWS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_EXPERIMENTS_DIR,
        help="Output directory for saved results",
    )
    parser.add_argument(
        "--balanced",
        metavar="METHOD",
        dest="filtered",
        default=None,
        help=(
            "Load balanced alerts from artifacts/alerts/balanced/<METHOD>/<scenario>_alerts.json "
            "instead of raw alerts from data/alerts_csv/. E.g. --balanced naive50"
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        metavar="F",
        help="Fraction of alert_groups (sorted by time) used as the train split (default: 0.7).",
    )
    args = parser.parse_args()

    if args.all_scenarios and args.scenario:
        parser.error("Provide either a scenario name or --all, not both.")
    if not args.all_scenarios and not args.scenario:
        parser.error("Provide a scenario name or use --all.")

    windows: list[float] = sorted(args.windows)
    train_frac: float = args.train_frac
    filter_method: str | None = args.filtered
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.all_scenarios:
        scenarios = _discover_scenarios(filter_method)
        if not scenarios:
            search_dir = (
                _BALANCED_ALERTS_DIR / filter_method
                if filter_method
                else _RAW_ALERTS_DIR
            )
            print(f"No scenarios found under {search_dir}.", file=sys.stderr)
            sys.exit(1)
        print(f"Running all {len(scenarios)} scenarios: {scenarios}")
    else:
        scenarios = [args.scenario]

    all_results: dict[str, list[dict]] = {}
    sweep_dir = args.out_dir / f"sweep_{run_ts}"
    for scenario in scenarios:
        out_dir = sweep_dir / scenario
        results = run_scenario(
            scenario, windows, train_frac, filter_method, out_dir, run_ts
        )
        if results:
            all_results[scenario] = results

    if all_results:
        plot_scenario_window_overview(
            all_results, sweep_dir, windows, filter_method=filter_method
        )
        for m in ("fixed_window", "fixed_window_host"):
            plot_metrics_heatmap(
                all_results, sweep_dir, windows, method=m, filter_method=filter_method
            )

    if len(all_results) > 1:
        plot_cross_scenario(
            all_results, sweep_dir, windows, filter_method=filter_method
        )


if __name__ == "__main__":
    main()
