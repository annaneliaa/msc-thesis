"""
Sweep over fixed-window sizes for fixed_window and fixed_window_host grouping.

For each (method, window_size) combination reports:
  - Transaction counts: total, benign, attack, mixed, purity
  - Alert-size distribution: mean, p50, p95, max, % single-alert
  - Token diversity: mean/max unique tokens, unique alert types per transaction
  - Recurring alerts: % transactions with repeated alert IDs, mean repetition ratio
  - Window duration: mean and max seconds spanned by a transaction

This helps identify a suitable window size for each method. The reference value of
2 s used in earlier runs came from the Landauer et al. (2022) time-delta method,
not from a fixed-window evaluation.

Usage:
    python src/thesis/scripts/run_fixed_window_sweep.py fox \\
        [--windows 1 2 5 10 30 60] \\
        [--out-dir artifacts/experiments/run_fixed_window_sweep]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.preprocessing.group_alerts import (
    group_alerts_fixed_window,
    group_alerts_fixed_window_host,
)
from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.schemas.preprocessing import IncomingAlert, TokenizedAlert, GroupSnapshot


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_WINDOWS = [1, 2, 5, 10, 30, 60]
_PROCESSED_DATA_DIR = _REPO / "artifacts" / "processed-data"
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_fixed_window_sweep"
_BENIGN_LABEL = "false_positive"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_tokenize(scenario: str, filtered: bool = False) -> list[TokenizedAlert]:
    filename = "alerts_filtered.json" if filtered else "alerts.json"
    alerts_path = _PROCESSED_DATA_DIR / scenario / filename
    if not alerts_path.exists():
        raise FileNotFoundError(f"Processed alerts not found: {alerts_path}")

    with alerts_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

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

    return tokenized


# ---------------------------------------------------------------------------
# In-memory snapshot builder (avoids disk I/O)
# ---------------------------------------------------------------------------


def _tx_label_from_labels(alert_labels: set[str]) -> str:
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
        tx_label = (
            _tx_label_from_labels(g["alert_labels"]) if g["alert_labels"] else "benign"
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
                tx_label=tx_label,
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
    labels = [s.tx_label for s in snapshots]
    n_benign = labels.count("benign")
    n_attack = labels.count("attack")
    n_mixed = labels.count("mixed")

    n_alerts_arr = np.array([s.n_alerts for s in snapshots], dtype=float)

    # Distinct alert profiles: how many unique (token-set) combinations appear in a
    # transaction. Unlike alert_id (which ties to a specific second), this captures
    # whether the same alert TYPE recurs across different seconds in the window.
    profile_counts = np.array(
        [len({frozenset(item_set) for item_set in s.sorted_items}) for s in snapshots],
        dtype=float,
    )
    profile_ratios = n_alerts_arr / np.maximum(profile_counts, 1)
    pct_profile_repeated = float(np.mean(profile_ratios > 1.0) * 100)

    token_counts = np.array([len(s.items) for s in snapshots], dtype=float)

    # Alert-type diversity: distinct "short:<name>" tokens per transaction.
    # Each unique short: token represents a distinct detector signature type.
    short_counts = np.array(
        [len([t for t in s.items if t.startswith("short:")]) for s in snapshots],
        dtype=float,
    )

    durations = np.array([s.end_ts - s.start_ts for s in snapshots], dtype=float)

    n_single = int(np.sum(n_alerts_arr == 1))

    # Per-label size stats
    attack_sizes = [s.n_alerts for s in snapshots if s.tx_label == "attack"]
    benign_sizes = [s.n_alerts for s in snapshots if s.tx_label == "benign"]

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
        # Temporal span of transactions
        "mean_duration_s": float(durations.mean()),
        "max_duration_s": float(durations.max()),
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


def print_all_tables(results: list[dict]) -> None:
    print_table(
        "Transaction counts and purity",
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
        "Alert-size distribution — all transactions (alerts per transaction)",
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
        "Token diversity (unique tokens and alert signature types per transaction)",
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_C_METHOD = {
    "fixed_window": "#4C72B0",
    "fixed_window_host": "#55A868",
}
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
    results: list[dict], scenario: str, out_dir: Path, filtered: bool = False
) -> None:
    methods = list(dict.fromkeys(r["method"] for r in results))
    windows = sorted({r["window_val"] for r in results})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Fixed-window sweep — {scenario}", fontsize=12)

    # ── (0,0) Transaction count vs window size ─────────────────────────────
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
    ax.set_ylabel("Number of transactions")
    ax.set_title("(A) Total transaction count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)

    # ── (0,1) Mean transaction size by label ───────────────────────────────
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
    ax.set_ylabel("Mean alerts per transaction  (log scale)")
    ax.set_title("(B) Mean transaction size by label")
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
    ax.set_ylabel("Mean count per transaction")
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
    ax.set_ylabel("% transactions with recurring alert profiles", color="k")
    ax2.set_ylabel("Mean ratio (alerts / distinct profiles)", color="grey")
    ax2.tick_params(labelcolor="grey")
    ax.set_title("(D) Alert profile recurrence within window")
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig.text(
        0.99,
        0.01,
        "data: filtered" if filtered else "data: raw",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_method(
    label: str,
    method_fn,
    alerts: list[TokenizedAlert],
    window_s: float,
) -> dict[str, Any]:
    # Pass float directly; Python's // operator handles floats fine (ts // 0.5 works).
    records = method_fn(alerts, window_size=window_s)
    snapshots = build_snapshots_in_memory(alerts, records)
    stats = compute_stats(snapshots)
    return {
        "method": label,
        "method_label": label,
        "window_s": f"{window_s:g}s",
        "window_val": window_s,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Scenario name (e.g. 'fox')")
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
        "--filtered",
        action="store_true",
        help="Use detector-filtered alerts (alerts_filtered.json) instead of alerts.json.",
    )
    args = parser.parse_args()

    scenario: str = args.scenario
    windows: list[float] = sorted(args.windows)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.out_dir / f"sweep_{run_ts}" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fixed-window sweep: scenario='{scenario}'  windows={windows}")
    print(f"Output: {out_dir}")

    print(f"\nLoading and tokenizing alerts for '{scenario}'...")
    alerts = load_and_tokenize(scenario, filtered=args.filtered)
    print(f"  Tokenized {len(alerts)} alerts.")

    methods = [
        ("fixed_window", group_alerts_fixed_window),
        ("fixed_window_host", group_alerts_fixed_window_host),
    ]

    results: list[dict] = []
    all_data: dict[str, Any] = {"scenario": scenario, "windows": windows, "rows": []}

    for method_name, method_fn in methods:
        print(f"\n[{method_name}]")
        for w in windows:
            print(f"  window={w:g}s ...", end=" ", flush=True)
            row = _run_method(method_name, method_fn, alerts, w)
            results.append(row)
            all_data["rows"].append(row)
            print(
                f"n_tx={row['n_tx']}  purity={row['purity_pct']:.1f}%  "
                f"mean_alerts={row['mean_alerts']:.1f}  "
                f"pct_recur={row['pct_profile_repeated']:.1f}%"
            )

    print("\n" + "=" * 84)
    print(f" Fixed-window sweep results — {scenario}")
    print("=" * 84)
    print_all_tables(results)
    print()

    # Save JSON
    json_path = out_dir / f"sweep_{run_ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Save plots
    print("\nGenerating plots...")
    plot_sweep(results, scenario, out_dir, filtered=args.filtered)


if __name__ == "__main__":
    main()
