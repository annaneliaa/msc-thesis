"""
Inspect a single mining run and produce plots + tables.

Usage:
    python inspect_mining_run.py <run_dir> [--out <output_dir>] [--diff-threshold 0.05]

<run_dir> is a directory under artifacts/mining/, e.g.:
    artifacts/mining/20260624_133945_symbolic_wheeler

Output (saved to <run_dir>/inspection/ by default):
    01_support_scatter.png     — support_benign vs support_attack, coloured by fate
    02_support_diff_hist.png   — distribution of support_diff per mining type
    03_feature_counts.png      — benign-leaning / attack-leaning / neutral breakdown
    04_token_frequency.png     — top alert tokens appearing in kept features
    tables.txt                 — top features + pipeline funnel, printed to stdout and saved
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── colour palette ────────────────────────────────────────────────────────────
C_BENIGN = "#2166ac"
C_ATTACK = "#d6604d"
C_NEUTRAL = "#aaaaaa"
C_OR = "#8e44ad"
ALPHA_DENSE = 0.35


# ── data loading ──────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # itemset_str is generated before mail-host abstraction in the mining job and may be stale.
    # Regenerate it from the (already abstracted) itemset column.
    if "itemset" in df.columns and "itemset_str" in df.columns:
        import ast as _ast

        def _refresh(raw):
            try:
                parsed = _ast.literal_eval(str(raw))
                if isinstance(parsed, (tuple, list, frozenset, set)):
                    return " | ".join(str(x) for x in parsed)
            except Exception:
                pass
            return raw

        df["itemset_str"] = df["itemset"].apply(_refresh)
    return df


def _classify(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add a 'fate' column based on support_diff."""
    if df.empty or "support_diff" not in df.columns:
        return df
    df = df.copy()
    df["fate"] = "neutral"
    df.loc[df["support_diff"] >= threshold, "fate"] = "benign-leaning"
    df.loc[df["support_diff"] <= -threshold, "fate"] = "attack-leaning"
    return df


def load_run(run_dir: Path, threshold: float) -> dict:
    """Load all raw artifacts and the final schema from a run directory."""
    raw: dict[str, pd.DataFrame] = {}

    # Benign pass artifacts
    raw["eclat"] = _classify(
        _load_csv(run_dir / "eclat" / "frequent_itemsets.csv"), threshold
    )
    raw["seq"] = _classify(
        _load_csv(run_dir / "prefixspan" / "items" / "frequent_sequences.csv"),
        threshold,
    )
    raw["or"] = _classify(
        _load_csv(run_dir / "eclat" / "or_feature_itemsets.csv"), threshold
    )

    # Attack pass artifacts (new runs only)
    raw["eclat_attack"] = _classify(
        _load_csv(run_dir / "attack" / "eclat" / "frequent_itemsets.csv"), threshold
    )
    raw["seq_attack"] = _classify(
        _load_csv(
            run_dir / "attack" / "prefixspan" / "items" / "frequent_sequences.csv"
        ),
        threshold,
    )
    raw["or_attack"] = _classify(
        _load_csv(run_dir / "attack" / "eclat" / "or_feature_itemsets.csv"), threshold
    )

    # Final schema
    raw["final"] = _load_csv(run_dir / "final_combined_mining_df.csv")

    return raw


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_tokens(itemset_str: str) -> list[str]:
    """Extract individual alert tokens from a frozenset/tuple string."""
    try:
        parsed = ast.literal_eval(itemset_str)
        if isinstance(parsed, (tuple, list, set, frozenset)):
            return [str(x).split(":")[-1] for x in parsed]
    except Exception:
        pass
    return [t.strip() for t in str(itemset_str).split("|") if t.strip()]


def _token_counts(df: pd.DataFrame, col: str = "itemset") -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=int)
    tokens = df[col].dropna().apply(_parse_tokens).explode()
    return tokens.value_counts()


# ── Figure 1: support scatter ─────────────────────────────────────────────────


def plot_support_scatter(raw: dict, threshold: float, out: Path) -> None:
    sources = [
        ("eclat", "ECLAT itemsets (benign pass)", False),
        ("seq", "PrefixSpan sequences (benign pass)", True),
        ("eclat_attack", "ECLAT itemsets (attack pass)", False),
        ("seq_attack", "PrefixSpan sequences (attack pass)", True),
        ("or", "OR patterns (benign pass)", False),
    ]
    sources = [(k, label, dense) for k, label, dense in sources if not raw[k].empty]
    if not sources:
        return

    ncols = min(3, len(sources))
    nrows = (len(sources) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    fate_colors = {
        "benign-leaning": C_BENIGN,
        "attack-leaning": C_ATTACK,
        "neutral": C_NEUTRAL,
    }
    fate_labels = {
        "benign-leaning": f"Benign-leaning (Δ≥+{threshold})",
        "attack-leaning": f"Attack-leaning (Δ≤−{threshold})",
        "neutral": f"Neutral (|Δ|<{threshold})",
    }

    for idx, (key, label, use_density) in enumerate(sources):
        ax = axes[idx // ncols][idx % ncols]
        df = raw[key]
        if "support_benign" not in df.columns or "support_attack" not in df.columns:
            ax.set_visible(False)
            continue

        if use_density and len(df) > 2000:
            # hexbin for dense sequence data
            hb = ax.hexbin(
                df["support_benign"],
                df["support_attack"],
                gridsize=40,
                cmap="Greys",
                mincnt=1,
                linewidths=0.1,
            )
            fig.colorbar(hb, ax=ax, label="count")
            # overlay kept patterns as coloured dots
            for fate in ["benign-leaning", "attack-leaning"]:
                sub = df[df["fate"] == fate]
                if not sub.empty:
                    ax.scatter(
                        sub["support_benign"],
                        sub["support_attack"],
                        c=fate_colors[fate],
                        s=12,
                        alpha=0.7,
                        zorder=3,
                        label=fate_labels[fate],
                    )
        else:
            for fate in ["neutral", "attack-leaning", "benign-leaning"]:
                sub = (
                    df[df.get("fate", pd.Series(["neutral"] * len(df))) == fate]
                    if "fate" in df
                    else df
                )
                if not sub.empty:
                    ax.scatter(
                        sub["support_benign"],
                        sub["support_attack"],
                        c=fate_colors.get(fate, C_NEUTRAL),
                        s=18 if fate != "neutral" else 10,
                        alpha=0.55 if fate == "neutral" else 0.8,
                        zorder=2 if fate == "neutral" else 3,
                        label=fate_labels.get(fate, fate),
                    )

        lim = max(df["support_benign"].max(), df["support_attack"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="y = x")
        ax.plot(
            [0, lim],
            [threshold, lim + threshold],
            "--",
            lw=0.8,
            color=C_BENIGN,
            alpha=0.5,
        )
        ax.plot(
            [threshold, lim + threshold],
            [0, lim],
            "--",
            lw=0.8,
            color=C_ATTACK,
            alpha=0.5,
        )

        ax.set_xlabel("support_benign")
        ax.set_ylabel("support_attack")
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for idx in range(len(sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Support scatter: benign vs attack (per mining type)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "01_support_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / '01_support_scatter.png'}")


# ── Figure 2: support_diff histogram ─────────────────────────────────────────


def plot_support_diff_hist(raw: dict, threshold: float, out: Path) -> None:
    panels = [
        ("eclat", "eclat_attack", "ECLAT itemsets"),
        ("seq", "seq_attack", "PrefixSpan sequences"),
        ("or", "or_attack", "OR patterns"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (benign_key, attack_key, label) in zip(axes, panels):
        plotted = False
        for key, colour, pass_label in [
            (benign_key, C_BENIGN, "benign pass"),
            (attack_key, C_ATTACK, "attack pass"),
        ]:
            df = raw.get(key, pd.DataFrame())
            if df.empty or "support_diff" not in df.columns:
                continue
            ax.hist(
                df["support_diff"],
                bins=50,
                color=colour,
                alpha=0.55,
                label=f"{pass_label} (n={len(df):,})",
                edgecolor="none",
            )
            plotted = True

        if plotted:
            ax.axvline(
                threshold,
                color=C_BENIGN,
                lw=1.2,
                linestyle="--",
                label=f"+{threshold} (benign threshold)",
            )
            ax.axvline(
                -threshold,
                color=C_ATTACK,
                lw=1.2,
                linestyle="--",
                label=f"−{threshold} (attack threshold)",
            )
            ax.axvline(0, color="black", lw=0.7, alpha=0.4)
            ax.set_xlabel("support_diff")
            ax.set_ylabel("count")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle(
        "Distribution of support_diff (target − other)", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out / "02_support_diff_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / '02_support_diff_hist.png'}")


# ── Figure 3: direction filter yield (100% stacked horizontal bar) ───────────


def plot_feature_counts(raw: dict, threshold: float, out: Path) -> None:
    """
    100% stacked horizontal bar chart — one bar per mining pass + type.

    Each bar shows what fraction of the raw mined patterns fall into each
    direction bucket:
      - Benign-leaning  (support_diff >= +threshold): kept as benign features
      - Attack-leaning  (support_diff <= -threshold): kept as attack features
      - Neutral         (|support_diff| < threshold):  discarded — too common in both classes

    This makes ECLAT and PrefixSpan visually comparable despite their very
    different absolute counts (76k vs ~500).  Absolute totals are annotated
    at the right edge of each bar.
    """
    rows = []
    for key, mtype, pass_label in [
        ("eclat", "ECLAT itemsets", "Benign pass"),
        ("seq", "PrefixSpan sequences", "Benign pass"),
        ("or", "OR patterns", "Benign pass"),
        ("eclat_attack", "ECLAT itemsets", "Attack pass"),
        ("seq_attack", "PrefixSpan sequences", "Attack pass"),
        ("or_attack", "OR patterns", "Attack pass"),
    ]:
        df = raw.get(key, pd.DataFrame())
        if df.empty or "support_diff" not in df.columns:
            continue
        total = len(df)
        n_benign = int((df["support_diff"] >= threshold).sum())
        n_attack = int((df["support_diff"] <= -threshold).sum())
        n_neutral = total - n_benign - n_attack
        rows.append(
            {
                "label": f"{pass_label} — {mtype}",
                "pct_benign": 100 * n_benign / total,
                "pct_neutral": 100 * n_neutral / total,
                "pct_attack": 100 * n_attack / total,
                "n_benign": n_benign,
                "n_neutral": n_neutral,
                "n_attack": n_attack,
                "total": total,
            }
        )

    if not rows:
        return

    counts = pd.DataFrame(rows)
    n = len(counts)
    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.7 + 1.5)))

    y = np.arange(n)
    # bar_h = 0.55

    # Draw stacked segments: benign | neutral | attack
    lefts = np.zeros(n)
    for pct_col, color, label in [
        ("pct_benign", C_BENIGN, f"Benign-leaning (Δ ≥ +{threshold})"),
        ("pct_neutral", C_NEUTRAL, f"Neutral — discarded (|Δ| < {threshold})"),
        ("pct_attack", C_ATTACK, f"Attack-leaning (Δ ≤ −{threshold})"),
    ]:
        vals = counts[pct_col].to_numpy()
        # bars = ax.barh(
        #     y, vals, left=lefts, height=bar_h, color=color, alpha=0.85, label=label
        # )
        # Label segments that are wide enough to annotate
        for i, (val, left) in enumerate(zip(vals, lefts)):
            if val >= 5:
                ax.text(
                    left + val / 2,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if color != C_NEUTRAL else "#444",
                )
        lefts = lefts + vals

    # Annotate total count at the right edge
    for i, row in counts.iterrows():
        ax.text(101, i, f"n={row['total']:,}", va="center", fontsize=7.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(counts["label"], fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Percentage of mined patterns")
    passes_present = []
    if any(r["label"].startswith("Benign") for r in rows):
        passes_present.append("benign")
    if any(r["label"].startswith("Attack") for r in rows):
        passes_present.append("attack")
    pass_str = "/".join(passes_present)
    ax.set_title(
        f"Direction breakdown of mined patterns for {pass_str} pass: what fraction is kept vs discarded",
        fontweight="bold",
        fontsize=10,
    )
    ax.axvline(100, color="#aaa", lw=0.7, linestyle="--")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out / "03_feature_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / '03_feature_counts.png'}")


# ── Figure 4: alert token frequency ──────────────────────────────────────────


def plot_token_frequency(
    raw: dict, threshold: float, out: Path, top_n: int = 25
) -> None:
    final = raw.get("final", pd.DataFrame())
    has_source_label = not final.empty and "source_label" in final.columns

    # Collect tokens from kept features
    if has_source_label:
        benign_feat = final[final["source_label"] == "benign"]
        attack_feat = final[final["source_label"] == "attack"]
        col = "itemset"
        benign_tokens = _token_counts(benign_feat, col)
        attack_tokens = _token_counts(attack_feat, col)
    else:
        # Fall back: classify raw ECLAT by support_diff
        df = raw.get("eclat", pd.DataFrame())
        if df.empty or "support_diff" not in df.columns:
            return
        benign_tokens = _token_counts(
            df[df["support_diff"] >= threshold], "itemset_str"
        )
        attack_tokens = _token_counts(
            df[df["support_diff"] <= -threshold], "itemset_str"
        )

    all_tokens = benign_tokens.add(attack_tokens, fill_value=0).sort_values(
        ascending=False
    )
    top_tokens = all_tokens.head(top_n).index.tolist()

    b_counts = [benign_tokens.get(t, 0) for t in top_tokens]
    a_counts = [attack_tokens.get(t, 0) for t in top_tokens]

    colors = []
    for b, a in zip(b_counts, a_counts):
        if b > 0 and a > 0:
            colors.append("#9b59b6")  # purple = both
        elif b > 0:
            colors.append(C_BENIGN)
        else:
            colors.append(C_ATTACK)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.32)))
    y = np.arange(len(top_tokens))
    ax.barh(
        y,
        [b + a for b, a in zip(b_counts, a_counts)],
        color=colors,
        alpha=0.75,
        height=0.65,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top_tokens, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences in mined features")
    ax.set_title(f"Top-{top_n} alert tokens in schema features", fontweight="bold")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(color=C_BENIGN, label="Benign features only"),
        Patch(color=C_ATTACK, label="Attack features only"),
        Patch(color="#9b59b6", label="Both"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out / "04_token_frequency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / '04_token_frequency.png'}")


# ── Tables ────────────────────────────────────────────────────────────────────


def print_tables(raw: dict, threshold: float, n: int = 20) -> str:
    lines = []

    def section(title: str) -> None:
        lines.append("\n" + "=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    # ── Pipeline funnel ──
    section("PIPELINE FUNNEL")
    for key, label in [
        ("eclat", "ECLAT (benign)"),
        ("seq", "PrefixSpan (benign)"),
        ("or", "OR (benign)"),
        ("eclat_attack", "ECLAT (attack)"),
        ("seq_attack", "PrefixSpan (attack)"),
        ("or_attack", "OR (attack)"),
    ]:
        df = raw.get(key, pd.DataFrame())
        if df.empty or "support_diff" not in df.columns:
            continue
        n_b = int((df["support_diff"] >= threshold).sum())
        n_a = int((df["support_diff"] <= -threshold).sum())
        n_n = len(df) - n_b - n_a
        lines.append(
            f"  {label:<30}  total={len(df):>6,}  "
            f"benign-leaning={n_b:>5,}  attack-leaning={n_a:>5,}  neutral={n_n:>5,}"
        )

    final = raw.get("final", pd.DataFrame())
    if not final.empty:
        lines.append(f"\n  Final schema features:          {len(final):,}")
        for mtype, count in final["mining_type"].value_counts().items():
            lines.append(f"    {mtype:<25}  {count:,}")
        if "source_label" in final.columns:
            for label, count in final["source_label"].value_counts().items():
                lines.append(f"    source={label:<18}  {count:,}")

    # ── Top benign features ──
    section(f"TOP {n} BENIGN-LEANING ITEMSETS (by support_diff)")
    df = raw.get("eclat", pd.DataFrame())
    if not df.empty and "support_diff" in df.columns:
        top = df[df["support_diff"] >= threshold].nlargest(n, "support_diff")
        cols = ["itemset_str", "support_benign", "support_attack", "support_diff", "k"]
        cols = [c for c in cols if c in top.columns]
        lines.append(top[cols].to_string(index=False))

    # ── Top attack features ──
    section(f"TOP {n} ATTACK-LEANING ITEMSETS (by |support_diff|)")
    df = raw.get("eclat", pd.DataFrame())
    if not df.empty and "support_diff" in df.columns:
        top = df[df["support_diff"] <= -threshold].nsmallest(n, "support_diff")
        cols = ["itemset_str", "support_benign", "support_attack", "support_diff", "k"]
        cols = [c for c in cols if c in top.columns]
        lines.append(top[cols].to_string(index=False))

    if not raw.get("eclat_attack", pd.DataFrame()).empty:
        section(f"TOP {n} ATTACK-MINED ITEMSETS (attack pass, by support_diff)")
        df = raw["eclat_attack"]
        top = df[df["support_diff"] >= threshold].nlargest(n, "support_diff")
        cols = ["itemset_str", "support_attack", "support_benign", "support_diff", "k"]
        cols = [c for c in cols if c in top.columns]
        lines.append(top[cols].to_string(index=False))

    # ── Top discarded (neutral) features ──
    section(f"TOP {n} DISCARDED (NEUTRAL) ITEMSETS — most common in both classes")
    df = raw.get("eclat", pd.DataFrame())
    if not df.empty and "support_diff" in df.columns:
        neutral = df[df["support_diff"].abs() < threshold].copy()
        neutral["avg_support"] = (
            neutral["support_benign"] + neutral["support_attack"]
        ) / 2
        top = neutral.nlargest(n, "avg_support")
        cols = [
            "itemset_str",
            "support_benign",
            "support_attack",
            "support_diff",
            "avg_support",
            "k",
        ]
        cols = [c for c in cols if c in top.columns]
        lines.append(top[cols].to_string(index=False))

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a mining run directory.")
    parser.add_argument("run_dir", type=Path, help="Path to the mining run directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/inspection)",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.05,
        help="support_diff threshold for direction classification (default: 0.05)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of rows in top-feature tables (default: 20)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir: Path = args.out or run_dir / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInspecting: {run_dir}")
    print(f"Output:     {out_dir}")
    print(f"Threshold:  ±{args.diff_threshold}\n")

    raw = load_run(run_dir, args.diff_threshold)

    has_attack_pass = not raw["eclat_attack"].empty
    print(
        f"Attack pass artifacts: {'found' if has_attack_pass else 'not found (benign-only run)'}"
    )

    print("\nGenerating figures...")
    plot_support_scatter(raw, args.diff_threshold, out_dir)
    plot_support_diff_hist(raw, args.diff_threshold, out_dir)
    plot_feature_counts(raw, args.diff_threshold, out_dir)
    plot_token_frequency(raw, args.diff_threshold, out_dir)

    print("\nGenerating tables...")
    table_text = print_tables(raw, args.diff_threshold, n=args.top_n)
    print(table_text)

    tables_file = out_dir / "tables.txt"
    tables_file.write_text(table_text, encoding="utf-8")
    print(f"\n  Saved: {tables_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
