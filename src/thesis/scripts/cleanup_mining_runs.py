"""
Clean up accumulated mining run directories.

Usage:
    python cleanup_mining_runs.py [--keep N] [--dry-run] [--mining-dir PATH]

Actions (always previewed first unless --no-confirm is passed):
  1. Delete old runs — for each dataset+type combo, keep only the N most recent.
  2. Strip intermediate files from kept runs — removes files that are not read by
     any downstream code (prepared_transactions.csv, tidsets.json, items_with_tidsets.json).

Safe files that are NEVER removed from kept runs:
  eclat/frequent_itemsets.csv
  eclat/or_feature_itemsets.csv
  prefixspan/items/frequent_sequences.csv
  final_combined_mining_df.csv
  (and their attack/ counterparts)
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Files that are safe to delete from kept runs — write-only intermediate artifacts.
INTERMEDIATE_FILES = [
    "prepared_transactions.csv",
    "tidsets.json",
    "items_with_tidsets.json",
]

# Globs relative to the eclat/ and prefixspan/items/ sub-dirs.
# These are searched recursively so they also match attack/eclat/, etc.
INTERMEDIATE_GLOBS = [f"**/{f}" for f in INTERMEDIATE_FILES]

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")


def parse_runs(mining_dir: Path) -> dict[str, list[Path]]:
    """Return {combo_key: [run_dir, ...]} sorted oldest-first per key."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(mining_dir.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        groups[m.group(1)].append(d)
    return dict(groups)


def find_intermediates(run_dir: Path) -> list[Path]:
    return [p for glob in INTERMEDIATE_GLOBS for p in run_dir.glob(glob) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=2,
        metavar="N",
        help="Number of most-recent runs to keep per combo (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without deleting anything",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation prompt and execute immediately",
    )
    parser.add_argument(
        "--mining-dir",
        default="artifacts/mining",
        help="Path to mining artifacts directory (default: artifacts/mining)",
    )
    args = parser.parse_args()

    mining_dir = Path(args.mining_dir)
    if not mining_dir.exists():
        print(f"Mining directory not found: {mining_dir}")
        return

    groups = parse_runs(mining_dir)

    runs_to_delete: list[Path] = []
    intermediates_to_strip: list[Path] = []

    for combo, runs in sorted(groups.items()):
        keep = runs[-args.keep :]
        drop = runs[: len(runs) - args.keep]

        for r in drop:
            runs_to_delete.append(r)

        for r in keep:
            intermediates_to_strip.extend(find_intermediates(r))

    if not runs_to_delete and not intermediates_to_strip:
        print("Nothing to clean up.")
        return

    # ── preview ───────────────────────────────────────────────────────────────
    total_runs = sum(len(v) for v in groups.values())
    print(f"Mining directory : {mining_dir.resolve()}")
    print(f"Total runs found : {total_runs}")
    print(f"Keep per combo   : {args.keep}")
    print()

    if runs_to_delete:
        sizes = [
            sum(f.stat().st_size for f in r.rglob("*") if f.is_file())
            for r in runs_to_delete
        ]
        total_mb = sum(sizes) / 1_048_576
        print(f"Runs to DELETE ({len(runs_to_delete)}, ~{total_mb:.0f} MB):")
        for r, sz in zip(runs_to_delete, sizes):
            print(f"  {r.name}  ({sz / 1_048_576:.1f} MB)")
        print()

    if intermediates_to_strip:
        strip_mb = sum(p.stat().st_size for p in intermediates_to_strip) / 1_048_576
        print(
            f"Intermediate files to STRIP from kept runs ({len(intermediates_to_strip)}, ~{strip_mb:.0f} MB):"
        )
        for p in intermediates_to_strip:
            print(f"  {p.relative_to(mining_dir)}")
        print()

    if args.dry_run:
        print("DRY RUN — no files were deleted.")
        return

    if not args.no_confirm:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # ── execute ───────────────────────────────────────────────────────────────
    for r in runs_to_delete:
        shutil.rmtree(r)
        print(f"Deleted {r.name}")

    for p in intermediates_to_strip:
        p.unlink()
        print(f"Stripped {p.relative_to(mining_dir)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
