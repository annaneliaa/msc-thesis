"""
Mine (or reuse an already-mined) attribute symbolic schema for one or more
scenarios, without training any models.

run_model_comparison_attribute.py mines the same attribute schema once per
scenario and reuses it for every model in the comparison (logreg/rf/mlp) --
but that in-memory reuse only lasts for a single invocation of the script.
This script isolates just the mining step (ingest -> attribute mining ->
schema registration) behind the same persistent, on-disk cache
(thesis.mining.attribute_schema_cache) that run_symbolic_experiment now
checks automatically, so:

  - Re-running run_model_comparison_attribute.py (e.g. after a crash, or
    with a different --models subset) skips mining entirely when the mining
    inputs (alert_groups file + attribute-mining thresholds) haven't changed.
  - You can pre-mine schemas for a batch of scenarios up front, independent
    of any comparison run, e.g. before kicking off several comparisons that
    all share the same mining config.

Usage:
  python src/thesis/scripts/mining/mine_attribute_schema.py fox wheeler harrison
  python src/thesis/scripts/mining/mine_attribute_schema.py --all --filtered naive50
  python src/thesis/scripts/mining/mine_attribute_schema.py cscas --train-frac 0.1 --test-frac 0.9
  python src/thesis/scripts/mining/mine_attribute_schema.py fox --force  # ignore cache, re-mine

--mine-frac / --random-split / --random-seed / attribute-mining thresholds
(--min-attack-coverage etc.) behave exactly as in
run_model_comparison_attribute.py, and must match between this script and a
later comparison run for the cache to be reused (they're part of the cache
key -- see thesis.mining.attribute_schema_cache.compute_fingerprint).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.mining.attribute_schema_cache import mine_or_reuse_attribute_schema
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import (
    ensure_feature_manifest,
    ingest_ait_scenario,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
    resolve_mining_alert_groups_path,
)
from thesis.schemas.mining import AttributeMiningConfig
from thesis.visualization.eda import SCENARIOS as ALL_SCENARIOS

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))


def mine_scenario(
    scenario: str,
    attribute_mining_config: AttributeMiningConfig,
    filtered: bool,
    method: str | None,
    window_size: int,
    mine_frac: float,
    random_split: bool,
    random_seed: int,
    force: bool,
) -> Path:
    """Ingest `scenario`'s alert_groups if needed, resolve the mining window,
    and mine (or reuse a cached) attribute schema. Returns the schema path."""
    is_cscas = dataset_for_scenario(scenario) == "cscas"
    window_tag = f"_w{window_size}" if window_size != 2 else ""
    if is_cscas:
        method_tag = CSCAS_PREGROUPED_METHOD
        grouping = GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
    elif filtered:
        method_tag = (
            f"filtered_{method}{window_tag}" if method else f"filtered{window_tag}"
        )
        grouping = GroupingConfig(window_size=window_size)
    elif window_tag:
        method_tag = f"w{window_size}"
        grouping = GroupingConfig(window_size=window_size)
    else:
        method_tag = "fixed_window"
        grouping = GroupingConfig(window_size=window_size)

    cache_dir = CACHE_DIR / scenario / "groups" / method_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )
    alerts_path = (
        _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
        if filtered and not is_cscas
        else None
    )

    print(f"\n{'=' * 60}\n  {scenario} (attribute mining)\n{'=' * 60}")
    if is_cscas:
        print("[1-2] Ingesting CSCAS scenario...")
        ingest_cscas_scenario(cache_dir=cache_dir)
    else:
        print("[1-2] Ingesting scenario...")
        ingest_ait_scenario(
            scenario,
            alerts_json_path=alerts_path,
            cache_dir=cache_dir,
            grouping=grouping,
        )

    print("[3] Checking feature manifest...")
    ensure_feature_manifest(scenario)

    print("[4] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(scenario, cache_dir)
    alert_groups_path = cache_dir / "alert_groups" / "alert_groups_raw.json"
    alert_groups.sort(key=lambda t: t.start_ts or "")

    if random_split:
        import random as _random

        rng = _random.Random(random_seed)
        rng.shuffle(alert_groups)
        print(
            f"  [random-split] Shuffled {len(alert_groups)} alert_groups (seed={random_seed})"
        )

    mining_alert_groups_path = resolve_mining_alert_groups_path(
        alert_groups,
        alert_groups_path,
        mine_frac=mine_frac,
        random_split=random_split,
        random_seed=random_seed,
    )
    if mine_frac < 1.0:
        n_mine = int(mine_frac * len(alert_groups))
        split_label = "random" if random_split else "first"
        print(
            f"  Mining on {split_label} {n_mine}/{len(alert_groups)} alert_groups "
            f"(mine_frac={mine_frac:.2f})"
        )

    print("[5] Mining attribute schema (or reusing cached one)...")
    schema_path, _run_dir, mining_stats = mine_or_reuse_attribute_schema(
        scenario=scenario,
        alert_groups_path=mining_alert_groups_path,
        run_name=f"symbolic_{scenario}",
        attribute_mining_config=attribute_mining_config,
        force=force,
    )
    status = "cache hit" if mining_stats.get("cache_hit") else "mined fresh"
    print(f"  [{scenario}] {status} → {schema_path}")
    return schema_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine (or reuse a cached) attribute symbolic schema per scenario, "
            "without training any models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenarios", nargs="*", help="Scenario names (e.g. fox wheeler)."
    )
    parser.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help=f"Mine all scenarios: {', '.join(ALL_SCENARIOS)}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any cached schema and re-mine.",
    )
    parser.add_argument(
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help="Use filtered alerts. Optionally pass a balancing method (e.g. naive50).",
    )
    parser.add_argument("--window-size", type=int, default=2, metavar="W")
    parser.add_argument("--mine-frac", type=float, default=1.0, dest="mine_frac")
    parser.add_argument("--random-split", action="store_true", dest="random_split")
    parser.add_argument("--random-seed", type=int, default=42, dest="random_seed")
    parser.add_argument("--min-attack-coverage", type=float, default=0.05)
    parser.add_argument("--min-benign-coverage", type=float, default=0.05)
    parser.add_argument("--min-growth-rate", type=float, default=3.0)
    parser.add_argument("--max-p-value", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--class-weight", type=str, default="balanced")
    args = parser.parse_args()

    if args.all_scenarios:
        args.scenarios = list(ALL_SCENARIOS)
    elif not args.scenarios:
        parser.error("Specify at least one scenario name or use --all.")

    attribute_mining_config = AttributeMiningConfig()
    attribute_mining_config.contrast.min_attack_coverage = args.min_attack_coverage
    attribute_mining_config.contrast.min_benign_coverage = args.min_benign_coverage
    attribute_mining_config.contrast.min_growth_rate = args.min_growth_rate
    attribute_mining_config.contrast.max_p_value = args.max_p_value
    attribute_mining_config.tree.max_depth = args.max_depth
    attribute_mining_config.tree.min_samples_leaf = args.min_samples_leaf
    attribute_mining_config.tree.class_weight = args.class_weight

    method = args.filtered if args.filtered else None
    filtered = args.filtered is not None

    results: dict[str, Path | None] = {}
    for scenario in args.scenarios:
        try:
            results[scenario] = mine_scenario(
                scenario=scenario,
                attribute_mining_config=attribute_mining_config,
                filtered=filtered,
                method=method,
                window_size=args.window_size,
                mine_frac=args.mine_frac,
                random_split=args.random_split,
                random_seed=args.random_seed,
                force=args.force,
            )
        except Exception as exc:
            print(f"\n[{scenario}] FAILED: {exc}")
            traceback.print_exc()
            results[scenario] = None

    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    for scenario, path in results.items():
        print(f"  {scenario:<20} {path if path else 'FAILED'}")


if __name__ == "__main__":
    main()
