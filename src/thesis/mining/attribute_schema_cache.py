"""Persistent cache for attribute-mined symbolic schemas.

Attribute mining (contrast-set + decision-tree rules, see
attribute_mining_job.py) is deterministic given its inputs: the alert_groups
file it runs on and the AttributeMiningConfig thresholds. Every model in a
run_model_comparison_attribute.py comparison mines the same schema for a
given scenario, and separate invocations of that script (e.g. --resume after
a crash, or a rerun with a different --models subset) would otherwise
re-mine from scratch every time even though nothing about the mining inputs
changed.

This module fingerprints the mining inputs and keeps a small on-disk index
(<root_dir>/<scenario>/attribute_mining_cache.json) mapping fingerprint ->
schema path, so `mine_or_reuse_attribute_schema` can skip mining entirely
when an equivalent schema has already been produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from thesis.features.service import build_persist_and_register_symbolic_schema
from thesis.paths import FEATURE_DIR
from thesis.schemas.mining import AttributeMiningConfig


def _cache_index_path(scenario_name: str, root_dir: Path) -> Path:
    return root_dir / scenario_name / "attribute_mining_cache.json"


def _load_cache_index(scenario_name: str, root_dir: Path) -> dict:
    path = _cache_index_path(scenario_name, root_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache_index(scenario_name: str, root_dir: Path, index: dict) -> None:
    path = _cache_index_path(scenario_name, root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def compute_fingerprint(
    alert_groups_path: Path,
    attribute_mining_config: AttributeMiningConfig,
) -> str:
    """Fingerprint the inputs that fully determine an attribute-mined schema:
    the alert_groups file being mined -- identified by path + size + mtime,
    so a regenerated file invalidates the cache without hashing its full
    content -- and the mining thresholds (contrast-set + decision-tree)."""
    stat = alert_groups_path.stat()
    payload = {
        "alert_groups_path": str(alert_groups_path.resolve()),
        "alert_groups_size": stat.st_size,
        "alert_groups_mtime": stat.st_mtime,
        "config": attribute_mining_config.model_dump(),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def lookup(
    scenario_name: str,
    fingerprint: str,
    root_dir: Path = FEATURE_DIR,
) -> Path | None:
    """Return the cached schema path for this fingerprint, or None if there's
    no matching entry or the schema file it points to no longer exists."""
    index = _load_cache_index(scenario_name, root_dir)
    entry = index.get(fingerprint)
    if entry is None:
        return None
    schema_path = Path(entry["schema_path"])
    return schema_path if schema_path.exists() else None


def record(
    scenario_name: str,
    fingerprint: str,
    schema_path: Path,
    root_dir: Path = FEATURE_DIR,
) -> None:
    index = _load_cache_index(scenario_name, root_dir)
    index[fingerprint] = {"schema_path": str(schema_path)}
    _save_cache_index(scenario_name, root_dir, index)


def mine_or_reuse_attribute_schema(
    scenario: str,
    alert_groups_path: Path,
    run_name: str,
    attribute_mining_config: AttributeMiningConfig,
    root_dir: Path = FEATURE_DIR,
    force: bool = False,
) -> tuple[Path, Path | None, dict]:
    """Mine an attribute schema for `scenario`, or reuse an already-mined one
    if the inputs (alert_groups file + config) match a previous run.

    Returns (schema_path, mining_run_dir, mining_stats). mining_run_dir is
    None on a cache hit, since no mining actually ran.
    """
    fingerprint = compute_fingerprint(alert_groups_path, attribute_mining_config)

    if not force:
        cached = lookup(scenario, fingerprint, root_dir)
        if cached is not None:
            print(
                f"  [cache] Reusing attribute schema for '{scenario}' "
                f"(fingerprint={fingerprint}) → {cached}"
            )
            return cached, None, {"cache_hit": True, "fingerprint": fingerprint}

    from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job

    result = run_alert_group_attribute_mining_job(
        alert_groups_path=alert_groups_path,
        scenario_name=scenario,
        run_name=run_name,
        config=attribute_mining_config,
    )

    print("--- Building and saving symbolic schema (attribute mining) ---")
    # source_label="attack" here is only a fallback for rows missing their own
    # label; result.mined_df always carries a real per-row source_label
    # (attribute_mining_job.py tags each survivor/leaf by its own
    # confidence_attack vs confidence_benign), so this never actually fires.
    schema_path, schema_build_stats = build_persist_and_register_symbolic_schema(
        df=result.mined_df,
        scenario_name=scenario,
        source_label="attack",
        schema_name="symbolic",
        root_dir=root_dir,
        predicates=result.predicates,
    )
    mining_stats = {
        "cache_hit": False,
        "fingerprint": fingerprint,
        "n_candidate_features": len(result.mined_df),
        "n_predicates": len(result.predicates),
        **schema_build_stats,
    }
    print(f"  Symbolic schema registered → {schema_path}")

    record(scenario, fingerprint, schema_path, root_dir)

    return schema_path, result.run_dir, mining_stats
