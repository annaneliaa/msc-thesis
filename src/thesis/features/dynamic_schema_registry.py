from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from thesis.features.dynamic_schema_persistence import (
    load_dynamic_schema,
    save_dynamic_schema,
)
from thesis.paths import FEATURE_DIR
from thesis.schemas.dynamic_schema import DynamicSchema

# Guards the full read-manifest -> allocate-version -> build -> write-schema
# -> write-manifest sequence in deploy(), same rationale as
# features/service.py::_REGISTER_LOCK: a future Experiment 4 will call
# deploy() from a thread pool the same way rolling_walk_forward.py mines
# concurrently today, and every step here is a read-then-write against
# shared on-disk state with no other synchronization.
_DYNAMIC_REGISTRY_LOCK = threading.Lock()

_MANIFEST_FILENAME = "manifest.json"
_SCHEMAS_SUBDIR = "schemas"


class DynamicSchemaRegistry:
    """
    Deployment-scoped registry for DynamicSchema (Vk) versions -- distinct
    from FeatureSchemaRegistry/manifest.json, which tracks experiment-scoped
    SymbolicFeatureSchema versions. Lives at its own manifest file so the two
    versioning systems never share state:

        artifacts/features/<scenario>/dynamic/manifest.json
        artifacts/features/<scenario>/dynamic/schemas/<version>.json
    """

    def __init__(self, root_dir: Path = FEATURE_DIR) -> None:
        self.root_dir = root_dir

    def deploy(
        self,
        scenario_name: str,
        build_fn: Callable[[int], DynamicSchema],
    ) -> Path:
        """
        Allocate the next integer version, build the schema for that version
        via build_fn(version) (so every predicate/rule's schema_version is
        stamped correctly before persisting), mark the previously-deployed
        entry's superseded_at, write the new entry as deployed, update the
        manifest. Returns the schema file path.
        """
        with _DYNAMIC_REGISTRY_LOCK:
            dynamic_dir = self._dynamic_dir(scenario_name)
            manifest_path = dynamic_dir / _MANIFEST_FILENAME
            manifest = self._load_or_init_manifest(scenario_name, manifest_path)

            version = self._next_version(manifest)
            now = datetime.now(timezone.utc)

            schema = build_fn(version)
            schema = DynamicSchema(
                version=schema.version,
                mined_at=schema.mined_at,
                mining_window_start=schema.mining_window_start,
                mining_window_end=schema.mining_window_end,
                base_attack_rate=schema.base_attack_rate,
                single_predicates=schema.single_predicates,
                compound_rules=schema.compound_rules,
                deployed_at=now,
                superseded_at=None,
            )

            schema_filename = f"{_SCHEMAS_SUBDIR}/{version}.json"
            schema_path = dynamic_dir / schema_filename
            if schema_path.exists():
                raise FileExistsError(f"Dynamic schema already exists: {schema_path}")
            save_dynamic_schema(schema, schema_path)

            for entry in manifest["history"]:
                if entry["superseded_at"] is None:
                    entry["superseded_at"] = now.isoformat()

            manifest["deployed"] = version
            manifest["history"].append(
                {
                    "version": version,
                    "path": schema_filename,
                    "deployed_at": now.isoformat(),
                    "superseded_at": None,
                }
            )
            self._write_manifest(manifest_path, manifest)

            return schema_path

    def load_deployed(self, scenario_name: str) -> DynamicSchema:
        manifest = self._load_manifest(scenario_name)
        deployed = manifest.get("deployed")
        if deployed is None:
            raise FileNotFoundError(
                f"No dynamic schema deployed yet for scenario '{scenario_name}'."
            )
        return self.load_version(scenario_name, deployed)

    def load_version(self, scenario_name: str, version: int) -> DynamicSchema:
        manifest = self._load_manifest(scenario_name)
        entry = next((h for h in manifest["history"] if h["version"] == version), None)
        if entry is None:
            raise KeyError(
                f"Dynamic schema version {version} not found for scenario "
                f"'{scenario_name}'. Available: "
                f"{[h['version'] for h in manifest['history']]}"
            )
        schema_path = self._dynamic_dir(scenario_name) / entry["path"]
        schema = load_dynamic_schema(schema_path)

        # deployed_at/superseded_at are registry-owned: the manifest, not the
        # schema file's own (write-time) copy, is the source of truth for
        # them -- a later deploy() only updates the manifest entry when
        # marking this version superseded, not the immutable schema file
        # written back when this version was first deployed.
        return DynamicSchema(
            version=schema.version,
            mined_at=schema.mined_at,
            mining_window_start=schema.mining_window_start,
            mining_window_end=schema.mining_window_end,
            base_attack_rate=schema.base_attack_rate,
            single_predicates=schema.single_predicates,
            compound_rules=schema.compound_rules,
            deployed_at=(
                datetime.fromisoformat(entry["deployed_at"])
                if entry["deployed_at"] is not None
                else None
            ),
            superseded_at=(
                datetime.fromisoformat(entry["superseded_at"])
                if entry["superseded_at"] is not None
                else None
            ),
        )

    def history(self, scenario_name: str) -> list[dict]:
        manifest = self._load_manifest(scenario_name)
        return list(manifest["history"])

    def _dynamic_dir(self, scenario_name: str) -> Path:
        return self.root_dir / scenario_name / "dynamic"

    def _load_or_init_manifest(self, scenario_name: str, manifest_path: Path) -> dict:
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"scenario_name": scenario_name, "deployed": None, "history": []}

    def _load_manifest(self, scenario_name: str) -> dict:
        manifest_path = self._dynamic_dir(scenario_name) / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dynamic schema manifest not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, manifest_path: Path, manifest: dict) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _next_version(self, manifest: dict) -> int:
        if not manifest["history"]:
            return 1
        return max(h["version"] for h in manifest["history"]) + 1


DYNAMIC_SCHEMAS = DynamicSchemaRegistry()
