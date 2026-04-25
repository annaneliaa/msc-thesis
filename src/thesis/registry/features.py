from __future__ import annotations

import json
from pathlib import Path

from thesis.paths import FEATURES_DIR


def get_manifest_path(scenario_name: str) -> Path:
    return FEATURES_DIR / scenario_name / "manifest.json"


def load_feature_manifest(scenario_name: str) -> dict:
    path = get_manifest_path(scenario_name)

    if not path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_scenarios() -> list[str]:
    if not FEATURES_DIR.exists():
        return []

    return sorted(
        path.name
        for path in FEATURES_DIR.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def list_all_feature_schemas(scenario_name: str) -> list[str]:
    manifest = load_feature_manifest(scenario_name)
    return sorted(manifest.get("schemas", {}).keys())


def list_symbolic_schema_versions(
    scenario_name: str,
    schema_name: str = "symbolic",
) -> list[str]:
    manifest = load_feature_manifest(scenario_name)
    schemas = manifest.get("schemas", {})

    if schema_name not in schemas:
        raise KeyError(
            f"Schema '{schema_name}' not found for scenario '{scenario_name}'. "
            f"Available: {list(schemas)}"
        )

    spec = schemas[schema_name]

    if spec.get("type") != "symbolic":
        raise ValueError(f"Schema '{schema_name}' is not symbolic.")

    versions = spec.get("versions", {})
    return sorted(versions.keys())


def get_latest_symbolic_schema_version(
    scenario_name: str,
    schema_name: str = "symbolic",
) -> str:
    manifest = load_feature_manifest(scenario_name)
    spec = manifest["schemas"][schema_name]

    latest = spec.get("latest")

    if latest is None:
        raise ValueError(
            f"Symbolic schema '{schema_name}' has no versions yet "
            f"for scenario '{scenario_name}'."
        )

    return latest


def get_schema_path(
    scenario_name: str,
    schema_name: str,
    version: str | None = None,
) -> Path:
    manifest = load_feature_manifest(scenario_name)
    schemas = manifest.get("schemas", {})

    if schema_name not in schemas:
        raise KeyError(
            f"Schema '{schema_name}' not found for scenario '{scenario_name}'. "
            f"Available: {list(schemas)}"
        )

    spec = schemas[schema_name]

    if spec.get("type") != "symbolic":
        raise ValueError(
            f"Only symbolic schemas have persisted schema files. "
            f"Schema '{schema_name}' has type '{spec.get('type')}'."
        )

    selected_version = version or spec.get("latest")

    if selected_version is None:
        raise ValueError(
            f"Symbolic schema '{schema_name}' has no versions yet "
            f"for scenario '{scenario_name}'."
        )

    versions = spec.get("versions", {})

    if selected_version not in versions:
        raise KeyError(
            f"Version '{selected_version}' not found for schema '{schema_name}'. "
            f"Available: {list(versions)}"
        )

    return FEATURES_DIR / scenario_name / versions[selected_version]["path"]
