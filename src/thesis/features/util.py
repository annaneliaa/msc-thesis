from __future__ import annotations

from pathlib import Path
import re
import json

from thesis.schemas.features import FeatureSchema, SymbolicFeatureSchema
from thesis.schemas.mining import FeatureSelectionConfig

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def next_schema_version(
    scenario_name: str,
    schema_family: str = "symbolic",
    root_dir: Path = Path("artifacts/features"),
    bump: str = "patch",
) -> str:
    schema_dir = root_dir / scenario_name / schema_family
    schema_dir.mkdir(parents=True, exist_ok=True)

    versions: list[tuple[int, int, int]] = []

    for path in schema_dir.glob("*.json"):
        match = _VERSION_RE.match(path.stem)
        if match:
            versions.append(tuple(map(int, match.groups())))

    if not versions:
        return "0.1.0"

    major, minor, patch = max(versions)

    if bump == "major":
        return f"{major + 1}.0.0"

    if bump == "minor":
        return f"{major}.{minor + 1}.0"

    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"

    raise ValueError(f"Unsupported bump type: {bump}")


def register_symbolic_schema_version(
    manifest_path: Path,
    schema_filename: str,
    schema_name: str,
    schema_version: str,
) -> None:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    schemas = manifest.setdefault("schemas", {})

    symbolic_spec = schemas.setdefault(
        schema_name,
        {
            "type": "symbolic",
            "latest": None,
            "versions": {},
        },
    )

    symbolic_spec["type"] = "symbolic"
    symbolic_spec.setdefault("versions", {})

    symbolic_spec["versions"][schema_version] = {
        "path": schema_filename,
        "schema_version": schema_version,
    }

    symbolic_spec["latest"] = schema_version

    schemas["base+symbolic"] = {
        "type": "composite",
        "base": True,
        "dynamic": False,
        "symbolic": schema_name,
        "symbolic_version": schema_version,
    }

    schemas["base+dynamic+symbolic"] = {
        "type": "composite",
        "base": True,
        "dynamic": True,
        "symbolic": schema_name,
        "symbolic_version": schema_version,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def select_symbolic_features(
    schema: FeatureSchema,
    config: FeatureSelectionConfig,
) -> FeatureSchema:
    if schema.symbolic is None:
        return schema

    features = list(schema.symbolic.features)

    if config.min_utility_score is not None:
        features = [f for f in features if f.utility_score >= config.min_utility_score]

    features.sort(key=lambda f: f.utility_score, reverse=True)

    if config.top_k is not None:
        features = features[: config.top_k]

    filtered_symbolic = SymbolicFeatureSchema(
        schema_name=schema.symbolic.schema_name,
        schema_version=schema.symbolic.schema_version,
        features=features,
    )

    return FeatureSchema(
        schema_name=schema.schema_name,
        schema_version=schema.schema_version,
        base=schema.base,
        dynamic=schema.dynamic,
        symbolic=filtered_symbolic,
    )
