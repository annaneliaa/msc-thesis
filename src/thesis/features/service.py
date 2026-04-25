from pathlib import Path
import pandas as pd

from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.features.persistence import save_symbolic_feature_schema
from thesis.features.manifest import initialize_feature_manifest
from thesis.features.util import next_schema_version, register_symbolic_schema_version


def build_persist_and_register_symbolic_schema(
    df: pd.DataFrame,
    scenario_name: str,
    source_label: str,
    schema_name: str = "symbolic",
    schema_version: str | None = None,
    max_features: int | None = None,
    root_dir: Path = Path("artifacts/features"),
    bump: str = "patch",
) -> Path:
    """
    Build symbolic features from mined itemsets, persist them versioned,
    and register them in the scenario manifest.

    Writes to:
        artifacts/features/{scenario_name}/symbolic/{schema_version}.json

    Updates manifest entries:
        symbolic
        base+symbolic
        base+dynamic+symbolic
    """
    manifest_path = root_dir / scenario_name / "manifest.json"

    if not manifest_path.exists():
        initialize_feature_manifest(
            scenario_name=scenario_name,
            root_dir=root_dir,
            overwrite=False,
        )

    if schema_version is None:
        schema_version = next_schema_version(
            scenario_name=scenario_name,
            schema_family=schema_name,
            root_dir=root_dir,
            bump=bump,
        )

    symbolic_schema = build_symbolic_feature_schema(
        df=df,
        source_label=source_label,
        schema_name=schema_name,
        schema_version=schema_version,
        max_features=max_features,
    )

    schema_filename = f"{schema_name}/{schema_version}.json"
    schema_path = root_dir / scenario_name / schema_filename

    if schema_path.exists():
        raise FileExistsError(
            f"Symbolic schema already exists: {schema_path}. "
            "Use a new version number or leave schema_version=None."
        )

    save_symbolic_feature_schema(symbolic_schema, schema_path)

    register_symbolic_schema_version(
        manifest_path=manifest_path,
        schema_filename=schema_filename,
        schema_name=schema_name,
        schema_version=schema_version,
    )

    return schema_path
