from pathlib import Path
import json


def initialize_feature_manifest(
    scenario_name: str,
    root_dir: Path = Path("artifacts/features"),
    overwrite: bool = False,
) -> Path:
    """
    Create a default manifest.json for a scenario.

    Includes:
    - base
    - base+dynamic
    (symbolic entries are added later by mining jobs)
    """
    scenario_dir = root_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = scenario_dir / "manifest.json"

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. "
            f"Use overwrite=True to replace it."
        )

    manifest = {
        "scenario_name": scenario_name,
        "schemas": {
            "base": {
                "type": "composite",
                "base": True,
                "dynamic": False,
                "symbolic": None,
            },
            "base+dynamic": {
                "type": "composite",
                "base": True,
                "dynamic": True,
                "symbolic": None,
            },
            "symbolic": {
                "type": "symbolic",
                "latest": None,
                "versions": {},
            },
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
