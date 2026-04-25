from __future__ import annotations

import json
from pathlib import Path

from thesis.schemas.features import (
    BaseFeatureSchema,
    DynamicFeatureSchema,
    FeatureSchema,
    SymbolicFeature,
    SymbolicFeatureSchema,
)


BASE_FEATURES = [
    "duration_sec",
    "n_items",
    "n_hosts",
    "n_shorts",
    "n_sigs",
    "n_internal_ips",
    "n_external_ips",
    "alerts_per_second",
]


DYNAMIC_FEATURES = [
    "short_count_1d",
    "short_fp_rate_1d",
    "short_attack_rate_1d",
    "host_count_1d",
    "host_fp_rate_1d",
    "host_attack_rate_1d",
    "ip_count_1d",
    "ip_fp_rate_1d",
    "ip_attack_rate_1d",
    "short_host_count_1d",
    "short_host_fp_rate_1d",
    "short_host_attack_rate_1d",
    "short_ip_count_1d",
    "short_ip_fp_rate_1d",
    "short_ip_attack_rate_1d",
    "seconds_since_short_seen",
    "seconds_since_host_seen",
    "seconds_since_ip_seen",
    "seconds_since_short_host_seen",
]


class FeatureSchemaRegistry:
    def __init__(self, root_dir: Path = Path("artifacts/features")) -> None:
        self.root_dir = root_dir

    def load(
        self,
        scenario_name: str,
        schema_name: str,
    ) -> FeatureSchema:
        manifest = self._load_manifest(scenario_name)

        schemas = manifest["schemas"]

        if schema_name not in schemas:
            raise KeyError(
                f"Schema '{schema_name}' not found for scenario '{scenario_name}'. "
                f"Available: {list(schemas)}"
            )

        spec = schemas[schema_name]

        if spec["type"] == "symbolic":
            symbolic_version_spec = self._resolve_symbolic_spec(spec)

            symbolic = self._load_symbolic_from_spec(
                scenario_name=scenario_name,
                spec=symbolic_version_spec,
            )

            return FeatureSchema(
                schema_name=schema_name,
                schema_version=symbolic_version_spec["schema_version"],
                base=None,
                dynamic=None,
                symbolic=symbolic,
            )

        if spec["type"] == "composite":
            symbolic = None
            symbolic_version = None

            symbolic_ref = spec.get("symbolic")

            if symbolic_ref:
                if symbolic_ref not in schemas:
                    raise KeyError(
                        f"Referenced symbolic schema '{symbolic_ref}' not found. "
                        f"Available: {list(schemas)}"
                    )

                symbolic_spec = schemas[symbolic_ref]

                symbolic_version_spec = self._resolve_symbolic_spec(
                    symbolic_spec,
                    version=spec.get("symbolic_version"),
                )

                symbolic_version = symbolic_version_spec["schema_version"]

                symbolic = self._load_symbolic_from_spec(
                    scenario_name=scenario_name,
                    spec=symbolic_version_spec,
                )

            return FeatureSchema(
                schema_name=schema_name,
                schema_version=symbolic_version or spec.get("schema_version", "0.1.0"),
                base=BaseFeatureSchema(BASE_FEATURES) if spec.get("base") else None,
                dynamic=(
                    DynamicFeatureSchema(DYNAMIC_FEATURES)
                    if spec.get("dynamic")
                    else None
                ),
                symbolic=symbolic,
            )

        raise ValueError(f"Unsupported schema spec type: {spec['type']}")

    def _resolve_symbolic_spec(
        self,
        spec: dict,
        version: str | None = None,
    ) -> dict:
        """
        Resolve a symbolic schema spec.

        Supports both:
        - old flat format:
          {"type": "symbolic", "path": "...", "schema_version": "0.1.0"}

        - new versioned format:
          {
            "type": "symbolic",
            "latest": "0.1.1",
            "versions": {
              "0.1.0": {"path": "..."},
              "0.1.1": {"path": "..."}
            }
          }
        """
        if "versions" not in spec:
            return spec

        selected_version = version or spec.get("latest")

        if selected_version is None:
            raise ValueError("Symbolic schema has no versions yet.")

        versions = spec["versions"]

        if selected_version not in versions:
            raise KeyError(
                f"Symbolic schema version '{selected_version}' not found. "
                f"Available: {list(versions)}"
            )

        resolved = dict(versions[selected_version])
        resolved["schema_version"] = selected_version

        return resolved

    def _load_manifest(self, scenario_name: str) -> dict:
        path = self.root_dir / scenario_name / "manifest.json"

        if not path.exists():
            raise FileNotFoundError(f"Feature schema manifest not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_symbolic_from_spec(
        self,
        scenario_name: str,
        spec: dict,
    ) -> SymbolicFeatureSchema:
        path = self.root_dir / scenario_name / spec["path"]

        if not path.exists():
            raise FileNotFoundError(f"Symbolic schema file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        return SymbolicFeatureSchema(
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
            features=[
                SymbolicFeature(
                    feature_name=item["feature_name"],
                    itemset=tuple(item["itemset"]),
                    source_label=item["source_label"],
                    support=item.get("support"),
                    confidence_attack=item.get("confidence_attack"),
                    confidence_benign=item.get("confidence_benign"),
                )
                for item in payload["features"]
            ],
        )


FEATURE_SCHEMAS = FeatureSchemaRegistry()
