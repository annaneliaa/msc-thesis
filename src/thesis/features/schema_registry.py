from __future__ import annotations

import json
from pathlib import Path

from thesis.schemas.features import (
    AttributePredicate,
    BaseFeatureSchema,
    FeatureSchema,
    SymbolicFeature,
    SymbolicFeatureSchema,
)

from thesis.configs import (
    dataset_for_scenario,
    load_base_features,
)


class FeatureSchemaRegistry:
    def __init__(self, root_dir: Path = Path("artifacts/features")) -> None:
        self.root_dir = root_dir

    def load(
        self,
        scenario_name: str,
        schema_name: str,
        schema_version: str | None = None,
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
            symbolic_version_spec = self._resolve_symbolic_spec(
                spec,
                version=schema_version,
            )

            symbolic = self._load_symbolic_from_spec(
                scenario_name=scenario_name,
                spec=symbolic_version_spec,
            )

            return FeatureSchema(
                schema_name=schema_name,
                schema_version=symbolic_version_spec["schema_version"],
                base=None,
                symbolic=symbolic,
            )

        if spec["type"] == "composite":
            symbolic = None
            resolved_schema_version = schema_version or spec.get(
                "schema_version", "0.1.0"
            )

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
                    version=schema_version or spec.get("symbolic_version"),
                )

                resolved_schema_version = symbolic_version_spec["schema_version"]

                symbolic = self._load_symbolic_from_spec(
                    scenario_name=scenario_name,
                    spec=symbolic_version_spec,
                )

            elif schema_version is not None:
                raise ValueError(
                    f"Schema '{schema_name}' has no versioned component, "
                    f"so schema_version='{schema_version}' cannot be applied."
                )

            base = None
            if spec.get("base"):
                dataset = dataset_for_scenario(scenario_name)
                if dataset is None:
                    raise ValueError(
                        f"Scenario '{scenario_name}' is not listed under any "
                        "dataset in scenarios.json; cannot resolve base features."
                    )
                base = BaseFeatureSchema(load_base_features(dataset))

            return FeatureSchema(
                schema_name=schema_name,
                schema_version=resolved_schema_version,
                base=base,
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

        predicates = payload.get("predicates")

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
                    mining_type=item.get("mining_type"),
                    utility_score=item.get("utility_score", 1.0),
                    clauses=(
                        tuple(tuple(c) for c in item["clauses"])
                        if item.get("clauses") is not None
                        else None
                    ),
                    p_value=item.get("p_value"),
                )
                for item in payload["features"]
            ],
            predicates=(
                [
                    AttributePredicate(
                        token=p["token"],
                        attribute=p["attribute"],
                        operator=p["operator"],
                        value=p["value"],
                    )
                    for p in predicates
                ]
                if predicates is not None
                else None
            ),
        )


FEATURE_SCHEMAS = FeatureSchemaRegistry()
