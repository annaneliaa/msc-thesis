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


class FeatureSchemaRegistry:
    def __init__(self, root_dir: Path = Path("artifacts/features")) -> None:
        self.root_dir = root_dir
        self._schemas: dict[str, FeatureSchema] = {
            "default": FeatureSchema(
                schema_name="default",
                schema_version="0.1.0",
                base=BaseFeatureSchema(
                    features=[
                        "duration_sec",
                        "n_alerts",
                        "n_items",
                        "n_hosts",
                        "n_shorts",
                        "n_sigs",
                        "n_internal_ips",
                        "n_external_ips",
                        "alerts_per_second",
                    ]
                ),
                dynamic=None,
                symbolic=None,
            ),
            "base+dynamic": FeatureSchema(
                schema_name="base+dynamic",
                schema_version="0.1.0",
                base=BaseFeatureSchema(
                    features=[
                        "duration_sec",
                        "n_alerts",
                        "n_items",
                        "n_hosts",
                        "n_shorts",
                        "n_sigs",
                        "n_internal_ips",
                        "n_external_ips",
                        "alerts_per_second",
                    ]
                ),
                dynamic=DynamicFeatureSchema(
                    features=[
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
                ),
                symbolic=None,
            ),
            "symbolic": FeatureSchema(
                schema_name="symbolic",
                schema_version="0.1.0",
                base=None,
                dynamic=None,
                symbolic=SymbolicFeatureSchema(features=[]),
            ),
        }

    def get_schema_by_name(self, name: str) -> FeatureSchema:
        return self._schemas[name]

    def load_symbolic_schema(
        self,
        scenario_name: str,
        schema_name: str,
    ) -> SymbolicFeatureSchema:
        schema_path = self.root_dir / scenario_name / f"{schema_name}.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"Symbolic schema not found: {schema_path}")

        with schema_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        features = [
            SymbolicFeature(
                feature_name=item["feature_name"],
                itemset=tuple(item["itemset"]),
                source_label=item["source_label"],
                support=item.get("support"),
                confidence_attack=item.get("confidence_attack"),
                confidence_benign=item.get("confidence_benign"),
            )
            for item in payload.get("features", [])
        ]

        return SymbolicFeatureSchema(features=features)

    def get_schema_with_loaded_symbolic(
        self,
        name: str,
        scenario_name: str,
        symbolic_schema_name: str,
    ) -> FeatureSchema:
        schema = self.get_schema_by_name(name)
        symbolic = self.load_symbolic_schema(
            scenario_name=scenario_name,
            schema_name=symbolic_schema_name,
        )

        return FeatureSchema(
            schema_name=schema.schema_name,
            schema_version=schema.schema_version,
            base=schema.base,
            dynamic=schema.dynamic,
            symbolic=symbolic,
        )


FEATURE_SCHEMAS = FeatureSchemaRegistry()
