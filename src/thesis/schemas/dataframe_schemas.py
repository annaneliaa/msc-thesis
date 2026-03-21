# src/thesis/schemas/definitions.py
from __future__ import annotations

# Define expected columns and dtypes for internal data contract
# column-based schemas for pandas DataFrames
SCHEMAS: dict[str, dict[str, str]] = {
    "meta_alerts": {
        "alert_id": "object",
        "tokens": "object",
        "label": "int64",
    },
    "mining_output": {
        "candidate": "object",
        "c0": "int64",
        "c1": "int64",
        "n0": "int64",
        "n1": "int64",
        "window_id": "int64",
    },
    "mining_metrics": {
        "candidate": "object",
        "window_frequency": "float64",
        "min_support_count": "int64",
    },
}
