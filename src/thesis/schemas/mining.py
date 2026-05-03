from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

# Pydantic object models used in mining module (API payloads and metadata objects


class MiningMetadata(BaseModel):
    # core identity
    run_name: str
    timestamp: datetime

    # data context
    scenario_name: Optional[str] = None
    n_candidates: int

    # run info
    run_id: Optional[str] = None
    artifact_path: Optional[str] = None

    # basic stats (optional but useful)
    n_windows: Optional[int] = None
    n_alerts: Optional[int] = None
    n_transactions: Optional[int] = None

    # config traceability
    config_name: Optional[str] = None


@dataclass(slots=True)
class MiningTransaction:
    """
    Canonical input record for the mining module.

    This is the transaction-level representation consumed by itemset mining.
    It is independent from preprocessing/cache schemas.
    Mining requires a label for the transaction (e.g. "benign" or "attack") and a set of items.
    """

    transaction_id: int | str
    tx_label: str
    items: set[str] = field(default_factory=set)
    sorted_items: list[set[str]] = field(default_factory=list)
    window_start: int | None = None
    window_end: int | None = None
    n_alerts: int | None = None
    alert_labels: set[str] | None = None
    weight: float = 1.0


@dataclass(slots=True)
class MiningJobResult:
    run_dir: Path
    mined_df: pd.DataFrame
    scenario_name: str
    target_label: str
