from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Pydantic object models used in mining module (API payloads and metadata objects)


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
