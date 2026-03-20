from datetime import datetime
from pathlib import Path

import pandas as pd

from thesis.paths import MINING_DIR, ensure_artifact_dirs


def run_dummy_mining_job(run_name: str = "debug") -> Path:
    ensure_artifact_dirs()

    df = pd.DataFrame(
        [
            {"candidate": "token_a", "support": 10, "confidence": 0.8},
            {"candidate": "token_b", "support": 5, "confidence": 0.4},
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = MINING_DIR / f"{run_name}_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    return out_path