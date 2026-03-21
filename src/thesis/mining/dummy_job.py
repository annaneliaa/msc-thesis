from datetime import datetime
from pathlib import Path

import pandas as pd

from thesis.paths import ensure_artifact_dirs
from thesis.schemas.mining import MiningMetadata


from thesis.utils.runs import create_run_dir, save_dataframe_artifact, write_manifest


def run_dummy_mining_job(run_name: str = "debug") -> Path:
    ensure_artifact_dirs()

    run_dir = create_run_dir(run_name)

    df = pd.DataFrame(
        [
            {"candidate": "token_a", "support": 10, "confidence": 0.8},
            {"candidate": "token_b", "support": 5, "confidence": 0.4},
        ]
    )

    save_dataframe_artifact(df, run_dir, "candidates")

    meta = MiningMetadata(
        run_name=run_name,
        n_candidates=len(df),
        timestamp=datetime.utcnow(),
    )

    write_manifest(
        run_dir,
        config={"run_name": run_name},
        metadata=meta.model_dump(),
    )

    return str(run_dir)
