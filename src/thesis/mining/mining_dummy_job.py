from datetime import datetime, timezone
from pathlib import Path
import time
import pandas as pd

from thesis.paths import ensure_artifact_dirs
from thesis.schemas.mining import MiningMetadata
from thesis.utils.runs import (
    create_run_dir,
    save_dataframe_artifact,
    write_manifest,
)

from thesis.utils.mlflow_utils import (
    start_run,
    log_params,
    log_metrics,
    log_artifact,
    set_tags,
)


def run_dummy_mining_job(run_name: str = "debug") -> Path:
    ensure_artifact_dirs()

    print("Starting run of mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        run_dir = create_run_dir(run_name)

        set_tags(
            {
                "stage": "dummy-mining",
                "component": "mining",
                "run_name": run_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "dummy_mining",
                "output_format": "csv",
            }
        )

        df = pd.DataFrame(
            [
                {"candidate": "token_a", "support": 10, "confidence": 0.8},
                {"candidate": "token_b", "support": 5, "confidence": 0.4},
            ]
        )

        save_dataframe_artifact(df, run_dir, "candidates")

        runtime_sec = time.perf_counter() - t0

        meta = MiningMetadata(
            run_name=run_name,
            timestamp=datetime.now(timezone.utc),
            n_candidates=len(df),
            run_id=run_dir.name,
            artifact_path=str(run_dir),
        )

        write_manifest(
            run_dir,
            config={"run_name": run_name},
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_candidates": len(df),
                "avg_support": float(df["support"].mean()),
                "max_support": float(df["support"].max()),
                "avg_confidence": float(df["confidence"].mean()),
                "runtime_sec": runtime_sec,
            }
        )

        log_artifact(str(run_dir))

        print(f"Finished mining job. Saved artifacts to {run_dir}")

        return run_dir
