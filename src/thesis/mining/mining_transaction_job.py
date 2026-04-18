from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.paths import ensure_artifact_dirs
from thesis.schemas.mining import MiningMetadata
from thesis.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    log_params,
    set_tags,
    start_run,
)
from thesis.utils.runs import (
    create_run_dir,
    save_dataframe_artifact,
    write_manifest,
)

from thesis.mining.eclat_mining import run_eclat
from thesis.mining.util import add_cross_label_supports
from thesis.mining.load_mining_transactions import load_and_prepare_mining_transactions


def run_transaction_eclat_job(
    transactions_path: str | Path,
    scenario_name: str,
    run_name: str = "debug",
    min_support: float = 0.05,
    max_len: int | None = 3,
    target_label: str = "benign",
) -> Path:
    """
    Mine frequent itemsets from in-memory MiningTransaction records.

    Expected input:
    - transactions: sequence of MiningTransaction
    - scenario_name: logical scenario identifier for logging/artifacts
    """
    ensure_artifact_dirs()

    print("Starting transaction Eclat mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        run_dir = create_run_dir(run_name)

        set_tags(
            {
                "stage": "mining",
                "component": "transaction-itemset-mining",
                "algorithm": "eclat",
                "run_name": run_name,
                "scenario_name": scenario_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "transaction_eclat_mining",
                "scenario_name": scenario_name,
                "target_label": target_label,
                "min_support": min_support,
                "max_len": max_len if max_len is not None else -1,
                "input_type": "MiningTransaction",
                "output_format": "csv",
            }
        )

        transactions = load_and_prepare_mining_transactions(
            path=transactions_path,
            run_dir=run_dir,
        )
        print(f"Loaded + prepared {len(transactions)} transactions for mining.")

        all_labels = sorted(
            {tx.tx_label for tx in transactions if tx.tx_label is not None}
        )
        other_labels = [label for label in all_labels if label != target_label]

        if len(other_labels) > 1:
            raise ValueError(
                f"Expected binary labels, got target={target_label} and others={other_labels}"
            )

        other_label = other_labels[0] if other_labels else "other"

        target_group = [tx for tx in transactions if tx.tx_label == target_label]
        other_group = [tx for tx in transactions if tx.tx_label != target_label]

        target_baskets = [frozenset(tx.items) for tx in target_group]
        other_baskets = [frozenset(tx.items) for tx in other_group]
        all_baskets = [frozenset(tx.items) for tx in transactions]

        mined_df = run_eclat(
            transactions=target_baskets,
            min_support=min_support,
            max_len=max_len,
            run_dir=run_dir,
        )

        mined_df = add_cross_label_supports(
            mined_df=mined_df,
            target_transactions=target_baskets,
            other_transactions=other_baskets,
            target_label=target_label,
            other_label=other_label,
        )

        save_dataframe_artifact(mined_df, run_dir, "frequent_itemsets")

        summary_df = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "n_transactions_total": len(transactions),
                    f"n_transactions_{target_label}": len(target_group),
                    f"n_transactions_{other_label}": len(other_group),
                    "n_unique_items": (
                        len(set().union(*all_baskets)) if all_baskets else 0
                    ),
                    "n_itemsets": len(mined_df),
                    "avg_itemset_size": (
                        float(mined_df["k"].mean()) if len(mined_df) else 0.0
                    ),
                    "max_itemset_size": (
                        int(mined_df["k"].max()) if len(mined_df) else 0
                    ),
                }
            ]
        )
        save_dataframe_artifact(summary_df, run_dir, "summary")

        runtime_sec = time.perf_counter() - t0

        meta = MiningMetadata(
            run_name=run_name,
            timestamp=datetime.now(timezone.utc),
            scenario_name=scenario_name,
            n_candidates=len(mined_df),
            run_id=run_dir.name,
            artifact_path=str(run_dir),
            n_transactions=len(transactions),
        )

        write_manifest(
            run_dir,
            config={
                "run_name": run_name,
                "scenario_name": scenario_name,
                "min_support": min_support,
                "max_len": max_len,
                "target_label": target_label,
                "input_type": "MiningTransaction",
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_transactions_total": float(len(transactions)),
                f"n_transactions_{target_label}": float(len(target_group)),
                f"n_transactions_{other_label}": float(len(other_group)),
                "n_itemsets": float(len(mined_df)),
                "n_unique_items": float(
                    len(set().union(*all_baskets)) if all_baskets else 0
                ),
                "avg_itemset_size": (
                    float(mined_df["k"].mean()) if len(mined_df) else 0.0
                ),
                "max_itemset_size": (
                    float(mined_df["k"].max()) if len(mined_df) else 0.0
                ),
                "avg_support_target": (
                    float(mined_df[f"support_{target_label}"].mean())
                    if len(mined_df)
                    else 0.0
                ),
                "avg_support_other": (
                    float(mined_df[f"support_{other_label}"].mean())
                    if len(mined_df)
                    else 0.0
                ),
                "avg_support_diff": (
                    float(mined_df["support_diff"].mean()) if len(mined_df) else 0.0
                ),
                "runtime_sec": runtime_sec,
            }
        )

        log_artifact(str(run_dir))

        print(f"Finished mining job. Saved artifacts to {run_dir}")
        return run_dir
