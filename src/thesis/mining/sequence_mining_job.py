from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.paths import ensure_artifact_dirs
from thesis.schemas.mining import MiningMetadata, MiningJobResult
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

from thesis.mining.prefixspan_mining import run_prefixspan, run_itemset_prefixspan
from thesis.mining.repeat_encoding import encode_runs, encode_sequence_of_itemsets
from thesis.mining.token_abstraction import (
    abstract_mail_hosts_item_sequence_df,
    abstract_mail_hosts_itemset_sequence_df,
)
from thesis.mining.util import (
    add_cross_label_sequence_supports,
    add_cross_label_itemset_sequence_supports,
    sort_sequences_for_class,
    save_filtered_sequence_views,
    save_filtered_itemset_sequence_views,
)
from thesis.mining.util import add_confidence_scores
from thesis.mining.load_mining_transactions import load_and_prepare_mining_transactions


def sort_frequent_sequences(
    df: pd.DataFrame, sort_by: str = "support_target"
) -> pd.DataFrame:
    """
    Sort frequent sequences DataFrame by specified column.
    """
    if sort_by not in df.columns:
        raise ValueError(f"Column {sort_by} not found in DataFrame")

    return df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)


def run_transaction_prefixspan_job(
    transactions_path: str | Path,
    scenario_name: str,
    run_name: str = "debug",
    min_support: float = 0.05,
    max_len: int | None = 3,
    target_label: str = "benign",
    run_dir: Path | None = None,
) -> MiningJobResult:
    """
    Mine frequent sequential patterns from MiningTransaction records.

    Expected input:
    - transactions: sequence of MiningTransaction
    - each transaction has sorted_items: list[set[str]] (one set per alert)
    - scenario_name: logical scenario identifier for logging/artifacts
    """
    ensure_artifact_dirs()

    print("Starting transaction PrefixSpan sequence mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        if run_dir is None:
            run_dir = create_run_dir(run_name)
        run_dir = run_dir / "prefixspan" / "items"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_tags(
            {
                "stage": "mining",
                "component": "transaction-sequence-mining",
                "algorithm": "prefixspan",
                "run_name": run_name,
                "scenario_name": scenario_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "transaction_prefixspan_mining",
                "scenario_name": scenario_name,
                "target_label": target_label,
                "min_support": min_support,
                "max_len": max_len if max_len is not None else -1,
                "input_type": "MiningTransaction.sorted_items",
                "output_format": "csv",
            }
        )

        transactions = load_and_prepare_mining_transactions(
            path=transactions_path,
            run_dir=run_dir,
        )

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

        target_sequences = [list(tx.sorted_items) for tx in target_group]
        other_sequences = [list(tx.sorted_items) for tx in other_group]
        all_sequences = [list(tx.sorted_items) for tx in transactions]

        mined_df = run_prefixspan(
            sequences=target_sequences,
            min_support=min_support,
            max_len=max_len,
            run_dir=run_dir,
        )

        mined_df = add_cross_label_sequence_supports(
            mined_df=mined_df,
            target_sequences=target_sequences,
            other_sequences=other_sequences,
            target_label=target_label,
            other_label=other_label,
        )

        mined_df = add_confidence_scores(mined_df)

        # Collapse consecutive same-item runs in each mined pattern into
        # repeat-encoded tokens (e.g. A, A, B → A__repeat_2, B).
        # Support counts are already correct; we only rename the patterns.
        print("Applying item-level repeat encoding to mined patterns...")
        mined_df["sequence"] = mined_df["sequence"].apply(
            lambda seq: tuple(encode_runs(list(seq)))
        )
        mined_df["sequence_str"] = mined_df["sequence"].apply(lambda x: " -> ".join(x))
        mined_df["k"] = mined_df["sequence"].apply(len)
        n_before = len(mined_df)
        mined_df = (
            mined_df.sort_values("support_count", ascending=False)
            .drop_duplicates(subset=["sequence"])
            .reset_index(drop=True)
        )
        if len(mined_df) < n_before:
            print(
                f"  Merged {n_before - len(mined_df)} patterns that collapsed to the same encoded form"
            )

        print("Applying mail host abstraction to mined patterns...")
        mined_df = abstract_mail_hosts_item_sequence_df(mined_df)

        save_dataframe_artifact(mined_df, run_dir, "frequent_sequences")

        benign_sorted_df = sort_sequences_for_class(mined_df, "benign")
        save_dataframe_artifact(
            benign_sorted_df, run_dir, "frequent_sequences_sorted_by_benign"
        )

        attack_sorted_df = sort_sequences_for_class(mined_df, "attack")
        save_dataframe_artifact(
            attack_sorted_df, run_dir, "frequent_sequences_sorted_by_attack"
        )

        print("Generating filtered views for top sequences...")
        save_filtered_sequence_views(mined_df, run_dir)

        print(f"Mining completed. Saved filtered sequence views to {run_dir}.")

        print("Generating summary statistics and metadata...")

        unique_items = set()
        for seq in all_sequences:
            for itemset in seq:
                unique_items.update(itemset)

        sequence_lengths = [len(seq) for seq in all_sequences]

        summary_df = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "n_transactions_total": len(transactions),
                    f"n_transactions_{target_label}": len(target_group),
                    f"n_transactions_{other_label}": len(other_group),
                    "n_unique_items": len(unique_items),
                    "avg_transaction_sequence_length": (
                        float(sum(sequence_lengths) / len(sequence_lengths))
                        if sequence_lengths
                        else 0.0
                    ),
                    "max_transaction_sequence_length": (
                        max(sequence_lengths) if sequence_lengths else 0
                    ),
                    "n_sequences": len(mined_df),
                    "avg_sequence_size": (
                        float(mined_df["k"].mean()) if len(mined_df) else 0.0
                    ),
                    "max_sequence_size": (
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
                "input_type": "MiningTransaction.sorted_items",
                "algorithm": "prefixspan",
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_transactions_total": float(len(transactions)),
                f"n_transactions_{target_label}": float(len(target_group)),
                f"n_transactions_{other_label}": float(len(other_group)),
                "n_sequences": float(len(mined_df)),
                "n_unique_items": float(len(unique_items)),
                "avg_transaction_sequence_length": (
                    float(sum(sequence_lengths) / len(sequence_lengths))
                    if sequence_lengths
                    else 0.0
                ),
                "max_transaction_sequence_length": (
                    float(max(sequence_lengths)) if sequence_lengths else 0.0
                ),
                "avg_sequence_size": (
                    float(mined_df["k"].mean()) if len(mined_df) else 0.0
                ),
                "max_sequence_size": (
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

        print(f"Finished sequence mining job. Saved artifacts to {run_dir}")

        return MiningJobResult(
            run_dir=run_dir,
            mined_df=mined_df,
            scenario_name=scenario_name,
            target_label=target_label,
        )


def run_transaction_itemset_prefixspan_job(
    transactions_path: str | Path,
    scenario_name: str,
    run_name: str = "debug",
    min_support: float = 0.05,
    max_len: int | None = 3,
    target_label: str = "benign",
    run_dir: Path | None = None,
) -> MiningJobResult:
    """
    Mine frequent itemset sequential patterns from MiningTransaction records.

    Each pattern step is a frozenset of items that must all be present in a
    single alert.  Steps must appear in strictly increasing alert order (s-extension).
    Items within one step may come from the same alert (i-extension).

    Expected input:
    - transactions: sequence of MiningTransaction
    - each transaction has sorted_items: list[set[str]] (one set per alert)
    """
    ensure_artifact_dirs()

    print("Starting transaction itemset PrefixSpan sequence mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        if run_dir is None:
            run_dir = create_run_dir(run_name)
        run_dir = run_dir / "prefixspan" / "itemsets"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_tags(
            {
                "stage": "mining",
                "component": "transaction-itemset-sequence-mining",
                "algorithm": "prefixspan-itemset",
                "run_name": run_name,
                "scenario_name": scenario_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "transaction_itemset_prefixspan_mining",
                "scenario_name": scenario_name,
                "target_label": target_label,
                "min_support": min_support,
                "max_len": max_len if max_len is not None else -1,
                "input_type": "MiningTransaction.sorted_items",
                "output_format": "csv",
            }
        )

        transactions = load_and_prepare_mining_transactions(
            path=transactions_path,
            run_dir=run_dir,
        )

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

        target_sequences = [list(tx.sorted_items) for tx in target_group]
        other_sequences = [list(tx.sorted_items) for tx in other_group]
        all_sequences = [list(tx.sorted_items) for tx in transactions]

        print("Encoding consecutive token repeats...")
        target_sequences_encoded = [
            encode_sequence_of_itemsets(seq) for seq in target_sequences
        ]
        other_sequences_encoded = [
            encode_sequence_of_itemsets(seq) for seq in other_sequences
        ]
        all_sequences_encoded = [
            encode_sequence_of_itemsets(seq) for seq in all_sequences
        ]

        mined_df = run_itemset_prefixspan(
            sequences=target_sequences_encoded,
            min_support=min_support,
            max_len=max_len,
            run_dir=run_dir,
        )

        mined_df = add_cross_label_itemset_sequence_supports(
            mined_df=mined_df,
            target_sequences=target_sequences_encoded,
            other_sequences=other_sequences_encoded,
            target_label=target_label,
            other_label=other_label,
        )

        mined_df = add_confidence_scores(mined_df)

        print("Applying mail host abstraction to mined itemset sequence patterns...")
        mined_df = abstract_mail_hosts_itemset_sequence_df(mined_df)

        save_dataframe_artifact(mined_df, run_dir, "frequent_itemset_sequences")

        benign_sorted_df = sort_sequences_for_class(mined_df, "benign")
        save_dataframe_artifact(
            benign_sorted_df, run_dir, "frequent_itemset_sequences_sorted_by_benign"
        )

        attack_sorted_df = sort_sequences_for_class(mined_df, "attack")
        save_dataframe_artifact(
            attack_sorted_df, run_dir, "frequent_itemset_sequences_sorted_by_attack"
        )

        print("Generating filtered views for top itemset sequences...")
        save_filtered_itemset_sequence_views(mined_df, run_dir)

        unique_items: set[str] = set()
        for seq in all_sequences_encoded:
            for itemset in seq:
                unique_items.update(itemset)

        sequence_lengths = [len(seq) for seq in all_sequences_encoded]

        summary_df = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "n_transactions_total": len(transactions),
                    f"n_transactions_{target_label}": len(target_group),
                    f"n_transactions_{other_label}": len(other_group),
                    "n_unique_items": len(unique_items),
                    "avg_transaction_length": (
                        float(sum(sequence_lengths) / len(sequence_lengths))
                        if sequence_lengths
                        else 0.0
                    ),
                    "max_transaction_length": (
                        max(sequence_lengths) if sequence_lengths else 0
                    ),
                    "n_itemset_sequences": len(mined_df),
                    "avg_sequence_steps": (
                        float(mined_df["k"].mean()) if len(mined_df) else 0.0
                    ),
                    "avg_sequence_items": (
                        float(mined_df["n_items"].mean()) if len(mined_df) else 0.0
                    ),
                    "max_sequence_steps": (
                        int(mined_df["k"].max()) if len(mined_df) else 0
                    ),
                }
            ]
        )
        save_dataframe_artifact(summary_df, run_dir, "summary_itemset_sequences")

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
                "input_type": "MiningTransaction.sorted_items (itemset sequences)",
                "algorithm": "prefixspan-itemset",
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_transactions_total": float(len(transactions)),
                f"n_transactions_{target_label}": float(len(target_group)),
                f"n_transactions_{other_label}": float(len(other_group)),
                "n_itemset_sequences": float(len(mined_df)),
                "n_unique_items": float(len(unique_items)),
                "runtime_sec": runtime_sec,
            }
        )

        log_artifact(str(run_dir))

        print(f"Finished itemset sequence mining job. Saved artifacts to {run_dir}")

        return MiningJobResult(
            run_dir=run_dir,
            mined_df=mined_df,
            scenario_name=scenario_name,
            target_label=target_label,
        )
