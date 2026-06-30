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

from thesis.mining.eclat_mining import run_eclat
from thesis.mining.or_mining import mine_or_disjunctions
from thesis.mining.util import (
    add_cross_label_supports,
    add_confidence_scores,
    cleanup_run_intermediates,
    sort_itemsets_for_class,
    save_filtered_views,
    select_top_itemsets_per_class,
)
from thesis.mining.load_mining_alert_groups import load_and_prepare_mining_alert_groups
from thesis.mining.token_abstraction import (
    abstract_mail_hosts_mined_df,
    abstract_mail_hosts_or_clauses_df,
)


def sort_frequent_itemsets(
    df: pd.DataFrame, sort_by: str = "support_target"
) -> pd.DataFrame:
    """
    Sort frequent itemsets DataFrame by specified column (default: target support).
    """
    if sort_by not in df.columns:
        raise ValueError(f"Column {sort_by} not found in DataFrame")

    return df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)


def run_alert_group_eclat_job(
    alert_groups_path: str | Path,
    scenario_name: str,
    run_name: str = "debug",
    min_support: float = 0.05,
    max_len: int | None = 3,
    target_label: str = "benign",
    run_dir: Path | None = None,
    jaccard_threshold: float = 0.98,
) -> MiningJobResult:
    """
    Mine frequent itemsets from in-memory MiningAlertGroup records.

    Expected input:
    - alert_groups: sequence of MiningAlertGroup
    - scenario_name: logical scenario identifier for logging/artifacts
    """
    ensure_artifact_dirs()

    print("Starting alert_group Eclat mining job...")
    t0 = time.perf_counter()

    with start_run(run_name):
        if run_dir is None:
            run_dir = create_run_dir(run_name)
        run_dir = run_dir / "eclat"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_tags(
            {
                "stage": "mining",
                "component": "alert_group-itemset-mining",
                "algorithm": "eclat",
                "run_name": run_name,
                "scenario_name": scenario_name,
            }
        )

        log_params(
            {
                "run_name": run_name,
                "job_type": "alert_group_eclat_mining",
                "scenario_name": scenario_name,
                "target_label": target_label,
                "min_support": min_support,
                "max_len": max_len if max_len is not None else -1,
                "input_type": "MiningAlertGroup",
                "output_format": "csv",
            }
        )

        alert_groups = load_and_prepare_mining_alert_groups(
            path=alert_groups_path,
            run_dir=run_dir,
        )

        n_mixed = sum(1 for tx in alert_groups if tx.group_label == "mixed")
        if n_mixed:
            print(
                f"  [warn] Dropping {n_mixed} mixed-label alert_groups before itemset mining"
            )
            alert_groups = [tx for tx in alert_groups if tx.group_label != "mixed"]

        all_labels = sorted(
            {tx.group_label for tx in alert_groups if tx.group_label is not None}
        )
        other_labels = [label for label in all_labels if label != target_label]

        if len(other_labels) > 1:
            raise ValueError(
                f"Expected binary labels, got target={target_label} and others={other_labels}"
            )

        _complement = {"benign": "attack", "attack": "benign"}
        if other_labels:
            other_label = other_labels[0]
        else:
            other_label = _complement.get(target_label, "other")
            print(
                f"  [warn] No '{other_label}' alert_groups in mining window; "
                f"confidence_{other_label} will be 0 for all patterns."
            )

        target_group = [tx for tx in alert_groups if tx.group_label == target_label]
        other_group = [tx for tx in alert_groups if tx.group_label != target_label]

        target_baskets = [frozenset(tx.items) for tx in target_group]
        other_baskets = [frozenset(tx.items) for tx in other_group]
        all_baskets = [frozenset(tx.items) for tx in alert_groups]

        mined_df = run_eclat(
            alert_groups=target_baskets,
            min_support=min_support,
            max_len=max_len,
            run_dir=run_dir,
        )

        mined_df = add_cross_label_supports(
            mined_df=mined_df,
            target_alert_groups=target_baskets,
            other_alert_groups=other_baskets,
            target_label=target_label,
            other_label=other_label,
        )

        mined_df = add_confidence_scores(mined_df)

        print("Applying mail host abstraction to mined itemsets...")
        mined_df = abstract_mail_hosts_mined_df(mined_df)

        save_dataframe_artifact(mined_df, run_dir, "frequent_itemsets")

        benign_sorted_df = sort_itemsets_for_class(mined_df, "benign")
        save_dataframe_artifact(
            benign_sorted_df, run_dir, "frequent_itemsets_sorted_by_benign"
        )

        attack_sorted_df = sort_itemsets_for_class(mined_df, "attack")
        save_dataframe_artifact(
            attack_sorted_df, run_dir, "frequent_itemsets_sorted_by_attack"
        )

        print("Generating filtered views for top itemsets...")
        save_filtered_views(mined_df, run_dir)

        print("Mining OR disjunctions from top feature itemsets...")
        feature_df = select_top_itemsets_per_class(
            mined_df,
            top_n_benign=100,
            top_n_attack=100,
            min_total_count=20,
            min_abs_support_diff=0.01,
            min_confidence=0.7,
        )
        or_df = mine_or_disjunctions(
            base_df=feature_df,
            target_alert_groups=target_baskets,
            other_alert_groups=other_baskets,
            target_label=target_label,
            other_label=other_label,
            jaccard_threshold=jaccard_threshold,
        )
        or_df["mining_type"] = "or_itemset"
        or_df = abstract_mail_hosts_or_clauses_df(or_df)
        save_dataframe_artifact(or_df, run_dir, "or_feature_itemsets")
        print(f"  Found {len(or_df)} OR patterns.")

        print(f"Mining completed. Saved filtered itemset views to {run_dir}.")

        print("Generating summary statistics and metadata...")

        summary_df = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "n_alert_groups_total": len(alert_groups),
                    f"n_alert_groups_{target_label}": len(target_group),
                    f"n_alert_groups_{other_label}": len(other_group),
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
            n_alert_groups=len(alert_groups),
        )

        write_manifest(
            run_dir,
            config={
                "run_name": run_name,
                "scenario_name": scenario_name,
                "min_support": min_support,
                "max_len": max_len,
                "target_label": target_label,
                "input_type": "MiningAlertGroup",
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_alert_groups_total": float(len(alert_groups)),
                f"n_alert_groups_{target_label}": float(len(target_group)),
                f"n_alert_groups_{other_label}": float(len(other_group)),
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

        cleanup_run_intermediates(run_dir)
        print(f"Finished mining job. Saved artifacts to {run_dir}")

        return MiningJobResult(
            run_dir=run_dir,
            mined_df=mined_df,
            scenario_name=scenario_name,
            target_label=target_label,
            or_df=or_df if not or_df.empty else None,
        )
