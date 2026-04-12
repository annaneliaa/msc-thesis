from __future__ import annotations

import ast
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from thesis.schemas.mining import MiningMetadata
from thesis.paths import ensure_artifact_dirs

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


def _parse_items_column(items: str) -> frozenset[str]:
    """
    Parse a stringified Python set from the CSV items column.

    Example input:
        "{'A-Aud-Com6', 'A-All-Evt', 'cloud_share'}"
    """
    if pd.isna(items):
        return frozenset()

    parsed = ast.literal_eval(items)

    if isinstance(parsed, (set, list, tuple)):
        return frozenset(str(x) for x in parsed if str(x).strip())

    raise ValueError(f"Unsupported items format: {type(parsed)}")


def _prepare_transactions(
    df: pd.DataFrame,
    items_col: str = "items",
    label_col: str = "tx_label",
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Parse and clean transaction baskets from the scenario CSV.
    """
    out = df.copy()

    out["basket"] = out[items_col].apply(_parse_items_column)
    out["basket_size"] = out["basket"].apply(len)

    out = out[out["basket_size"] > 0].copy()
    out = out[out[label_col].notna()].copy()

    out.to_csv(run_dir / "prepared_transactions.csv", index=False)

    return out.reset_index(drop=True)


def _build_tidsets(
    transactions: Iterable[frozenset[str]], run_dir: Path
) -> dict[str, set[int]]:
    """
    Build vertical tidsets for Eclat.
    """
    tidsets: dict[str, set[int]] = {}

    for tid, basket in enumerate(transactions):
        for item in basket:
            tidsets.setdefault(item, set()).add(tid)

    # save tidsets to artifacts dir for debugging
    tidsets_path = run_dir / "tidsets.json"
    with open(tidsets_path, "w") as f:
        json.dump(
            {item: sorted(list(tids)) for item, tids in tidsets.items()}, f, indent=2
        )

    return tidsets


def _eclat_recursive(
    prefix: tuple[str, ...],
    items_with_tidsets: list[tuple[str, set[int]]],
    min_count: int,
    results: list[dict],
    max_len: int | None = None,
) -> None:
    """
    Simple Eclat recursion.
    """
    for i, (item, tidset) in enumerate(items_with_tidsets):
        new_itemset = prefix + (item,)
        support_count = len(tidset)

        if support_count < min_count:
            continue

        results.append(
            {
                "itemset": new_itemset,
                "k": len(new_itemset),
                "support_count": support_count,
            }
        )

        if max_len is not None and len(new_itemset) >= max_len:
            continue

        suffix: list[tuple[str, set[int]]] = []
        for j in range(i + 1, len(items_with_tidsets)):
            other_item, other_tidset = items_with_tidsets[j]
            new_tidset = tidset & other_tidset
            if len(new_tidset) >= min_count:
                suffix.append((other_item, new_tidset))

        if suffix:
            _eclat_recursive(
                prefix=new_itemset,
                items_with_tidsets=suffix,
                min_count=min_count,
                results=results,
                max_len=max_len,
            )


def _run_eclat(
    transactions: list[frozenset[str]],
    min_support: float = 0.05,
    max_len: int | None = 3,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run Eclat on a list of transactions.

    Returns one row per frequent itemset.
    """
    n_tx = len(transactions)
    if n_tx == 0:
        return pd.DataFrame(
            columns=["itemset", "itemset_str", "k", "support_count", "support"]
        )

    min_count = max(
        1, int(min_support * n_tx)
    )  # each itemset must appear in at least this many transactions to be considered frequent
    tidsets = _build_tidsets(transactions, run_dir)

    # sort for deterministic output
    items_with_tidsets = sorted(tidsets.items(), key=lambda x: (x[0], len(x[1])))

    items_with_tidsets_path = run_dir / "items_with_tidsets.json"
    with open(items_with_tidsets_path, "w") as f:
        json.dump(
            [
                {
                    "item": item,
                    "support_count": len(tidset),
                    "tids": sorted(list(tidset)),
                }
                for item, tidset in items_with_tidsets
            ],
            f,
            indent=2,
        )

    results: list[dict] = []
    _eclat_recursive(
        prefix=(),
        items_with_tidsets=items_with_tidsets,
        min_count=min_count,
        results=results,
        max_len=max_len,
    )

    if not results:
        return pd.DataFrame(
            columns=["itemset", "itemset_str", "k", "support_count", "support"]
        )

    out = pd.DataFrame(results).drop_duplicates(subset=["itemset"]).copy()
    out["itemset_str"] = out["itemset"].apply(lambda x: " | ".join(x))
    out["support"] = out["support_count"] / n_tx
    out["n_transactions"] = n_tx
    out["min_support"] = min_support
    out["min_count"] = min_count

    return out.sort_values(
        by=["k", "support_count", "itemset_str"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _support_in_group(
    transactions: list[frozenset[str]],
    itemset: tuple[str, ...],
) -> tuple[int, float]:
    """
    Count how many transactions contain the itemset.
    """
    if not transactions:
        return 0, 0.0

    itemset_set = set(itemset)
    count = sum(1 for basket in transactions if itemset_set.issubset(basket))
    support = count / len(transactions)
    return count, support


def _add_cross_label_supports(
    mined_df: pd.DataFrame,
    target_transactions: list[frozenset[str]],
    other_transactions: list[frozenset[str]],
    target_label: str,
    other_label: str,
) -> pd.DataFrame:
    """
    For each mined itemset, compute support in both target and other label groups.
    """
    if mined_df.empty:
        return mined_df.copy()

    out = mined_df.copy()

    target_counts = []
    target_supports = []
    other_counts = []
    other_supports = []

    for itemset in out["itemset"]:
        c_t, s_t = _support_in_group(target_transactions, itemset)
        c_o, s_o = _support_in_group(other_transactions, itemset)

        target_counts.append(c_t)
        target_supports.append(s_t)
        other_counts.append(c_o)
        other_supports.append(s_o)

    out[f"count_{target_label}"] = target_counts
    out[f"support_{target_label}"] = target_supports
    out[f"count_{other_label}"] = other_counts
    out[f"support_{other_label}"] = other_supports
    out["support_diff"] = out[f"support_{target_label}"] - out[f"support_{other_label}"]

    # sort by descending support difference, then by support in target label, then by itemset size
    return out.sort_values(
        by=["support_diff", f"support_{target_label}", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_transaction_eclat_job(
    scenario_csv: str | Path,
    run_name: str = "debug",
    min_support: float = 0.05,
    max_len: int | None = 3,
    target_label: str = "benign",
    label_col: str = "tx_label",
    items_col: str = "items",
) -> Path:
    """
    Mine frequent itemsets from transaction-level baskets in one scenario CSV.

    Expected input columns:
    - items
    - tx_label

    Typical usage:
        run_transaction_eclat_job(
            scenario_csv="data/processed/falcon_transactions.csv",
            run_name="falcon_eclat_v1",
            min_support=0.05,
            max_len=3,
        )
    """
    ensure_artifact_dirs()

    print("Starting transaction Eclat mining job...")
    t0 = time.perf_counter()

    scenario_csv = Path(scenario_csv)
    scenario_name = scenario_csv.stem

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
                "scenario_csv": str(scenario_csv),
                "scenario_name": scenario_name,
                "target_label": target_label,
                "label_col": label_col,
                "items_col": items_col,
                "min_support": min_support,
                "max_len": max_len if max_len is not None else -1,
                "output_format": "csv",
            }
        )

        df_raw = pd.read_csv(scenario_csv)
        df = _prepare_transactions(
            df_raw, items_col=items_col, label_col=label_col, run_dir=run_dir
        )

        all_labels = sorted(df[label_col].unique().tolist())
        other_labels = [x for x in all_labels if x != target_label]

        if len(other_labels) > 1:
            raise ValueError(
                f"Expected binary labels, got target={target_label} and others={other_labels}"
            )

        other_label = other_labels[0] if other_labels else "other"

        df_target = df[df[label_col] == target_label].copy()
        df_other = df[df[label_col] != target_label].copy()

        target_transactions = df_target["basket"].tolist()
        other_transactions = df_other["basket"].tolist()

        mined_df = _run_eclat(
            transactions=target_transactions,
            min_support=min_support,
            max_len=max_len,
            run_dir=run_dir,
        )

        mined_df = _add_cross_label_supports(
            mined_df=mined_df,
            target_transactions=target_transactions,
            other_transactions=other_transactions,
            target_label=target_label,
            other_label=other_label,
        )

        save_dataframe_artifact(mined_df, run_dir, "frequent_itemsets")

        summary_df = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "n_transactions_total": len(df),
                    f"n_transactions_{target_label}": len(df_target),
                    f"n_transactions_{other_label}": len(df_other),
                    "n_unique_items": len(set().union(*df["basket"])) if len(df) else 0,
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
            n_transactions=len(df),
        )

        write_manifest(
            run_dir,
            config={
                "run_name": run_name,
                "scenario_csv": str(scenario_csv),
                "scenario_name": scenario_name,
                "min_support": min_support,
                "max_len": max_len,
                "target_label": target_label,
            },
            metadata=meta.model_dump(mode="json"),
        )

        log_metrics(
            {
                "n_transactions_total": float(len(df)),
                f"n_transactions_{target_label}": float(len(df_target)),
                f"n_transactions_{other_label}": float(len(df_other)),
                "n_itemsets": float(len(mined_df)),
                "n_unique_items": float(
                    len(set().union(*df["basket"])) if len(df) else 0
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
