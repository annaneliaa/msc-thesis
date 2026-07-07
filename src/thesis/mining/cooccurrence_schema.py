"""Cross-signature/cross-alert cooccurrence mining (Eclat + PrefixSpan).

Extracted from experiments/symbolic.py so that module only orchestrates the
8-step symbolic experiment pipeline and doesn't also carry the ~400-line
implementation of one of its two mining strategies. Behavior is unchanged
from before the extraction; this is the "cooccurrence" counterpart to
mining/attribute_schema_cache.py's per-alert-group attribute mining path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from thesis.config import load_mining_filter_config
from thesis.features.service import build_persist_and_register_symbolic_schema
from thesis.mining.itemset_mining_job import run_alert_group_eclat_job
from thesis.mining.sequence_mining_job import run_alert_group_prefixspan_job
from thesis.mining.token_abstraction import (
    abstract_mined_df,
    abstract_or_clauses_df,
    load_abstraction_map,
)
from thesis.mining.util import (
    filter_mined_itemsets,
    filter_mined_sequences,
    filter_or_patterns,
)
from thesis.utils.runs import create_run_dir

_ROOT = Path(__file__).resolve().parents[3]


def mine_and_register_cooccurrence_schema(
    scenario: str,
    alert_groups_path: Path,
    run_name: str,
    min_support: float,
    max_itemset_size: int,
    max_seq_len: int,
    target_label: str,
    filter_config: Path | None,
    jaccard_threshold: float = 0.98,
    abstraction_map_path: Path | None = None,
    abstraction_level: int = 0,
    min_support_diff: float = 0.05,
    run_attack_pass: bool = True,
    has_sequence_data: bool = True,
) -> tuple[Path, Path, dict]:
    run_dir = create_run_dir(run_name)

    _top_k_per_pass: int | None = None
    if filter_config is not None:
        _fc = filter_config if filter_config.is_absolute() else _ROOT / filter_config
        _top_k_per_pass = load_mining_filter_config(_fc).item_sequences.top_k_per_pass

    print("--- Running Eclat itemset mining on benign (AND + AND/OR) ---")
    eclat_result = run_alert_group_eclat_job(
        alert_groups_path=alert_groups_path,
        scenario_name=scenario,
        run_name=run_name,
        min_support=min_support,
        max_len=max_itemset_size,
        target_label=target_label,
        run_dir=run_dir,
        jaccard_threshold=jaccard_threshold,
    )

    if has_sequence_data:
        print("--- Running PrefixSpan sequence mining ---")
        item_seq_result = run_alert_group_prefixspan_job(
            alert_groups_path=alert_groups_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_seq_len,
            target_label=target_label,
            run_dir=run_dir,
            top_k_per_pass=_top_k_per_pass,
        )
        item_seq_df = item_seq_result.mined_df.copy()
    else:
        print(
            "  [skip] All alert_groups have empty sorted_items (pre-grouped "
            "scenario) — skipping sequence mining."
        )
        item_seq_df = pd.DataFrame()

    eclat_df = eclat_result.mined_df.copy()

    mining_stats: dict = {
        "n_itemsets_mined": len(eclat_df),
        "n_sequences_mined": len(item_seq_df),
        "n_or_mined": len(eclat_result.or_df) if eclat_result.or_df is not None else 0,
        "abstraction_applied": abstraction_map_path is not None,
        "abstraction_level": abstraction_level
        if abstraction_map_path is not None
        else None,
        "n_itemsets_after_abstraction": None,
        "n_sequences_after_abstraction": None,
        "n_or_after_abstraction": None,
        "n_itemsets_after_filter": None,
        "n_sequences_after_filter": None,
    }

    if abstraction_map_path is not None:
        print("--- Applying token abstraction ---")
        abstraction_map = load_abstraction_map(abstraction_map_path)
        n_eclat_before, n_seq_before = len(eclat_df), len(item_seq_df)
        eclat_df = abstract_mined_df(eclat_df, abstraction_map, level=abstraction_level)
        item_seq_df = abstract_mined_df(
            item_seq_df, abstraction_map, level=abstraction_level, column="sequence"
        )
        print(
            f"  itemsets {n_eclat_before}→{len(eclat_df)}, "
            f"sequences {n_seq_before}→{len(item_seq_df)} "
            f"(level={abstraction_level})"
        )
        mining_stats["n_itemsets_after_abstraction"] = len(eclat_df)
        mining_stats["n_sequences_after_abstraction"] = len(item_seq_df)
        if eclat_result.or_df is not None and not eclat_result.or_df.empty:
            n_or_before = len(eclat_result.or_df)
            eclat_result.or_df = abstract_or_clauses_df(
                eclat_result.or_df, abstraction_map, level=abstraction_level
            )
            print(f"  OR patterns {n_or_before}→{len(eclat_result.or_df)}")
            mining_stats["n_or_after_abstraction"] = len(eclat_result.or_df)

    print("--- Applying direction filter to benign mining results ---")
    if "support_diff" in eclat_df.columns:
        n_before = len(eclat_df)
        eclat_df = eclat_df[eclat_df["support_diff"] >= min_support_diff].reset_index(
            drop=True
        )
        print(
            f"  Benign itemsets: {n_before} → {len(eclat_df)} (support_diff >= {min_support_diff})"
        )
    if "support_diff" in item_seq_df.columns:
        n_before = len(item_seq_df)
        item_seq_df = item_seq_df[
            item_seq_df["support_diff"] >= min_support_diff
        ].reset_index(drop=True)
        print(
            f"  Benign sequences: {n_before} → {len(item_seq_df)} (support_diff >= {min_support_diff})"
        )
    if eclat_result.or_df is not None and not eclat_result.or_df.empty:
        if "support_diff" in eclat_result.or_df.columns:
            n_before = len(eclat_result.or_df)
            eclat_result.or_df = eclat_result.or_df[
                eclat_result.or_df["support_diff"] >= min_support_diff
            ].reset_index(drop=True)
            print(
                f"  Benign OR patterns: {n_before} → {len(eclat_result.or_df)} (support_diff >= {min_support_diff})"
            )

    eclat_df["source_label"] = "benign"
    item_seq_df["source_label"] = "benign"
    if eclat_result.or_df is not None and not eclat_result.or_df.empty:
        eclat_result.or_df["source_label"] = "benign"

    mining_stats["n_itemsets_benign_leaning"] = len(eclat_df)
    mining_stats["n_sequences_benign_leaning"] = len(item_seq_df)

    # --- Attack mining pass (binary classifier only; skip for anomaly) ---
    attack_eclat_df = pd.DataFrame()
    attack_seq_df = pd.DataFrame()
    attack_or_df: pd.DataFrame | None = None
    mining_stats["attack_pass_skipped"] = not run_attack_pass

    if run_attack_pass:
        print("--- Running Eclat itemset mining on attack (AND + AND/OR) ---")
        attack_eclat_result = run_alert_group_eclat_job(
            alert_groups_path=alert_groups_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_itemset_size,
            target_label="attack",
            run_dir=run_dir / "attack",
            jaccard_threshold=jaccard_threshold,
        )

        if has_sequence_data:
            print("--- Running PrefixSpan sequence mining on attack ---")
            attack_seq_result = run_alert_group_prefixspan_job(
                alert_groups_path=alert_groups_path,
                scenario_name=scenario,
                run_name=run_name,
                min_support=min_support,
                max_len=max_seq_len,
                target_label="attack",
                run_dir=run_dir / "attack",
                top_k_per_pass=_top_k_per_pass,
            )
            attack_seq_df = attack_seq_result.mined_df.copy()
        else:
            print(
                "  [skip] All alert_groups have empty sorted_items (pre-grouped "
                "scenario) — skipping attack sequence mining."
            )
            attack_seq_df = pd.DataFrame()

        attack_eclat_df = attack_eclat_result.mined_df.copy()

        mining_stats["n_attack_itemsets_mined"] = len(attack_eclat_df)
        mining_stats["n_attack_sequences_mined"] = len(attack_seq_df)
        mining_stats["n_attack_or_mined"] = (
            len(attack_eclat_result.or_df)
            if attack_eclat_result.or_df is not None
            else 0
        )

        if abstraction_map_path is not None:
            abstraction_map = load_abstraction_map(abstraction_map_path)
            attack_eclat_df = abstract_mined_df(
                attack_eclat_df, abstraction_map, level=abstraction_level
            )
            attack_seq_df = abstract_mined_df(
                attack_seq_df,
                abstraction_map,
                level=abstraction_level,
                column="sequence",
            )
            if (
                attack_eclat_result.or_df is not None
                and not attack_eclat_result.or_df.empty
            ):
                attack_eclat_result.or_df = abstract_or_clauses_df(
                    attack_eclat_result.or_df, abstraction_map, level=abstraction_level
                )

        print("--- Applying direction filter to attack mining results ---")
        if "support_diff" in attack_eclat_df.columns:
            n_before = len(attack_eclat_df)
            attack_eclat_df = attack_eclat_df[
                attack_eclat_df["support_diff"] >= min_support_diff
            ].reset_index(drop=True)
            print(
                f"  Attack itemsets: {n_before} → {len(attack_eclat_df)} (support_diff >= {min_support_diff})"
            )
        if "support_diff" in attack_seq_df.columns:
            n_before = len(attack_seq_df)
            attack_seq_df = attack_seq_df[
                attack_seq_df["support_diff"] >= min_support_diff
            ].reset_index(drop=True)
            print(
                f"  Attack sequences: {n_before} → {len(attack_seq_df)} (support_diff >= {min_support_diff})"
            )
        if (
            attack_eclat_result.or_df is not None
            and not attack_eclat_result.or_df.empty
        ):
            if "support_diff" in attack_eclat_result.or_df.columns:
                n_before = len(attack_eclat_result.or_df)
                attack_eclat_result.or_df = attack_eclat_result.or_df[
                    attack_eclat_result.or_df["support_diff"] >= min_support_diff
                ].reset_index(drop=True)
                print(
                    f"  Attack OR patterns: {n_before} → {len(attack_eclat_result.or_df)} (support_diff >= {min_support_diff})"
                )
            attack_or_df = attack_eclat_result.or_df

        attack_eclat_df["source_label"] = "attack"
        attack_seq_df["source_label"] = "attack"
        if attack_or_df is not None and not attack_or_df.empty:
            attack_or_df["source_label"] = "attack"

        mining_stats["n_attack_itemsets_attack_leaning"] = len(attack_eclat_df)
        mining_stats["n_attack_sequences_attack_leaning"] = len(attack_seq_df)

    print("--- Filtering mining results ---")
    if filter_config is not None:
        if not filter_config.is_absolute():
            filter_config = _ROOT / filter_config
        mining_filters = load_mining_filter_config(filter_config)

        f = mining_filters.itemsets
        eclat_df = filter_mined_itemsets(
            eclat_df,
            min_k=f.min_k,
            max_k=f.max_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )

        f = mining_filters.item_sequences
        item_seq_df = filter_mined_sequences(
            item_seq_df,
            min_k=f.min_k,
            min_support_count=f.min_support_count,
            min_abs_support_diff=f.min_abs_support_diff,
            min_confidence_attack=f.min_confidence_attack,
            max_confidence_attack=f.max_confidence_attack,
            min_confidence_benign=f.min_confidence_benign,
            min_lift=f.min_lift,
            max_overlap=f.max_overlap,
            remove_subsumed=f.remove_subsumed,
        )

        if eclat_result.or_df is not None and not eclat_result.or_df.empty:
            f = mining_filters.or_features
            eclat_result.or_df = filter_or_patterns(
                eclat_result.or_df,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                max_n_clauses=f.max_n_clauses,
            )

        # Apply the same filter to attack-pass results
        if not attack_eclat_df.empty:
            f = mining_filters.itemsets
            attack_eclat_df = filter_mined_itemsets(
                attack_eclat_df,
                min_k=f.min_k,
                max_k=f.max_k,
                min_support_count=f.min_support_count,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                max_overlap=f.max_overlap,
                remove_subsumed=f.remove_subsumed,
            )
        if not attack_seq_df.empty:
            f = mining_filters.item_sequences
            attack_seq_df = filter_mined_sequences(
                attack_seq_df,
                min_k=f.min_k,
                min_support_count=f.min_support_count,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                min_lift=f.min_lift,
                max_overlap=f.max_overlap,
                remove_subsumed=f.remove_subsumed,
            )
        if attack_or_df is not None and not attack_or_df.empty:
            f = mining_filters.or_features
            attack_or_df = filter_or_patterns(
                attack_or_df,
                min_abs_support_diff=f.min_abs_support_diff,
                min_confidence_attack=f.min_confidence_attack,
                max_confidence_attack=f.max_confidence_attack,
                min_confidence_benign=f.min_confidence_benign,
                max_n_clauses=f.max_n_clauses,
            )

        n_or_after = len(eclat_result.or_df) if eclat_result.or_df is not None else 0
        n_attack_or_after = len(attack_or_df) if attack_or_df is not None else 0
        print(
            f"  After filtering (benign): {len(eclat_df)} itemsets, "
            f"{len(item_seq_df)} sequences, {n_or_after} OR patterns"
        )
        print(
            f"  After filtering (attack): {len(attack_eclat_df)} itemsets, "
            f"{len(attack_seq_df)} sequences, {n_attack_or_after} OR patterns"
        )

    mining_stats["n_itemsets_after_filter"] = len(eclat_df)
    mining_stats["n_sequences_after_filter"] = len(item_seq_df)
    mining_stats["n_or_after_filter"] = (
        len(eclat_result.or_df) if eclat_result.or_df is not None else 0
    )

    eclat_df["mining_type"] = "itemset"
    item_seq_df = item_seq_df.rename(columns={"sequence": "itemset"})
    item_seq_df["mining_type"] = "item_sequence"
    if not attack_eclat_df.empty:
        attack_eclat_df["mining_type"] = "itemset"
    if not attack_seq_df.empty:
        attack_seq_df = attack_seq_df.rename(columns={"sequence": "itemset"})
        attack_seq_df["mining_type"] = "item_sequence"

    print("--- Constructing combined dataframe from mining results ---")
    cols_to_keep = [
        "itemset",
        "mining_type",
        "support",
        "confidence_attack",
        "confidence_benign",
        "source_label",
    ]
    or_cols_to_keep = [
        "clauses",
        "mining_type",
        "confidence_attack",
        "confidence_benign",
        "source_label",
    ]
    eclat_df = eclat_df[[c for c in cols_to_keep if c in eclat_df.columns]]
    item_seq_df = item_seq_df[[c for c in cols_to_keep if c in item_seq_df.columns]]

    dfs_to_concat = [eclat_df, item_seq_df]
    if eclat_result.or_df is not None and not eclat_result.or_df.empty:
        or_df = eclat_result.or_df[
            [c for c in or_cols_to_keep if c in eclat_result.or_df.columns]
        ]
        dfs_to_concat.append(or_df)
    if not attack_eclat_df.empty:
        dfs_to_concat.append(
            attack_eclat_df[[c for c in cols_to_keep if c in attack_eclat_df.columns]]
        )
    if not attack_seq_df.empty:
        dfs_to_concat.append(
            attack_seq_df[[c for c in cols_to_keep if c in attack_seq_df.columns]]
        )
    if attack_or_df is not None and not attack_or_df.empty:
        dfs_to_concat.append(
            attack_or_df[[c for c in or_cols_to_keep if c in attack_or_df.columns]]
        )

    combined_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
    combined_df.to_csv(
        os.path.join(run_dir, "final_combined_mining_df.csv"), index=False
    )
    n_or = len(eclat_result.or_df) if eclat_result.or_df is not None else 0
    n_attack_or = len(attack_or_df) if attack_or_df is not None else 0
    mining_stats["n_candidate_features"] = len(combined_df)
    print(
        f"  Combined {len(eclat_df) + len(attack_eclat_df)} itemsets "
        f"+ {len(item_seq_df) + len(attack_seq_df)} sequences "
        f"+ {n_or + n_attack_or} OR patterns = {len(combined_df)} candidate features"
    )

    print("--- Building and saving symbolic schema ---")
    schema_path, schema_build_stats = build_persist_and_register_symbolic_schema(
        df=combined_df,
        scenario_name=scenario,
        source_label=target_label,
        schema_name="symbolic",
        root_dir=_ROOT / "artifacts" / "features",
    )
    mining_stats.update(schema_build_stats)
    print(f"  Symbolic schema registered → {schema_path}")
    return schema_path, run_dir, mining_stats
