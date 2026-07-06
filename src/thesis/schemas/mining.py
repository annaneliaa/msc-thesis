from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


class ItemsetFilterConfig(BaseModel):
    min_k: int = 1
    max_k: int | None = None
    min_support_count: int = 10
    min_abs_support_diff: float = 0.0
    min_confidence_attack: float = 0.0
    max_confidence_attack: float | None = None
    min_confidence_benign: float = 0.0
    max_overlap: float | None = None
    remove_subsumed: bool = True


class SequenceFilterConfig(BaseModel):
    min_k: int = 3
    min_support_count: int = 10
    min_abs_support_diff: float = 0.0
    min_confidence_attack: float = 0.0
    max_confidence_attack: float | None = None
    min_confidence_benign: float = 0.0
    min_lift: float | None = None
    max_overlap: float | None = None
    remove_subsumed: bool = True
    top_k_per_pass: int | None = None


class FeatureSelectionConfig(BaseModel):
    top_k: int | None = None
    min_utility_score: float | None = None
    filter_cross_host_or: bool = False


class OrFilterConfig(BaseModel):
    min_abs_support_diff: float = 0.0
    min_confidence_attack: float = 0.0
    max_confidence_attack: float | None = None
    min_confidence_benign: float = 0.0
    max_n_clauses: int | None = None


class MiningFilterConfig(BaseModel):
    itemsets: ItemsetFilterConfig = ItemsetFilterConfig()
    item_sequences: SequenceFilterConfig = SequenceFilterConfig()
    itemset_sequences: SequenceFilterConfig = (
        SequenceFilterConfig()
    )  # defined but not yet applied
    or_features: OrFilterConfig = OrFilterConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()


class ContrastSetFilterConfig(BaseModel):
    """Step 1 (contrast-set mining over categorical predicates) thresholds."""

    min_attack_coverage: float = 0.05
    min_benign_coverage: float = 0.05
    min_growth_rate: float = 3.0
    max_p_value: float | None = None


class DecisionTreeRuleConfig(BaseModel):
    """Step 2 (decision-tree rule extraction) hyperparameters."""

    max_depth: int = 4
    min_samples_leaf: int = 20
    class_weight: str | None = "balanced"
    random_state: int = 0
    # Guards against class_weight="balanced" floating-point noise: a node that
    # is truly 100% pure can still show impurity ~1e-13 (rounding error from
    # weighted Gini over many samples) instead of exactly 0, which sits above
    # sklearn's internal near-zero cutoff (~2.22e-16) and so isn't auto-stopped
    # -- the tree then "splits" a zero-information node just to shave that
    # noise down to 0, producing a rule with no real discriminative content.
    # 1e-9 is far above that noise floor but far below any real split's
    # impurity decrease (see decision_tree_rule_mining.fit_rule_tree).
    min_impurity_decrease: float = 1e-9


class AttributeMiningConfig(BaseModel):
    """Config for the two-stage per-alert-group attribute mining pipeline."""

    contrast: ContrastSetFilterConfig = ContrastSetFilterConfig()
    tree: DecisionTreeRuleConfig = DecisionTreeRuleConfig()


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
    n_alert_groups: Optional[int] = None

    # config traceability
    config_name: Optional[str] = None


@dataclass(slots=True)
class MiningAlertGroup:
    """
    Canonical input record for the mining module.

    This is the alert_group-level representation consumed by itemset mining.
    It is independent from preprocessing/cache schemas.
    Mining requires a label for the alert_group (e.g. "benign" or "attack") and a set of items.
    """

    alert_group_id: int | str
    group_label: str
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
    or_df: pd.DataFrame | None = None
