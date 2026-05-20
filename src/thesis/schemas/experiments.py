from dataclasses import dataclass, field
from thesis.schemas.mining import FeatureSelectionConfig
from thesis.paths import CACHE_DIR
from pathlib import Path
from thesis.config import GroupingConfig


@dataclass
class SymbolicExperimentConfig:
    scenario: str
    # mining
    min_support: float = 0.05
    max_itemset_size: int = 3
    max_seq_len: int = 5
    target_label: str = "benign"
    filter_config: Path | None = None
    jaccard_threshold: float = 0.98
    abstraction_map_path: Path | None = None
    abstraction_level: int = 0  # 0 = mid-level (recommended set up), 1 = coarse
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    # training
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base+symbolic"
    test_frac: float = 0.3
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)


@dataclass
class BaselineExperimentConfig:
    scenario: str
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base"
    test_frac: float = 0.3
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)


@dataclass
class ExperimentResult:
    scenario: str
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    auc: float
    n_transactions: int
    n_features: int
    metrics: dict
    results_file: Path
    grouping_mode: str
    symbolic_schema_path: Path | None = None
