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
    # mining scope: 1.0 = full timeline, <1.0 = first mine_frac of transactions (sorted by time)
    mine_frac: float = 1.0
    # if True, training starts after the mine window ([mine_frac, 1-test_frac));
    # if False (default), training always starts from 0 regardless of mine_frac
    no_overlap: bool = False
    # split strategy: False = temporal (default), True = random shuffle before any split
    random_split: bool = False
    random_seed: int = 42
    # training
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base+symbolic"
    test_frac: float = 0.3
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    results_dir: Path | None = None
    alerts_json_path: Path | None = None


@dataclass
class BaselineExperimentConfig:
    scenario: str
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base"
    test_frac: float = 0.3
    # split strategy: False = temporal (default), True = random shuffle before any split
    random_split: bool = False
    random_seed: int = 42
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    results_dir: Path | None = None
    alerts_json_path: Path | None = None


@dataclass
class AnomalyExperimentConfig:
    scenario: str
    model_name: str = "iforest"
    model_version: str = "0.1.0"
    schema_name: str = "base"  # "base" or "base+symbolic"
    test_frac: float = 0.3
    mine_frac: float = 1.0
    no_overlap: bool = False
    filter_config: Path | None = None
    filter_attack_leaning: bool = (
        True  # drop attack-leaning symbolic features before encoding
    )
    abstraction_map_path: Path | None = None
    abstraction_level: int = 0
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    results_dir: Path | None = None
    alerts_json_path: Path | None = None


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
    n_mixed_dropped: int = 0
    symbolic_schema_path: Path | None = None
