from dataclasses import dataclass, field
from thesis.schemas.mining import AttributeMiningConfig, FeatureSelectionConfig
from thesis.paths import CACHE_DIR
from pathlib import Path
from thesis.config import GroupingConfig


@dataclass
class SymbolicExperimentConfig:
    scenario: str
    # mining strategy: "cooccurrence" (default, existing Eclat/PrefixSpan
    # cross-signature/cross-alert basket mining) or "attribute" (per-alert-group
    # contrast-set + decision-tree rule mining -- see mining/attribute_mining_job.py)
    mining_strategy: str = "cooccurrence"
    attribute_mining_config: AttributeMiningConfig = field(
        default_factory=AttributeMiningConfig
    )
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
    # mining scope: 1.0 = full timeline, <1.0 = first mine_frac of alert_groups (sorted by time)
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
    # None (default) = train on everything before the test split (old behavior).
    # Set to reproduce a published "first N / rest" split as a fraction of the
    # full timeline, e.g. train_frac=0.1 + test_frac=0.9 for CSCAS's paper
    # (6 of 60 days train, remainder test). See training/util.effective_train_start.
    train_frac: float | None = None
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    results_dir: Path | None = None
    alerts_json_path: Path | None = None
    prebuilt_symbolic_schema_path: Path | None = None


@dataclass
class BaselineExperimentConfig:
    scenario: str
    model_name: str = "logreg"
    model_version: str = "0.1.0"
    schema_name: str = "base"
    test_frac: float = 0.3
    train_frac: float | None = None
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
    model_name: str = "bernoulli_oc"
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
    prebuilt_symbolic_schema_path: Path | None = None


@dataclass
class ScreeningSweepConfig:
    """In-Window Baseline (Screening Sweep): for a grid of
    (mining_setting x granularity), mine a schema inside every chronological
    window using only that window's train split, then train/evaluate cheap
    screening models purely within that window. See
    experiments/screening_sweep.py."""

    scenario: str
    granularities: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.33, 0.5, 1.0]
    )
    mining_settings_path: Path = field(
        default_factory=lambda: Path(
            "src/thesis/configs/screening_mining_settings.yaml"
        )
    )
    models: list[str] = field(default_factory=lambda: ["logreg"])
    baseline_models: list[str] = field(default_factory=lambda: ["logreg"])
    # fraction of each window (chronologically first) used for train; the
    # remaining fraction is the window's held-out test split
    train_frac_within_window: float = 0.7
    # None = evaluate every window (default, per the screening sweep spec);
    # else N evenly-spaced window indices per granularity (documented
    # fallback for when the full sweep is computationally infeasible)
    windows_per_gran: int | None = None
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    alerts_json_path: Path | None = None
    results_dir: Path | None = None
    force_remine: bool = False
    random_seed: int = 42


@dataclass
class ExperimentResult:
    scenario: str
    model_name: str
    model_version: str
    schema_name: str
    schema_version: str
    auc: float
    n_alert_groups: int
    n_features: int
    metrics: dict
    results_file: Path
    grouping_mode: str
    n_mixed_dropped: int = 0
    symbolic_schema_path: Path | None = None
    mining_run_dir: Path | None = None
