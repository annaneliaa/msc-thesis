from dataclasses import dataclass, field
from typing import Literal
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
    # (granularity, window) pairs are independent (each mines/fits/evaluates
    # entirely on its own window), so they run concurrently on a thread pool
    # -- see TemporalDecayConfig.n_jobs for why threads, not processes.
    n_jobs: int = 4


@dataclass
class TemporalDecayConfig:
    """Experiment 2 (Temporal Generalization / Fixed-Horizon Decay): for each
    shortlisted (feature_set, mining_setting, granularity, model) config,
    mine a schema and fit a model once on window 0's train split (W_src is
    always the first window -- no other source-window roles), freeze
    schema/model/threshold, then walk forward one window at a time to the
    end of the timeline, evaluating the frozen model on each window in turn
    (h=0 is W_src's own held-out test split; h=1..n_windows-1 are fully
    external windows). SHAP + LIME importances are tracked at every horizon
    step so feature-attribution drift is visible alongside the metric decay.
    See experiments/temporal_decay.py."""

    scenario: str
    shortlist_path: Path
    # W_src's internal train/test split -- mining + fitting only ever see the
    # train side; h=0 is scored on the held-out test side.
    train_frac_within_window: float = 0.7
    mining_settings_path: Path = field(
        default_factory=lambda: Path(
            "src/thesis/configs/screening_mining_settings.yaml"
        )
    )
    # how the frozen decision threshold is chosen from W_src's own train-split
    # data, once, before any horizon is evaluated -- never recalibrated per
    # horizon
    threshold_mode: Literal["fixed", "calibrated_recall"] = "fixed"
    calibrated_recall_target: float = 0.90
    # SHAP/LIME are expensive (LIME especially -- one perturbed-sample
    # predict_proba batch per explained row) -- off switch for a metrics-only
    # run.
    compute_explanations: bool = True
    # rows sampled (once, from W_src's train split) as the SHAP/LIME
    # background/reference set, frozen and reused at every horizon so
    # attribution drift reflects the target window changing, not the
    # background.
    explain_background_n: int = 100
    # rows sampled from each horizon's window to explain
    explain_sample_n: int = 50
    # perturbed samples LIME draws per explained row (LimeTabularExplainer's
    # own default is 5000; kept lower here since it's paid n_windows x
    # explain_sample_n times per config)
    lime_num_samples: int = 1000
    top_n_importances: int = 30
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    alerts_json_path: Path | None = None
    results_dir: Path | None = None
    force_remine: bool = False
    random_seed: int = 42
    # Shortlisted configs are independent (each mines/fits/evaluates on its own
    # windows), so they run concurrently on a thread pool -- threads, not
    # processes, since the dominant per-config cost (BLAS ops inside the LogReg
    # fit, vectorized pandas/numpy encoding) releases the GIL, and threads avoid
    # having to re-pickle the multi-million-row alert_groups list per worker
    # (macOS's default 'spawn' start method would otherwise pay that cost for
    # every process). Kept modest by default to avoid oversubscribing cores
    # against BLAS's own internal threading within each fit.
    n_jobs: int = 4


@dataclass
class RollingWalkForwardConfig:
    """Experiment 3 (Rolling / Walk-Forward Evaluation): for each shortlisted
    (feature_set, mining_setting, granularity, model) config, walk i = 0 ..
    n(g)-2 across the timeline. At each step, mine a schema and fit a model
    from scratch on the *full* window Wi (no held-out split within Wi --
    unlike screening_sweep/temporal_decay, the held-out evaluation set here
    is the disjoint window Wi+1, so there's no reason to withhold part of Wi
    itself), decide a threshold from Wi's own in-sample scores, evaluate on
    W(i+1), then discard the schema/model and move on -- no accumulation, no
    state carried between steps. This is the "always retrain" anchor
    contrasted against Experiment 2's "never retrain" frozen-model decay
    curve. See experiments/rolling_walk_forward.py."""

    scenario: str
    shortlist_path: Path
    mining_settings_path: Path = field(
        default_factory=lambda: Path(
            "src/thesis/configs/screening_mining_settings.yaml"
        )
    )
    # how the decision threshold is chosen from each step's own (in-sample)
    # training scores -- same method at every step (locked in once, per the
    # experiment spec), recomputed fresh each step since the model itself is
    # refit each step; "fixed" makes the recomputation a no-op (always 0.5).
    threshold_mode: Literal["fixed", "calibrated_recall"] = "fixed"
    calibrated_recall_target: float = 0.90
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    alerts_json_path: Path | None = None
    results_dir: Path | None = None
    force_remine: bool = False
    random_seed: int = 42
    # Shortlisted configs are independent (each mines/fits/evaluates entirely
    # on its own windows), so they run concurrently on a thread pool -- see
    # TemporalDecayConfig.n_jobs for why threads, not processes.
    n_jobs: int = 4


@dataclass
class MonitorDriftConfig:
    """Experiment 4 (Drift-Monitor Evaluation, observe-only): for each
    shortlisted (feature_set, mining_setting, granularity, model) config,
    mine a schema and fit a model once on window 0's train split (same
    freeze-and-decay design as TemporalDecayConfig), plus -- for symbolic
    configs -- build a deployment-scoped DynamicSchema (Vk) from that same
    mining pass. Walk the frozen schema/model/threshold forward one window
    at a time, and at every horizon also run the drift monitor
    (thesis.monitor.monitor.run_monitor_window) against the frozen Vk over
    that horizon's raw incoming alert groups, logging every signal and every
    alarm it raises. The monitor is observe-only here: nothing is ever
    actually re-mined or retrained, no matter what action it reports. See
    experiments/monitor_drift.py."""

    scenario: str
    shortlist_path: Path
    train_frac_within_window: float = 0.7
    mining_settings_path: Path = field(
        default_factory=lambda: Path(
            "src/thesis/configs/screening_mining_settings.yaml"
        )
    )
    threshold_mode: Literal["fixed", "calibrated_recall"] = "fixed"
    calibrated_recall_target: float = 0.90
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    alerts_json_path: Path | None = None
    results_dir: Path | None = None
    random_seed: int = 42
    # Shortlisted configs are independent, so they run concurrently on a
    # thread pool -- see TemporalDecayConfig.n_jobs for why threads, not
    # processes.
    n_jobs: int = 4
    # Passed straight through to run_monitor_window at every horizon.
    monitor_consecutive_windows: int = 3
    monitor_min_samples_signal_2: int = 30


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
