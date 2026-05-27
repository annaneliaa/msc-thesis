# MSc Thesis: Symbolic Knowledge Mining System for Intrusion Detection

A neural-symbolic (NeSy) system for intrusion detection in SOCs with focus on reducing false positives. This project combines machine learning models with symbolic reasoning for modelling benign behaviour to separate true attacks from false positives.

## Project Overview

- **Language**: Python 3.10+
- **Framework**: FastAPI + Uvicorn (API), Typer (CLI)
- **ML Stack**: PyTorch, Transformers, Scikit-learn, MLflow
- **Data**: Hybrid security datasets (AIT_ADS with Aminer and Wazuh alerts)
- **Deployment**: Docker containerization support

## Installation

### Environment Setup

```bash
conda activate thesis
```

### Development Install

```bash
# Install the package in editable mode (run from repo root)
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.10 (managed via the `thesis` conda environment)
- Dependencies include: PyTorch, Transformers, Scikit-learn, Pandas, and more (see `pyproject.toml`)

## Project Structure

```
src/thesis/
├── api/              # FastAPI REST endpoints
├── cli.py            # Command-line interface
├── config.py         # Configuration management
├── data/             # Data loading and processing
├── preprocessing/    # Data preprocessing pipelines
├── features/         # Feature engineering
├── encoders/         # Model encoders
├── training/         # Model training logic
├── inference/        # Inference pipelines
├── mining/           # Alert mining and analysis
├── baselines/        # Baseline model implementations
├── schemas/          # Pydantic data schemas
├── experiments/      # Experimental code
├── visualization/    # Plotting and analysis tools
└── ontology/         # Domain ontology definitions
```

## Quick Start

### CLI Commands

```bash
# Show available commands
thesis-cli --help

# Initialize configuration
thesis-cli init

# Display current configuration
thesis-cli show-config

# Run mining with a specific name
thesis-cli mine --run-name debug
```

### Run API Server

```bash
# Development mode with auto-reload
uvicorn thesis.api.main:app --reload

# Production mode
uvicorn thesis.api.main:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/path/to/test_file.py
```

## AlertBERT Grouper

AlertBERT is a pre-trained masked language model that embeds alerts into a vector space and clusters them into groups. Pre-trained checkpoints live in `external/AlertBERT/saved_models/`. No local training is needed — the models are used for inference only.

### Configure

Point `configs/alertbert_grouping.yaml` at the checkpoint you want to use:

```yaml
alertbert:
  model_id: mlm_1l_1h_16d_original_1_60k   # directory name under saved_models/
  models_path: external/AlertBERT/saved_models
  delta: 2.0      # time-gap threshold (seconds) for pre-clustering
  theta: 6.0      # cosine-distance scale factor (>= delta); higher = more semantic splitting
  padding: 1024   # context alerts added on each side of an embedding chunk
  readout: 2048   # alerts per embedding chunk (padding + readout ≤ model context window)
  device: cpu     # or cuda
```

`delta` and `theta` control the grouping sensitivity:
- **delta** — maximum combined distance (time + semantic) to merge two alerts. Also used as the pre-clustering time gap.
- **theta** — scales the semantic (cosine) component. When `theta == delta` embeddings are ignored; increasing `theta` relative to `delta` allows semantically different alerts to be split even when they are temporally close.

### Embedding and chunking strategy

AlertBERT embeds the full alert sequence of a scenario in order. Because the upstream clustering step builds a pairwise distance matrix (O(n²) memory), the sequence is processed in **6-hour time windows**. Within each window, if the alert count exceeds 30 000, the window is further divided into equal-sized **sub-chunks**.

Each forward pass through the transformer sees `readout + 2 × padding` alerts. The model embeds all of them, but only the central `readout` embeddings are kept — the padding alerts on each side are discarded. This ensures boundary alerts in each chunk still see neighbours on both sides, giving them richer contextual representations. For scenarios smaller than `readout`, chunking is skipped and everything is embedded in a single pass.

### Model inputs vs. transaction item content

AlertBERT uses only three fields from each `TokenizedAlert` to decide how to group alerts:

| Field | Role |
|---|---|
| `short` | Detector/rule short name — looked up in the model's vocab |
| `host` | Host name — looked up in the model's vocab |
| `ts` | Unix timestamp — used as `raw_time` for time-gap pre-clustering |

The whitelisted signature tokens (`sig:login`, `sig:failed`, etc., derived from `alert.name` by `tokenization.py`) are **not** used by AlertBERT to determine group membership. They exist only in `alert.tokens`.

However, signature tokens **do** end up in the resulting transactions for both grouping methods. After grouping, `cache_ingestor.py` builds each group's itemset by unioning `alert.tokens` across all member alerts. Since both the fixed-2s grouper and AlertBERT operate on the same `TokenizedAlert` objects — with the same `tokens` field — the transaction item content is symmetric. The only difference between the two methods is which alerts are assigned to the same group.

### Run

AlertBERT grouping is invoked as part of the preprocessing pipeline. Configure the grouping method in the experiment config and run via the CLI or experiment scripts:

```bash
# Preprocess a scenario with AlertBERT grouping
thesis-cli preprocess <scenario> --grouping alertbert
```

## Visualization

### EDA Plots
```
# All scenarios (PDF, default output dir)
python src/thesis/scripts/run_eda_plots.py --all

# Specific scenarios
python src/thesis/scripts/run_eda_plots.py fox harrison wheeler

# Custom options
python src/thesis/scripts/run_eda_plots.py --all \
  --out-dir plots/eda \
  --fmt png \
  --bin-hours 2 \
  --top-k 25
```

**Flags:**

|Flag|Default|Description|
|---|---|---|
|`--all`|—|include all 8 scenarios|
|`--data-dir`|`data/alerts_csv`|path to the CSV directory|
|`--out-dir`|`artifacts/experiments/plots/eda`|where to save figures|
|`--fmt`|`pdf`|`pdf`, `png`, or `svg`|
|`--bin-hours`|`1.0`|histogram bin width for the volume plot|
|`--top-k`|`20`|number of signatures in the top-names plot|

## Configuration

Configuration files are located in `configs/`:
- `base.yaml` - Base configuration
- `dev.yaml` - Development overrides
- `prod.yaml` - Production overrides
- `alertbert_grouping.yaml` - AlertBERT grouper runtime settings (model, delta, theta, chunking)

Key configuration includes:
- App name, environment, host, and port
- Artifact directories (models, mining, runs)
- Model settings

## Docker

Build and run the application in Docker:

```bash
# Build the image
docker build -t msc-thesis:latest .

# Run the container
docker run -p 8000:8000 msc-thesis:latest
```

The container exposes port 8000 for the FastAPI application.

## Data

Datasets are located in `data/`:
- `ait_ads/` - Alert dataset with:
  - Aminer and Wazuh JSON data
  - Alert labels and timestamps
  - Multiple subject profiles (fox, harrison, santos, shaw, wheeler, wilson, etc.)
- `abstraction_map.json` - Mapping for alert abstraction
- `alerts_csv/` - Processed alert data in CSV format (using this one)

## Artifacts & Runs

- `artifacts/` - Generated models, features, encoders, and experimental results
- `mlruns/` - MLflow experiment tracking data
- `logs/` - Application and mining logs

## Development

### Code Quality Tools
- **Formatter**: Black
- **Linter**: Ruff
- **Pre-commit hooks**: Available via `pre-commit`

### Testing
- Framework: Pytest
- Location: `tests/`
- Coverage includes: API, mining, preprocessing, and other modules

### Notebooks
Jupyter notebooks available in `src/thesis/notebooks/` for exploratory analysis and visualization.

## Main Features

- **Hybrid Architecture**: Combines neural models (PyTorch/Transformers) with symbolic reasoning
- **MLflow Integration**: Experiment tracking and model management
- **Multiple Baselines**: Reference implementations for comparison
- **Flexible Configuration**: YAML-based config with environment overrides
- **REST API**: FastAPI-based interface for inference
- **Command-line Interface**: Typer-based CLI for training and analysis
- **Mining Pipeline**: Alert mining and preprocessing from raw security data

## License

MSc Thesis - TU Delft
