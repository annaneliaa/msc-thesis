# MSc Thesis: Hybrid Symbolic-Neural System for Intrusion Detection

A neural-symbolic (NeSy) system for intrusion detection with focus on reducing false positives (FP). This project combines machine learning models with symbolic reasoning for improved anomaly detection.

## Project Overview

- **Language**: Python 3.10+
- **Framework**: FastAPI + Uvicorn (API), Typer (CLI)
- **ML Stack**: PyTorch, Transformers, Scikit-learn, MLflow
- **Data**: Hybrid security datasets (AIT_ADS with Aminer and Wazuh alerts)
- **Deployment**: Docker containerization support

## Installation

### Development Setup

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.10
- Dependencies include: FastAPI, PyTorch, Transformers, MLflow, Pandas, and more (see `pyproject.toml`)

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

## Configuration

Configuration files are located in `configs/`:
- `base.yaml` - Base configuration
- `dev.yaml` - Development overrides
- `prod.yaml` - Production overrides

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

## Key Features

- **Hybrid Architecture**: Combines neural models (PyTorch/Transformers) with symbolic reasoning
- **MLflow Integration**: Experiment tracking and model management
- **Multiple Baselines**: Reference implementations for comparison
- **Flexible Configuration**: YAML-based config with environment overrides
- **REST API**: FastAPI-based interface for inference
- **Command-line Interface**: Typer-based CLI for training and analysis
- **Mining Pipeline**: Alert mining and preprocessing from raw security data

## License

MSc Thesis - TU Delft
