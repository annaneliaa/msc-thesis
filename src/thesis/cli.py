import csv
import json
import os
from pathlib import Path

import pandas as pd
import typer

from thesis.config import (
    AlertBERTConfig,
    GroupingConfig,
    load_mining_filter_config,
    load_settings,
)
from thesis.features.manifest import initialize_feature_manifest
from thesis.features.schema_registry import FEATURE_SCHEMAS
from thesis.features.service import build_persist_and_register_symbolic_schema
from thesis.mining.itemset_mining_job import run_alert_group_eclat_job
from thesis.mining.sequence_mining_job import run_alert_group_prefixspan_job
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import (
    build_encoded_alert_groups_df,
    build_grouper,
    combine_mining_results,
    load_alert_rows_from_json,
    open_scenario_cache,
    run_preprocess_batch,
    save_alert_groups_json,
    save_encoded_df,
    save_snapshots_json,
    select_alert_groups,
    select_snapshots,
)
from thesis.registry.encoders import list_all_encoders
from thesis.registry.models import get_model_path, list_all_models
from thesis.schemas.dataframe_schemas import SCHEMAS
from thesis.schemas.validation import validate_dataframe
from thesis.training.service import train_model_for_schema
from thesis.utils.runs import create_run_dir

"""
Entry point to the system
"""

app = typer.Typer(help="Thesis system CLI")


@app.command()
def init() -> None:
    ensure_artifact_dirs()
    typer.echo("Artifact directories created.")


@app.command("init-scenario")
def init_scenario(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing manifest."
    ),
) -> None:
    """
    Initialise the feature schema manifest for a scenario.

    Creates artifacts/features/<scenario>/manifest.json with the base and
    base+dynamic composite schemas. Run this once per scenario before
    encoding alert_groups or training a model.
    """
    try:
        path = initialize_feature_manifest(scenario_name=scenario, overwrite=overwrite)
        typer.echo(f"Feature manifest created at: {path}")
    except FileExistsError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)


@app.command()
def show_config(config_name: str = "base.yaml") -> None:
    settings = load_settings(config_name)
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def validate(
    schema: str,
    path: str,
) -> None:
    """
    Validate a parquet file against a schema.
    """
    try:
        df = pd.read_parquet(path)
        validate_dataframe(df, schema)
        typer.echo(f"Schema '{schema}' is valid for {path}")
    except Exception as e:
        typer.echo(f"Validation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    config_name: str = "base.yaml",
) -> None:
    """
    Run full experiment pipeline (MLflow + artifacts).
    """
    try:
        # settings = load_settings(config_name)
        # result = run_experiment(settings.model_dump())
        typer.echo("Dummy run completed successfully.")
        # typer.echo(f"Run ID: {result['run_id']}")
        # typer.echo(f"Artifacts: {result['artifact_dir']}")

    except Exception as e:
        typer.echo(f"Run failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def list_models():
    models = list_all_models()
    for m in models:
        typer.echo(m)


@app.command()
def list_encoders():
    encoders = list_all_encoders()
    for e in encoders:
        typer.echo(e)


@app.command()
def list_schemas() -> None:
    for name, schema in SCHEMAS.items():
        typer.echo(f"{name}: {list(schema.keys())}")


@app.command()
def convert_alerts_to_json(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
) -> None:
    input_path = Path(f"data/alerts_csv/{scenario}_alerts.txt")
    output_dir = Path(f"artifacts/processed-data/{scenario}")
    output_path = output_dir / "alerts.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    alerts: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            alert = {
                "time": int(row["time"]),
                "signature": row["name"],
                "ip": row["ip"],
                "host": row["host"],
                "short": row["short"],
                "label": row["time_label"],
                "event_label": row["event_label"],
            }
            alerts.append(alert)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    print(f"Wrote {len(alerts)} alerts to {output_path}")


@app.command("process-batch")
def preprocess_alert_batch(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
    grouping_mode: str = typer.Option(
        "fixed_window",
        "--grouping-mode",
        "-g",
        help="Grouping method: 'fixed_window' or 'alertbert'.",
    ),
    alertbert_model_id: str = typer.Option(
        "",
        "--alertbert-model-id",
        help="AlertBERT model ID (subdirectory under models path).",
    ),
    alertbert_models_path: str = typer.Option(
        "artifacts/alertbert",
        "--alertbert-models-path",
        help="Directory containing AlertBERT saved models.",
    ),
    alertbert_delta: float = typer.Option(
        2.0, "--alertbert-delta", help="AlertBERT delta (time threshold)."
    ),
    alertbert_theta: float = typer.Option(
        6.0, "--alertbert-theta", help="AlertBERT theta (cosine scale)."
    ),
    alertbert_device: str = typer.Option(
        "cpu", "--alertbert-device", help="PyTorch device, e.g. 'cpu' or 'cuda'."
    ),
) -> None:
    """
    Run parsing -> tokenization -> cache ingestion for multiple alerts.
    Also runs grouping on the batch and ingests groups into cache.
    from one input JSON file.
    """
    print(
        f"Preprocessing alert batch for scenario '{scenario}' (grouping={grouping_mode})..."
    )
    try:
        alerts_path = Path(f"artifacts/processed-data/{scenario}/alerts.json")
        cache_path = Path(cache_dir)

        rows = load_alert_rows_from_json(alerts_path)

        grouping = GroupingConfig(
            mode=grouping_mode,
            alertbert=AlertBERTConfig(
                model_id=alertbert_model_id,
                models_path=alertbert_models_path,
                delta=alertbert_delta,
                theta=alertbert_theta,
                device=alertbert_device,
            ),
        )
        grouper = build_grouper(grouping)

        processed_count = run_preprocess_batch(
            scenario=scenario,
            cache_dir=cache_path,
            rows=rows,
            grouping_mode=grouping_mode,
            grouper=grouper,
        )

        typer.echo(f"Processed {processed_count} alerts.")
        typer.echo(f"Cache written to: {cache_path}/{scenario}")

    except Exception as e:
        typer.echo(f"Batch preprocessing failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def select_groups(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
    min_start_ts: int | None = typer.Option(None, help="Minimum window id."),
    max_end_ts: int | None = typer.Option(None, help="Maximum window id."),
    only_closed: bool = typer.Option(False, help="Select only closed windows."),
    retention_windows: int | None = typer.Option(
        None, help="How many most recent windows to keep."
    ),
    decay_factor: float = typer.Option(
        1.0, help="Decay factor for alert_group weights."
    ),
) -> None:
    """
    Query cache and build group snapshots ready to pass to mining preparation layer.
    """
    try:
        cache = open_scenario_cache(scenario, cache_dir)

        snapshots = select_snapshots(
            cache=cache,
            allowed_methods=None,
            limit=retention_windows,
            min_start_ts=min_start_ts,
            max_end_ts=max_end_ts,
            require_closed=only_closed,
        )

        typer.echo(f"Selected {len(snapshots)} groups.")
        for s in snapshots[:5]:
            typer.echo(
                f"n_alerts={s.n_alerts} "
                f"statis={s.status} "
                f"items={sorted(s.items)}"
            )

        out_path = Path(f"artifacts/cache/{scenario}/snapshots/groupsnapshots.json")
        save_snapshots_json(snapshots, out_path)
        typer.echo(f"Saved group snapshots to {out_path}")

    except Exception as e:
        typer.echo(f"Group selection failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def load_alert_groups(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
) -> None:
    """
    Load group snapshots from cache, convert to alert_groups and save back to cache.
    """
    try:
        cache = open_scenario_cache(scenario, cache_dir)
        alert_groups = select_alert_groups(cache=cache, require_closed=True)

        out_path = Path(
            f"artifacts/cache/{scenario}/alert_groups/alert_groups_raw.json"
        )
        save_alert_groups_json(alert_groups, out_path)

        typer.echo(f"Saved {len(alert_groups)} alert_groups to {out_path}")

    except Exception as e:
        typer.echo(f"AlertGroup preparation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def encode_alert_groups(
    scenario_name: str = typer.Option(..., "--scenario", "-s"),
    schema_name: str = typer.Option("base", "--schema-name"),
    schema_version: str | None = typer.Option(None, "--schema-version"),
    cache_dir: str = typer.Option("artifacts/cache"),
    top_k: int | None = typer.Option(None, "--top-k"),
) -> None:
    """
    Load group snapshots from cache, convert to alert_groups,
    load a FeatureSchema, encode alert_groups, and save row-based features.
    """
    try:
        cache = open_scenario_cache(scenario_name, cache_dir)
        alert_groups = select_alert_groups(cache=cache, require_closed=True)

        schema = FEATURE_SCHEMAS.load(
            scenario_name=scenario_name,
            schema_name=schema_name,
            schema_version=schema_version,
        )
        print(f"Loaded schema {schema_name} for scenario {scenario_name}.")

        df = build_encoded_alert_groups_df(alert_groups, schema, top_k)

        out_dir = Path(f"artifacts/cache/{scenario_name}/alert_groups")
        parquet_path, csv_path = save_encoded_df(df, schema_name, out_dir)

        typer.echo(
            f"Saved {len(alert_groups)} encoded alert_groups under schema "
            f"'{schema_name}' version '{schema.schema_version}' to "
            f"{parquet_path} and {csv_path}"
        )

    except Exception as e:
        typer.echo(f"AlertGroup encoding failed: {e}")
        raise typer.Exit(code=1)


@app.command("mine")
def mine_alert_groups(
    scenario: str = typer.Option(
        "debug_scenario",
        help="Dataset scenario name for logging and artifacts.",
    ),
    run_name: str = typer.Option(
        "debug",
        help="MLflow run name.",
    ),
    min_support: float = typer.Option(
        0.05,
        help="Minimum support threshold.",
    ),
    max_itemset_size: int = typer.Option(
        3,
        help="Maximum itemset size.",
    ),
    max_seq_len: int = typer.Option(
        5,
        help="Maximum sequence length.",
    ),
    target_label: str = typer.Option(
        "benign",
        help="Label to mine from.",
    ),
    filter_config: Path | None = typer.Option(
        None,
        "--filter-config",
        help="Path to a YAML filter config file. If omitted, no post-mining filters are applied.",
    ),
) -> None:
    """
    Load cached AlertGroups and run alert_group-level Eclat and PrefixSpan mining.
    """
    alert_groups_path = Path(
        f"artifacts/cache/{scenario}/alert_groups/alert_groups_raw.json"
    )

    mining_filters = None
    if filter_config is not None:
        if not filter_config.exists():
            typer.echo(f"Filter config not found: {filter_config}")
            raise typer.Exit(code=1)
        mining_filters = load_mining_filter_config(filter_config)
        typer.echo(f"Loaded filter config from {filter_config}.")

    try:
        typer.echo(f"Loading alert_groups from {alert_groups_path}...")
        tx_path = Path(alert_groups_path)
        run_dir = create_run_dir(run_name)

        eclat_result = run_alert_group_eclat_job(
            alert_groups_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_itemset_size,
            target_label=target_label,
            run_dir=run_dir,
        )
        typer.echo(f"Eclat mining complete. Artifacts saved to: {eclat_result.run_dir}")

    except Exception as e:
        typer.echo(f"AlertGroup mining failed: {e}")
        raise typer.Exit(code=1)

    try:
        typer.echo("Running PrefixSpan item sequence mining...")
        item_seq_result = run_alert_group_prefixspan_job(
            alert_groups_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_seq_len,
            target_label=target_label,
            run_dir=run_dir,
        )
        typer.echo(
            f"Item sequence mining complete. Artifacts saved to: {item_seq_result.run_dir}"
        )

    except Exception as e:
        typer.echo(f"Sequence mining failed: {e}")
        raise typer.Exit(code=1)

    try:
        typer.echo("Building combined feature schema from all mining results...")

        combined_df, n_itemsets, n_sequences = combine_mining_results(
            eclat_result.mined_df,
            item_seq_result.mined_df,
            mining_filters,
        )

        if mining_filters is not None:
            typer.echo(f"Itemsets after filtering: {n_itemsets}")
            typer.echo(f"Item sequences after filtering: {n_sequences}")

        combined_df.to_csv(os.path.join(run_dir, "combined_mining_df.csv"), index=False)

        typer.echo(
            f"Combined {n_itemsets} itemsets, {n_sequences} item sequences, "
            f"into {len(combined_df)} features."
        )

        schema_path, _ = build_persist_and_register_symbolic_schema(
            df=combined_df,
            scenario_name=eclat_result.scenario_name,
            source_label=eclat_result.target_label,
            schema_name="symbolic",
        )
        typer.echo(f"Symbolic feature schema written to: {schema_path}")

    except Exception as e:
        typer.echo(f"Schema building failed: {e}")
        raise typer.Exit(code=1)


@app.command("train-model")
def train_model_cmd(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
    label_col: str = typer.Option(..., help="Name of label column"),
    scenario_name: str = typer.Option(..., "--scenario", "-s"),
    schema_name: str = typer.Option(..., help="Feature schema name"),
    schema_version: str | None = typer.Option(
        None,
        help="Feature schema version. If omitted, latest symbolic version is used.",
    ),
    model_name: str = typer.Option("logreg", help="Model name"),
    model_version: str = typer.Option("0.1.0", help="Model version"),
    test_frac: float = typer.Option(0.3, help="Test split fraction"),
):
    """
    Train a model for a versioned feature schema and persist it.
    """

    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset not found: {dataset_path}")

    typer.echo(f"Loading dataset from {dataset_path}...")

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix in {".csv", ".txt"}:
        df = pd.read_csv(dataset_path)
    else:
        raise typer.BadParameter("Only .csv, .txt, .parquet are supported")

    if label_col not in df.columns:
        raise typer.BadParameter(f"Label column '{label_col}' not in dataset")

    y = df[label_col].map({"benign": 0, "attack": 1})

    if y.isna().any():
        raise typer.BadParameter(
            f"Label column '{label_col}' must only contain 'benign' and 'attack'."
        )

    X = df.drop(columns=[label_col])

    typer.echo("Loading feature schema...")

    schema = FEATURE_SCHEMAS.load(
        scenario_name=scenario_name,
        schema_name=schema_name,
        schema_version=schema_version,
    )

    output_dir = get_model_path(scenario_name, model_name, model_version)

    if output_dir.exists():
        raise typer.BadParameter(f"Model version already exists: {output_dir}")

    typer.echo("Starting training process...")

    summary = train_model_for_schema(
        X=X,
        y=y,
        schema=schema,
        model_name=model_name,
        model_version=model_version,
        output_dir=output_dir,
        test_frac=test_frac,
    )

    typer.echo("\nTraining completed:")
    typer.echo(f"  Model: {summary.model_name}")
    typer.echo(f"  Model version: {summary.model_version}")
    typer.echo(f"  Schema: {summary.schema_name}")
    typer.echo(f"  Schema version: {summary.schema_version}")
    typer.echo(f"  AUC: {summary.auc:.4f}")
    typer.echo(f"  Features: {summary.n_features}")
    typer.echo(f"  Output dir: {summary.output_dir}")
