import typer
import pandas as pd
from pathlib import Path
import json
import csv

from thesis.schemas.dataframe_schemas import SCHEMAS
from thesis.config import load_settings
from thesis.mining.mining_transaction_job import run_transaction_eclat_job
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.validation import validate_dataframe
from thesis.registry.models import list_all_models, get_model_path
from thesis.registry.encoders import list_all_encoders
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.preprocessing.service import process_alert_batch, select_groups_from_cache
from thesis.preprocessing.mining_prep import build_transactions
from thesis.training.service import train_model_for_schema
from thesis.encoder.service import encode_transactions

"""
Entry point to the system
"""

app = typer.Typer(help="Thesis system CLI")


@app.command()
def init() -> None:
    ensure_artifact_dirs()
    typer.echo("Artifact directories created.")


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
                "name": row["name"],
                "ip": row["ip"],
                "host": row["host"],
                "short": row["short"],
                "time_label": row["time_label"],
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
) -> None:
    """
    Run parsing -> tokenization -> cache ingestion for multiple alerts.
    Also runs grouping on the batch and ingests groups into cache.
    from one input JSON file.
    """
    print(f"Preprocessing alert batch for scenario '{scenario}'...")
    try:
        alerts_path = Path(f"artifacts/processed-data/{scenario}/alerts.json")
        cache_path = Path(cache_dir)

        with alerts_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError("Input file must contain a JSON list of alert objects.")

        cache = TokenCache(cache_dir=cache_path, scenario=scenario)
        cache_ingestor = CacheIngestor(cache=cache)

        processed_count = process_alert_batch(
            rows=payload, scenario=scenario, ingestor=cache_ingestor
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
        1.0, help="Decay factor for transaction weights."
    ),
) -> None:
    """
    Query cache and build group snapshots ready to pass to mining preparation layer.
    """
    try:
        cache = TokenCache(cache_dir=Path(cache_dir), scenario=scenario)

        snapshots = select_groups_from_cache(
            cache=cache,
            allowed_methods=None,  # TODO: add option to filter by method
            limit=retention_windows,
            min_start_ts=min_start_ts,  # TODO: add option to filter by time range
            max_end_ts=max_end_ts,
            require_closed=only_closed,
        )

        typer.echo(f"Selected {len(snapshots)} groups.")
        for s in snapshots[:5]:  # print first 5 transactions as sample
            typer.echo(
                f"n_alerts={s.n_alerts} "
                f"statis={s.status} "
                f"items={sorted(s.items)}"
            )

        out_dir = Path(f"artifacts/cache/{scenario}/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir / "groupsnapshots.json"
        )  # TODO: check if this means there is always one transactions file in cache per scenario

        # convert to serializable format
        serialized = [
            {
                "group_id": s.group_id,  # transaction_id is just window_id for now
                "method": s.method,
                "version": s.version,
                "start_ts": s.start_ts,
                "end_ts": s.end_ts,
                "alert_ids": s.alert_ids,
                "n_alerts": s.n_alerts,
                "items": sorted(list(s.items)),
                "alert_ips": sorted(list(s.alert_ips)),
                "tx_label": s.tx_label,
                "alert_labels": (
                    sorted(list(s.alert_labels)) if s.alert_labels is not None else None
                ),
                "status": s.status,
            }
            for s in snapshots
        ]

        with open(out_path, "w") as f:
            json.dump(serialized, f, indent=2)

        typer.echo(f"Saved group snapshots to {out_path}")

    except Exception as e:
        typer.echo(f"Group selection failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def build_transactions_json(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
) -> None:
    """
    Load group snapshots from cache, convert to transactions and save back to cache.
    """
    try:
        cache = TokenCache(cache_dir=Path(cache_dir), scenario=scenario)

        snapshots = select_groups_from_cache(
            cache=cache,
            allowed_methods=None,
            limit=None,
            min_start_ts=None,
            max_end_ts=None,
            require_closed=True,
        )

        transactions = build_transactions(snapshots)

        out_dir = Path(f"artifacts/cache/{scenario}/transactions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transactions.json"

        # convert to serializable format
        # TODO: put in format ready for training + inference
        serialized = [
            {
                "transaction_id": t.transaction_id,
                "group_id": t.group_id,
                "method": t.method,
                "start_ts": t.start_ts,
                "end_ts": t.end_ts,
                "n_alerts": t.n_alerts,
                "alert_ids": t.alert_ids,
                "abs_items": sorted(list(t.abs_items)),
                "raw_items": sorted(list(t.raw_items)),
                "alert_ips": sorted(list(t.alert_ips)),
                "tx_label": t.tx_label,
                "alert_labels": (
                    sorted(list(t.alert_labels)) if t.alert_labels is not None else None
                ),
                "weight": t.weight,
            }
            for t in transactions
        ]

        with open(out_path, "w") as f:
            json.dump(serialized, f, indent=2)

        typer.echo(f"Saved {len(transactions)} transactions to {out_path}")

    except Exception as e:
        typer.echo(f"Transaction preparation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def build_row_transactions(
    scenario: str = typer.Option(..., "--scenario", "-s"),
    schema_name: str = typer.Option("baseline", "--schema-name"),
    cache_dir: str = typer.Option("artifacts/cache"),
) -> None:
    """
    Load group snapshots from cache, convert to transactions,
    encode them under a schema, and save in row-based format.
    """
    try:
        cache = TokenCache(cache_dir=Path(cache_dir), scenario=scenario)

        snapshots = select_groups_from_cache(
            cache=cache,
            allowed_methods=None,
            limit=None,
            min_start_ts=None,
            max_end_ts=None,
            require_closed=True,
        )

        transactions = build_transactions(snapshots)

        feature_df = encode_transactions(
            transactions=transactions,
            schema_name=schema_name,
        )

        meta_rows = []
        for t in transactions:
            meta_rows.append(
                {
                    "transaction_id": t.transaction_id,
                    "group_id": t.group_id,
                    "method": t.method,
                    "start_ts": t.start_ts,
                    "end_ts": t.end_ts,
                    "n_alerts": t.n_alerts,
                    "tx_label": t.tx_label,
                    "weight": t.weight,
                }
            )

        meta_df = pd.DataFrame(meta_rows)

        df = pd.concat(
            [meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
            axis=1,
        )

        out_dir = Path(f"artifacts/cache/{scenario}/transactions")
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / f"transactions_{schema_name}.parquet"
        csv_path = out_dir / f"transactions_{schema_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        typer.echo(
            f"Saved {len(transactions)} encoded transactions under schema "
            f"'{schema_name}' to {parquet_path} and {csv_path}"
        )

    except Exception as e:
        typer.echo(f"Row transactions preparation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def mine_transactions(
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
    max_len: int = typer.Option(
        3,
        help="Maximum itemset size.",
    ),
    target_label: str = typer.Option(
        "benign",
        help="Label to mine from.",
    ),
) -> None:
    """
    Load cached Transactions and run transaction-level Eclat mining.
    """
    transactions_path = Path(
        f"artifacts/cache/{scenario}/transactions/transactions.json"
    )
    try:
        typer.echo(f"Loading transactions from {transactions_path}...")
        tx_path = Path(transactions_path)

        path = run_transaction_eclat_job(
            transactions_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_len,
            target_label=target_label,
        )

        typer.echo(f"Transaction mining output written to: {path}")

    except Exception as e:
        typer.echo(f"Transaction mining failed: {e}")
        raise typer.Exit(code=1)


@app.command("train-model")
def train_model_cmd(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
    label_col: str = typer.Option(..., help="Name of label column"),
    schema_name: str = typer.Option(..., help="Feature schema name"),
    model_name: str = typer.Option("logreg", help="Model name"),
    model_version: str = typer.Option("0.1.0", help="Model version"),
    test_frac: float = typer.Option(0.3, help="Test split fraction"),
):
    """
    Train a model for a given feature schema and persist it.
    """

    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset not found: {dataset_path}")

    typer.echo(f"Loading dataset from {dataset_path}...")
    # load dataset
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix == ".txt":
        df = pd.read_csv(dataset_path)
    else:
        raise typer.BadParameter("Only .csv, .txt, .parquet are supported")

    if label_col not in df.columns:
        raise typer.BadParameter(f"Label column '{label_col}' not in dataset")

    # map labels to 0 and 1
    y = df[label_col].map({"benign": 0, "attack": 1})
    X = df.drop(columns=[label_col])

    # resolve output dir for model
    output_dir = get_model_path(model_name, model_version)

    if output_dir.exists():
        raise typer.BadParameter(f"Model version already exists: {output_dir}")

    typer.echo("Starting training process...")
    # train
    summary = train_model_for_schema(
        X=X,
        y=y,
        schema_name=schema_name,
        model_name=model_name,
        model_version=model_version,
        output_dir=output_dir,
        test_frac=test_frac,
    )

    typer.echo("\nTraining completed:")
    typer.echo(f"  Model: {summary.model_name}")
    typer.echo(f"  Version: {summary.model_version}")
    typer.echo(f"  Schema: {summary.schema_name}")
    typer.echo(f"  Feature names: {summary.feature_names}")
    typer.echo(f"  AUC: {summary.auc:.4f}")
    typer.echo(f"  Features: {summary.n_features}")
    typer.echo(f"  Output dir: {summary.output_dir}")
