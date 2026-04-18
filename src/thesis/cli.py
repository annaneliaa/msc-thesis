import typer
import pandas as pd
from pathlib import Path
import json

from thesis.schemas.dataframe_schemas import SCHEMAS
from thesis.schemas.cache import CacheQuery
from thesis.config import load_settings
from thesis.mining.mining_dummy_job import run_dummy_mining_job
from thesis.mining.mining_transaction_csv_job import run_transaction_eclat_job
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.validation import validate_dataframe
from thesis.registry.models import list_all_models
from thesis.registry.encoders import list_all_encoders
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.service import process_one_alert
from thesis.preprocessing.transaction_selector import (
    select_transactions as select_transactions_from_cache,
)

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
def mine_dummy(run_name: str = "debug") -> None:
    path = run_dummy_mining_job(run_name=run_name)
    typer.echo(f"Dummy mining output written to: {path}")


@app.command()
def mine_transactions_csv(
    scenario_csv: str = typer.Argument(
        ..., help="Path to one scenario transaction CSV."
    ),
    run_name: str = typer.Option("debug", help="MLflow run name."),
    min_support: float = typer.Option(0.05, help="Minimum support threshold."),
    max_len: int = typer.Option(3, help="Maximum itemset size."),
    target_label: str = typer.Option("benign", help="Label to mine from."),
    label_col: str = typer.Option("tx_label", help="Transaction label column."),
    items_col: str = typer.Option("items", help="Items column."),
) -> None:
    path = run_transaction_eclat_job(
        scenario_csv=Path(scenario_csv),
        run_name=run_name,
        min_support=min_support,
        max_len=max_len,
        target_label=target_label,
        label_col=label_col,
        items_col=items_col,
    )
    typer.echo(f"Transaction mining output written to: {path}")


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
def preprocess_single_alert(
    alert_file: str = typer.Argument(..., help="Path to a single alert JSON file."),
    scenario: str = typer.Option(..., help="Scenario name."),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
) -> None:
    """
    Run parsing -> tokenization -> cache ingestion for one alert.
    """
    try:
        alert_path = Path(alert_file)
        cache_path = Path(cache_dir)

        with alert_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        cache = TokenCache(cache_dir=cache_path)
        tokenized = process_one_alert(row=payload, scenario=scenario, cache=cache)

        typer.echo(f"Processed alert_id={tokenized.alert_id}")
        typer.echo(f"Window ID={tokenized.window_id}")
        typer.echo(f"repr_tokens={sorted(tokenized.repr_tokens)}")
        typer.echo(f"mining_tokens={sorted(tokenized.mining_tokens)}")

    except Exception as e:
        typer.echo(f"Preprocessing failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def preprocess_alert_batch(
    alerts_file: str = typer.Argument(
        ..., help="Path to a JSON file with multiple alerts."
    ),
    scenario: str = typer.Option(..., help="Scenario name."),
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
) -> None:
    """
    Run parsing -> tokenization -> cache ingestion for multiple alerts
    from one input JSON file.
    """
    try:
        alerts_path = Path(alerts_file)
        cache_path = Path(cache_dir)

        with alerts_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError("Input file must contain a JSON list of alert objects.")

        cache = TokenCache(cache_dir=cache_path)

        processed_count = 0
        for row in payload:
            try:
                process_one_alert(row=row, scenario=scenario, cache=cache)
                processed_count += 1
            except Exception:
                continue  # skip bad alert

        typer.echo(f"Processed {processed_count} alerts.")
        typer.echo(f"Cache written to: {cache_path}")

    except Exception as e:
        typer.echo(f"Batch preprocessing failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def select_transactions(
    cache_dir: str = typer.Option(
        "artifacts/cache", help="Directory where cache files are stored."
    ),
    min_window_id: int | None = typer.Option(None, help="Minimum window id."),
    max_window_id: int | None = typer.Option(None, help="Maximum window id."),
    only_closed: bool = typer.Option(False, help="Select only closed windows."),
    retention_windows: int | None = typer.Option(
        None, help="How many most recent windows to keep."
    ),
    decay_factor: float = typer.Option(
        1.0, help="Decay factor for transaction weights."
    ),
) -> None:
    """
    Query cache and build window transactions ready to pass to miner.
    """
    try:
        cache = TokenCache(cache_dir=Path(cache_dir))

        query = CacheQuery(
            min_window_id=min_window_id,
            max_window_id=max_window_id,
            only_closed=only_closed,
        )

        transactions = select_transactions_from_cache(
            cache=cache,
            query=query,
            retention_windows=retention_windows,
            decay_factor=decay_factor,
        )

        typer.echo(f"Selected {len(transactions)} transactions.")
        for tx in transactions:
            typer.echo(
                f"window_id={tx.window_id} "
                f"n_alerts={tx.n_alerts} "
                f"weight={tx.weight} "
                f"items={sorted(tx.items)}"
            )

        out_dir = Path("artifacts/cache/transactions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir / "transactions.json"
        )  # TODO: check if this means there is always one transactions file in cache

        # convert to serializable format
        serialized = [
            {
                "transaction_id": tx.window_id,  # transaction_id is just window_id for now
                "window_start": tx.window_start,
                "window_end": tx.window_end,
                "n_alerts": tx.n_alerts,
                "items": sorted(list(tx.items)),
                "tx_label": tx.tx_label,
                "alert_labels": (
                    sorted(list(tx.alert_labels))
                    if tx.alert_labels is not None
                    else None
                ),
                "weight": tx.weight,
            }
            for tx in transactions
        ]

        with open(out_path, "w") as f:
            json.dump(serialized, f, indent=2)

        typer.echo(f"Saved transactions to {out_path}")

    except Exception as e:
        typer.echo(f"Transaction selection failed: {e}")
        raise typer.Exit(code=1)
