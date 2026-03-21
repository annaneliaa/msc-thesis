import typer
import pandas as pd

from thesis.schemas.dataframe_schemas import SCHEMAS
from thesis.config import load_settings
from thesis.mining.dummy_job import run_dummy_mining_job
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.validation import validate_dataframe
from thesis.registry.models import list_all_models
from thesis.registry.encoders import list_all_encoders

# from thesis.experiments.runner import run_experiment

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
def mine(run_name: str = "debug") -> None:
    path = run_dummy_mining_job(run_name=run_name)
    typer.echo(f"Dummy mining output written to: {path}")


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
