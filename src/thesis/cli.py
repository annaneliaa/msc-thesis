import typer

from thesis.config import load_settings
from thesis.mining.dummy_job import run_dummy_mining_job
from thesis.paths import ensure_artifact_dirs

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