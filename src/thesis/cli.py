import typer
import pandas as pd
from pathlib import Path
import json
import csv
import os

from thesis.schemas.dataframe_schemas import SCHEMAS
from thesis.features.schema_registry import FEATURE_SCHEMAS
from thesis.config import load_settings
from thesis.mining.itemset_mining_job import run_transaction_eclat_job
from thesis.mining.sequence_mining_job import (
    run_transaction_prefixspan_job,
)
from thesis.utils.runs import create_run_dir
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.validation import validate_dataframe
from thesis.registry.models import list_all_models, get_model_path
from thesis.registry.encoders import list_all_encoders
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.preprocessing.service import process_alert_batch, select_groups_from_cache
from thesis.preprocessing.mining_prep import build_transactions
from thesis.training.service import train_model_for_schema
from thesis.encoders.service import encode_transactions_for_schema
from thesis.features.service import build_persist_and_register_symbolic_schema

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
                "sorted_items": [sorted(itemset) for itemset in s.sorted_items],
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
def load_transactions(
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
        out_path = out_dir / "transactions_raw.json"

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
                "sorted_items": [sorted(itemset) for itemset in t.sorted_items],
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
def encode_transactions(
    scenario_name: str = typer.Option(..., "--scenario", "-s"),
    schema_name: str = typer.Option("base", "--schema-name"),
    schema_version: str | None = typer.Option(None, "--schema-version"),
    cache_dir: str = typer.Option("artifacts/cache"),
    top_k: int | None = typer.Option(None, "--top-k"),
) -> None:
    """
    Load group snapshots from cache, convert to transactions,
    load a FeatureSchema, encode transactions, and save row-based features.
    """
    try:
        cache = TokenCache(cache_dir=Path(cache_dir), scenario=scenario_name)

        snapshots = select_groups_from_cache(
            cache=cache,
            allowed_methods=None,
            limit=None,
            min_start_ts=None,
            max_end_ts=None,
            require_closed=True,
        )

        transactions = list(build_transactions(snapshots))

        schema = FEATURE_SCHEMAS.load(
            scenario_name=scenario_name,
            schema_name=schema_name,
            schema_version=schema_version,
        )

        print(f"Loaded schema {schema_name} for scenario {scenario_name}.")

        feature_df = encode_transactions_for_schema(
            transactions=transactions,
            schema=schema,
            top_k=top_k,
        )

        meta_df = pd.DataFrame(
            [
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
                for t in transactions
            ]
        )

        df = pd.concat(
            [
                meta_df.reset_index(drop=True),
                feature_df.reset_index(drop=True),
            ],
            axis=1,
        )

        safe_schema_name = schema_name.replace("+", "_").replace("/", "_")

        out_dir = Path(f"artifacts/cache/{scenario_name}/transactions")
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / f"transactions_{safe_schema_name}.parquet"
        csv_path = out_dir / f"transactions_{safe_schema_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        typer.echo(
            f"Saved {len(transactions)} encoded transactions under schema "
            f"'{schema_name}' version '{schema.schema_version}' to "
            f"{parquet_path} and {csv_path}"
        )

    except Exception as e:
        typer.echo(f"Transaction encoding failed: {e}")
        raise typer.Exit(code=1)


@app.command("mine")
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
    Load cached Transactions and run transaction-level Eclat and PrefixSpan mining.
    """
    transactions_path = Path(
        f"artifacts/cache/{scenario}/transactions/transactions_raw.json"
    )
    try:
        typer.echo(f"Loading transactions from {transactions_path}...")
        tx_path = Path(transactions_path)
        run_dir = create_run_dir(run_name)

        eclat_result = run_transaction_eclat_job(
            transactions_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_len,
            target_label=target_label,
            run_dir=run_dir,
        )
        typer.echo(f"Eclat mining complete. Artifacts saved to: {eclat_result.run_dir}")

    except Exception as e:
        typer.echo(f"Transaction mining failed: {e}")
        raise typer.Exit(code=1)

    try:
        typer.echo("Running PrefixSpan item sequence mining...")
        item_seq_result = run_transaction_prefixspan_job(
            transactions_path=tx_path,
            scenario_name=scenario,
            run_name=run_name,
            min_support=min_support,
            max_len=max_len,
            target_label=target_label,
            run_dir=run_dir,
        )
        typer.echo(
            f"Item sequence mining complete. Artifacts saved to: {item_seq_result.run_dir}"
        )

        # typer.echo("Running PrefixSpan itemset sequence mining...")
        # itemset_seq_result = run_transaction_itemset_prefixspan_job(
        #     transactions_path=tx_path,
        #     scenario_name=scenario,
        #     run_name=run_name,
        #     min_support=min_support,
        #     max_len=max_len,
        #     target_label=target_label,
        #     run_dir=run_dir,
        # )
        # typer.echo(
        #     f"Itemset sequence mining complete. Artifacts saved to: {itemset_seq_result.run_dir}"
        # )

    except Exception as e:
        typer.echo(f"Sequence mining failed: {e}")
        raise typer.Exit(code=1)

    try:
        # Combine features from all three mining approaches
        typer.echo("Building combined feature schema from all mining results...")

        # Prepare ECLAT itemsets
        eclat_df = eclat_result.mined_df.copy()
        eclat_df["mining_type"] = "itemset"

        # Prepare item sequences (rename sequence → itemset)
        item_seq_df = item_seq_result.mined_df.copy()
        item_seq_df = item_seq_df.rename(columns={"sequence": "itemset"})
        item_seq_df["mining_type"] = "item_sequence"

        # Prepare itemset sequences (rename sequence → itemset)
        # itemset_seq_df = itemset_seq_result.mined_df.copy()
        # itemset_seq_df = itemset_seq_df.rename(columns={"sequence": "itemset"})
        # itemset_seq_df["mining_type"] = "itemset_sequence"

        # Select relevant columns for feature schema
        cols_to_keep = [
            "itemset",
            "mining_type",
            "support",
            "confidence_attack",
            "confidence_benign",
        ]
        eclat_df = eclat_df[[c for c in cols_to_keep if c in eclat_df.columns]]
        item_seq_df = item_seq_df[[c for c in cols_to_keep if c in item_seq_df.columns]]
        # itemset_seq_df = itemset_seq_df[
        #     [c for c in cols_to_keep if c in itemset_seq_df.columns]
        # ]

        # Combine all features
        combined_df = pd.concat(
            [eclat_df, item_seq_df],
            axis=0,
            ignore_index=True,
        )
        combined_df.to_csv(os.path.join(run_dir, "combined_mining_df.csv"), index=False)

        # Sort by confidence in target label (descending), then by support
        sort_cols = []
        if "confidence_attack" in combined_df.columns:
            sort_cols.append("confidence_attack")
        elif "confidence_benign" in combined_df.columns:
            sort_cols.append("confidence_benign")
        if "support" in combined_df.columns:
            sort_cols.append("support")

        if sort_cols:
            combined_df = combined_df.sort_values(
                by=sort_cols,
                ascending=False,
                na_position="last",
            ).reset_index(drop=True)

        typer.echo(
            f"Combined {len(eclat_df)} itemsets, {len(item_seq_df)} item sequences, "
            f"into {len(combined_df)} features."
        )

        schema_path = build_persist_and_register_symbolic_schema(
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

    output_dir = get_model_path(model_name, model_version)

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
