"""Train an AlertBERT MLM on thesis scenario alert data from a YAML config file.

Usage (from msc-thesis/):
    python src/thesis/scripts/train_alertbert.py
    python src/thesis/scripts/train_alertbert.py --config configs/alertbert_training.yaml

Alert data is read from artifacts/processed-data/{scenario}/alerts.json (sorted by
timestamp). The first (1 - test_frac - val_frac) fraction is used for training and
the next val_frac fraction for validation — consistent with the downstream classifier's
time-based test_frac holdout (make_holdout_split in training/util.py). The final
test_frac fraction is never touched.

The trained model is saved under artifacts/alertbert/ with the ID:
    mlm_{num_layers}l_{n_heads}h_{dim_per_head}d_{scenario}_{id_suffix}_{k}k

After training, set alertbert_grouping.yaml → alertbert.model_id to that ID.

Notes on compatibility
----------------------
tensordict: MaskedLangModelTrainWrapper / MaskedLangModelEvalWrapper use a dict
in_keys API that renames TensorDict keys (e.g. batch['short_mask'] → kwarg 'short').
tensordict 0.12.x passes the original key name instead, breaking the rename. We
call MaskedLanguageModel(**src) directly (plain nn.ModuleDict) to work around this.

MPS / float32: Unix timestamps are too large for float32 precision (~7 significant
digits vs. ~10 needed to resolve individual seconds in 1.6e9-range values). To enable
MPS (Apple Silicon GPU), timestamps are normalised by subtracting the training-set
minimum before encoding, so values start near 0 and float32 is sufficient. The offset
is saved in each checkpoint's report.json as "time_offset" and must be applied at
inference time (see alertbert_grouper.py). models.py has been patched to use float32
throughout the RotaryPositionalEncoding (freqs buffer + _get_rotation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime as dt
from math import ceil, floor
from pathlib import Path

import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[3]  # msc-thesis/
_ALERTBERT_ROOT = _REPO / "external" / "AlertBERT"

if str(_ALERTBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALERTBERT_ROOT))

from alertbert.aitads import AlertSequenceBatchSampler  # noqa: E402
from alertbert.eval_mlm import classification_report  # noqa: E402
from alertbert.models import (  # noqa: E402
    MaskedLangModelParams,
    MaskedLanguageModel,
    MultiTargetLoss,
)
from alertbert.preprocessing import (  # noqa: E402
    MaskedLangModelingSequenceCollate,
    build_feature_vocabs,
    default_collate_fn,
)
from alertbert.utils import OptimWrapper, get_device, log_to_stdout, set_up_log  # noqa: E402

from thesis.paths import ALERTBERT_MODELS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory AlertBERT-compatible dataset
# ---------------------------------------------------------------------------


class _ScenarioAlertDataset:
    """AlertBERT-compatible in-memory dataset backed by thesis scenario alert dicts.

    Mirrors the interface of alertbert.aitads.AlertDataset (data dict of numpy
    arrays, keys, __len__, __getitem__ with cyclic-index slice support) so it can
    be used directly with AlertSequenceBatchSampler and build_feature_vocabs.

    The thesis JSON uses "time" for the Unix timestamp; this class exposes it as
    "raw_time" to match the field name AlertBERT expects internally.
    """

    def __init__(self, alerts: list[dict], time_offset: float = 0.0) -> None:
        self.time_offset = time_offset
        self.data: dict[str, np.ndarray] = {
            "raw_time": np.array(
                [a["time"] - time_offset for a in alerts], dtype=np.float32
            ),
            "short": np.array([a.get("short") or "" for a in alerts]),
            "host": np.array([a.get("host") or "" for a in alerts]),
        }
        self.keys = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data["raw_time"])

    def __getitem__(self, idx: int | slice | np.ndarray) -> dict:
        if isinstance(idx, int):
            try:
                return {k: self.data[k][idx] for k in self.keys}
            except IndexError:
                return {k: self.data[k][idx % len(self)] for k in self.keys}
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            idx_array = np.arange(start=start, stop=stop, step=step)
        elif isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        if len(idx_array) > 0 and (idx_array[0] < 0 or idx_array[-1] >= len(self)):
            idx_array = idx_array % len(self)
        return {k: self.data[k][idx_array] for k in self.keys}

    def __iter__(self):
        for i in range(len(self)):
            yield {k: self.data[k][i] for k in self.keys}


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def _load_alerts(path: Path) -> list[dict]:
    with path.open() as f:
        alerts = json.load(f)
    alerts.sort(key=lambda a: a["time"])
    return alerts


def _split(alerts: list[dict], test_frac: float, val_frac: float):
    """Return (train_alerts, val_alerts); test portion is silently excluded."""
    n = len(alerts)
    test_start = int((1 - test_frac) * n)
    val_start = int((1 - val_frac) * test_start)
    return alerts[:val_start], alerts[val_start:test_start]


# ---------------------------------------------------------------------------
# Config → MaskedLangModelParams
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _build_params(cfg: dict) -> MaskedLangModelParams:
    scenario = cfg["scenario"]
    save_intervals = tuple(
        tuple(pair)
        for pair in cfg.get(
            "save_intervals", [(18000, 20000), (38000, 40000), (58000, 60000)]
        )
    )
    return MaskedLangModelParams(
        # Use scenario name as the "augment" slot so the model ID encodes the scenario
        # (e.g., 1l_4h_16d_fox_1). This is purely for identification — we never call
        # AITAlertDataset, so the augment value has no effect on data loading here.
        id_suffix=str(cfg.get("id_suffix", "1")),
        augment=scenario,
        context_size=cfg.get("context_size", 4096),
        batch_size=cfg.get("batch_size", 16),
        features=tuple(cfg.get("features", ["short", "host"])),
        targets=tuple(cfg.get("targets", ["short", "host"])),
        sampling=cfg.get("sampling", "index"),
        min_freq=cfg.get("min_freq", 10),
        n_heads=cfg.get("n_heads", 4),
        dim_per_head=cfg.get("dim_per_head", 16),
        num_layers=cfg.get("num_layers", 1),
        feedforward_factor=cfg.get("feedforward_factor", 4),
        activation=cfg.get("activation", "gelu"),
        gated_activation=cfg.get("gated_activation", True),
        encoding=cfg.get("encoding", "raw_time"),
        encoding_type=cfg.get("encoding_type", "rotary"),
        rotary_max_exp=cfg.get("rotary_max_exp", 14),
        rotary_cutoff=cfg.get("rotary_cutoff", 0.75),
        biases=cfg.get("biases", False),
        head_bias=cfg.get("head_bias", True),
        tie_weights=cfg.get("tie_weights", True),
        emb_init_std=cfg.get("emb_init_std", None),
        save_intervals=save_intervals,
        optimizer=cfg.get("optimizer", "adam"),
        scheduler=cfg.get("scheduler", "linear"),
        lr=cfg.get("lr", 5e-3),
        warm_up_steps=cfg.get("warm_up_steps", 200),
        decay=cfg.get("decay", 0.1),
        momentum=cfg.get("momentum", 0.9),
        gamma=cfg.get("gamma", None),
        class_balance=cfg.get("class_balance", 2.0),
        target_ratio=cfg.get("target_ratio", 0.2),
        mask_ratio=cfg.get("mask_ratio", 0.8),
        perturb_ratio=cfg.get("perturb_ratio", 0.1),
        path=str(ALERTBERT_MODELS_DIR),
        log=cfg.get("log", "train"),
    )


# ---------------------------------------------------------------------------
# Training helpers — bypass MaskedLangModelTrainWrapper / EvalWrapper because
# tensordict 0.12.x doesn't support the dict in_keys rename API they rely on.
# MaskedLanguageModel is a plain nn.ModuleDict and can be called directly.
# ---------------------------------------------------------------------------


def _model_src(batch, features, encoding):
    """Build the **src dict for a direct MaskedLanguageModel(**src) call."""
    src = {f: batch[f"{f}_mask"] for f in features}
    src[encoding] = batch[encoding]
    return src


def _train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    collate_fn_map,
    device,
    features,
    targets,
    encoding,
    epoch_num=None,
    total_epochs=None,
):
    model.train()
    n = len(loader)
    losses = torch.empty(n)
    log_every = (
        max(1, n // 4) if n >= 8 else None
    )  # log ~4 checkpoints per epoch if enough batches
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        mask_idx = torch.unbind(batch["mask_index"])
        preds = model(**_model_src(batch, features, encoding))
        true = tuple(
            collate_fn_map[t].compute_targets(batch[t][mask_idx]) for t in targets
        )
        masked_preds = tuple(pred[mask_idx] for pred in preds)
        loss = loss_fn(masked_preds, true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
        if log_every and (i + 1) % log_every == 0:
            epoch_str = f"epoch {epoch_num}/{total_epochs} " if epoch_num else ""
            logging.info(
                f"  {epoch_str}batch {i + 1}/{n}: loss = {losses[i].item():.5f}"
            )
    return losses.mean(), losses.std()


def _eval_model(
    model, loader, device, epochs, collate_fn_map, features, targets, encoding
):
    """Evaluate model; mirrors eval_masked_lang_model without the broken wrapper."""
    stats = {
        t: {"loss": [], "corr": [], "rank": [], "pred": [], "true": [], "size": []}
        for t in targets
    }
    xent = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with torch.no_grad():
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                mask_idx = torch.unbind(batch["mask_index"])
                preds = model(**_model_src(batch, features, encoding))
                for t, pred in zip(targets, preds):
                    true = collate_fn_map[t].compute_targets(batch[t][mask_idx])
                    out = pred[mask_idx]
                    stats[t]["true"].append(true.cpu())
                    stats[t]["size"].append(len(true))
                    stats[t]["loss"].append(xent(out, true).item())
                    pred_labels = out.argmax(dim=1)
                    stats[t]["pred"].append(pred_labels.cpu())
                    stats[t]["corr"].append((pred_labels == true).sum().item())
                    rank = torch.sum(
                        out.t() >= out[(torch.arange(len(true)), true)], dim=0
                    )
                    stats[t]["rank"].append(rank.cpu())
    for t in targets:
        stats[t]["size"] = sum(stats[t]["size"])
        stats[t]["loss"] = sum(stats[t]["loss"]) / stats[t]["size"]
        stats[t]["corr"] = sum(stats[t]["corr"]) / stats[t]["size"]
        stats[t]["rank"] = torch.cat(stats[t]["rank"])
        stats[t]["pred"] = torch.cat(stats[t]["pred"])
        stats[t]["true"] = torch.cat(stats[t]["true"])
    stats["total_loss"] = sum(stats[t]["loss"] for t in targets) / len(targets)
    return stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _run_training(
    params: MaskedLangModelParams,
    train_data: _ScenarioAlertDataset,
    val_data: _ScenarioAlertDataset,
) -> None:
    logging.info("Run id: " + params["id"])
    device = get_device()
    logging.info(f"Device: {device}")

    features = params["features"]
    targets = params["targets"]
    encoding = params["encoding"]

    logging.info(
        f"Training alerts: {len(train_data)}, validation alerts: {len(val_data)}"
    )

    collate_function_map = build_feature_vocabs(
        dataset=train_data,
        features=set(features) | set(targets),
        min_freq=params["min_freq"],
    )

    if encoding == "raw_time" and params["encoding_type"] == "learned":
        raise ValueError("Time encoding is not available for learned encoding.")
    collate_function_map[encoding] = default_collate_fn

    collate_function = MaskedLangModelingSequenceCollate(
        collate_function_map,
        params["target_ratio"],
        params["mask_ratio"],
        params["perturb_ratio"],
    )

    train_sampler = AlertSequenceBatchSampler(
        train_data,
        context_size=params["context_size"],
        batch_size=params["batch_size"],
        sampling_method=params["sampling"],
    )
    val_sampler = AlertSequenceBatchSampler(
        val_data,
        context_size=params["context_size"],
        batch_size=params["batch_size"],
        drop_last=False,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler, collate_fn=collate_function
    )
    val_loader = DataLoader(
        val_data, batch_sampler=val_sampler, collate_fn=collate_function
    )

    logging.info("Building model...")
    model = MaskedLanguageModel(params=params, vocabs=collate_function_map)
    model.to(device)

    save_intervals = params.save_intervals
    save_intervals_epochs = [
        (floor(s / len(train_loader)), ceil(e / len(train_loader)))
        for s, e in save_intervals
    ]
    epochs = save_intervals_epochs[-1][1]
    updates_per_epoch = len(train_loader)

    save_interval_index = 0
    save_interval_start, save_interval_end = save_intervals_epochs[save_interval_index]
    saving = False

    logging.info(
        f"{params['updates']} updates → {epochs} epochs "
        f"({updates_per_epoch} updates/epoch). Save intervals: {save_intervals_epochs}"
    )

    if params["scheduler"] == "schedulefree":
        if params["optimizer"] == "adam":
            from schedulefree import AdamWScheduleFree

            optimizer = AdamWScheduleFree(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                betas=(params["momentum"], 0.999),
                warmup_steps=params["warm_up_steps"],
            )
        elif params["optimizer"] == "sgd":
            from schedulefree import SGDScheduleFree

            optimizer = SGDScheduleFree(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                momentum=params["momentum"],
                warmup_steps=params["warm_up_steps"],
            )
    elif params["scheduler"] == "linear":
        if params["optimizer"] == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                betas=(params["momentum"], 0.999),
            )
        elif params["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                momentum=params["momentum"],
            )
        optimizer = OptimWrapper(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params["lr"],
                total_steps=epochs * updates_per_epoch,
                pct_start=float(params["warm_up_steps"]) / (epochs * updates_per_epoch),
                anneal_strategy="linear",
                cycle_momentum=False,
                div_factor=1e3,
                final_div_factor=1e4,
            )
        )

    class_weights = {
        t: torch.softmax(
            collate_function_map[t].get_frequencies().to(device)
            * params["class_balance"]
            * -1.0,
            dim=0,
        )
        for t in targets
    }
    loss_fn = MultiTargetLoss(
        [nn.CrossEntropyLoss(weight=class_weights[t]) for t in targets]
    )

    logging.info(
        f"Starting training: {epochs} epochs, {updates_per_epoch} batches/epoch"
    )
    model_name = None
    report = None
    best_train_stats = None
    best_val_stats = None
    t_start = time.monotonic()

    for e in range(epochs):
        model.train()
        optimizer.train()

        mean, std = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            collate_function_map,
            device,
            features,
            targets,
            encoding,
            epoch_num=e + 1,
            total_epochs=epochs,
        )

        elapsed = time.monotonic() - t_start
        frac = (e + 1) / epochs
        eta_s = elapsed / frac * (1 - frac) if frac > 0 else 0
        eta_str = f"{int(eta_s // 3600):02d}h{int(eta_s % 3600 // 60):02d}m"
        update = (e + 1) * updates_per_epoch
        logging.info(
            f"Epoch {e + 1:5d}/{epochs}  update {update:7d}  "
            f"train loss = {mean:.5f} ± {std:.5f}  ETA {eta_str}"
        )

        if e + 1 == save_interval_start:
            logging.info(
                f"--- entering save interval {save_interval_index + 1} "
                f"(updates {save_intervals[save_interval_index]}) ---"
            )
            saving = True
            best_val_loss = float("inf")
            best_train_stats = None
            best_val_stats = None

        if ((e % 10 == 9) and saving) or (e % 100 == 99):
            optimizer.eval()

            if params["scheduler"] == "schedulefree":
                # Flush optimizer momentum (schedulefree requirement).
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(device)
                        model(**_model_src(batch, features, encoding))

            logging.info(f"Evaluating at epoch {e + 1}...")
            model.eval()
            tr_stats = _eval_model(
                model,
                train_loader,
                device,
                3,
                collate_function_map,
                features,
                targets,
                encoding,
            )
            val_stats = _eval_model(
                model,
                val_loader,
                device,
                5,
                collate_function_map,
                features,
                targets,
                encoding,
            )
            logging.info(
                f"  total  train loss={tr_stats['total_loss']:.5f}  "
                f"val loss={val_stats['total_loss']:.5f}"
            )
            for t in targets:
                logging.info(
                    f"  {t:>8}  train loss={tr_stats[t]['loss']:.5f}  "
                    f"acc={tr_stats[t]['corr']:.5f}  |  "
                    f"val loss={val_stats[t]['loss']:.5f}  "
                    f"acc={val_stats[t]['corr']:.5f}"
                )

            if saving and val_stats["total_loss"] < best_val_loss:
                best_val_loss = val_stats["total_loss"]
                best_train_stats = tr_stats
                best_val_stats = val_stats
                params["updates"] = (e + 1) * updates_per_epoch
                model_name = f"mlm_{params['id']}_{save_intervals[save_interval_index][1] // 1000}k"
                save_location = f"{params['path']}/{model_name}"
                os.makedirs(save_location, exist_ok=True)
                torch.save(model.state_dict(), save_location + "/model.pt")
                for feat in set(features) | set(targets):
                    collate_function_map[feat].save(
                        save_location + f"/vocab_{feat}.json"
                    )
                report = {
                    "model": model_name,
                    "timestamp": str(dt.now()),
                    "epochs": e + 1,
                    "time_offset": train_data.time_offset,
                    "training": {
                        t: classification_report(tr_stats[t]) for t in targets
                    },
                    "validation": {
                        t: classification_report(val_stats[t]) for t in targets
                    },
                    "params": params.dict,
                }
                with open(save_location + "/report.json", "w") as fh:
                    json.dump(report, fh, indent=4)
                logging.info(
                    f"  checkpoint saved → {model_name}  (val loss {best_val_loss:.5f})"
                )

        if e + 1 == save_interval_end:
            logging.info(f"--- end of save interval {save_interval_index + 1} ---")
            saving = False
            if model_name and best_train_stats and best_val_stats:
                results = (
                    f"{str(dt.now())} Model: {model_name} "
                    f"train loss={best_train_stats['total_loss']:1.05f}, "
                    f"val loss={best_val_stats['total_loss']:1.05f}"
                )
                for t in targets:
                    results += (
                        f", {t} train acc={best_train_stats[t]['corr']:1.05f}"
                        f", {t} val acc={best_val_stats[t]['corr']:1.05f}"
                    )
                    if report:
                        results += (
                            f", {t} train f1={report['training'][t]['macro_f1']:1.05f}"
                            f", {t} val f1={report['validation'][t]['macro_f1']:1.05f}"
                        )
                results += f"; params={params}"
                with open(params["path"] + "/results.log", "a") as fh:
                    print(results, file=fh)

            save_interval_index += 1
            if save_interval_index < len(save_intervals_epochs):
                save_interval_start, save_interval_end = save_intervals_epochs[
                    save_interval_index
                ]

    total_time = time.monotonic() - t_start
    logging.info(
        f"Training complete in {int(total_time // 3600):02d}h{int(total_time % 3600 // 60):02d}m."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an AlertBERT MLM on thesis scenario alert data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "alertbert_training.yaml",
        help="Path to the training config YAML (default: configs/alertbert_training.yaml)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    scenario = cfg["scenario"]
    test_frac = cfg.get("test_frac", 0.3)
    val_frac = cfg.get("val_frac", 0.1)

    data_path = (
        Path(cfg["data_path"])
        if cfg.get("data_path")
        else (_REPO / "artifacts" / "processed-data" / scenario / "alerts.json")
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Alert data not found: {data_path}\n"
            "Run 'python -m thesis convert-alerts-to-json' first."
        )

    ALERTBERT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params = _build_params(cfg)
    model_id = params["id"]

    # Set up logging before any output so the full run is captured in the file.
    log_name = cfg.get("log", "train")
    if log_name:
        log_path = f"{ALERTBERT_MODELS_DIR}/{log_name}.log"
        set_up_log(str(ALERTBERT_MODELS_DIR / log_name))
        print(f"Logging to {log_path}")
    else:
        log_to_stdout()

    alerts = _load_alerts(data_path)
    train_alerts, val_alerts = _split(alerts, test_frac, val_frac)
    time_offset = float(min(a["time"] for a in train_alerts))
    train_data = _ScenarioAlertDataset(train_alerts, time_offset=time_offset)
    val_data = _ScenarioAlertDataset(val_alerts, time_offset=time_offset)

    logging.info("AlertBERT training")
    logging.info(f"  Scenario:   {scenario}  ({len(alerts)} alerts total)")
    logging.info(
        f"  Split:      {len(train_alerts)} train / {len(val_alerts)} val / "
        f"{len(alerts) - len(train_alerts) - len(val_alerts)} held-out test  "
        f"(test_frac={test_frac}, val_frac={val_frac})"
    )
    logging.info(f"  Model ID:   {model_id}")
    logging.info(f"  Time offset: {time_offset}")
    logging.info(f"  Output dir: {ALERTBERT_MODELS_DIR}")

    _run_training(params, train_data, val_data)

    model_prefix = f"mlm_{model_id}"
    logging.info(f"Done. Models saved under: {ALERTBERT_MODELS_DIR}")
    logging.info(f"Model directories:  {model_prefix}_<k>k")
    logging.info(
        f"To use this model for grouping, update alertbert_grouping.yaml:\n"
        f"  alertbert:\n"
        f"    model_id: {model_prefix}_<k>k\n"
        f"    models_path: artifacts/alertbert"
    )


if __name__ == "__main__":
    main()
