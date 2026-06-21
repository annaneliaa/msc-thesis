from __future__ import annotations

import pandas as pd

from thesis.preprocessing.tokenization import extract_signature_tokens


def extract_short_tokens(short_value: str) -> set[str]:
    if pd.isna(short_value):
        return set()
    parts = [part.strip() for part in str(short_value).split("-")]
    return {part for part in parts if part}


def time_of_day_bucket(ts) -> str:
    if pd.isna(ts):
        return "unknown"
    h = ts.hour
    if 5 <= h < 10:
        return "morning"
    elif 10 <= h < 14:
        return "midday"
    elif 14 <= h < 18:
        return "afternoon"
    elif 18 <= h < 22:
        return "evening"
    else:
        return "night"


def build_labeled_window_transactions(
    df: pd.DataFrame,
    time_col: str = "time",
    detector_col: str = "short",
    host_col: str = "host",
    label_col: str = "time_label",
    signature_col: str = "name",
    benign_label: str = "false_positive",
    window_size_s: int = 2,
) -> pd.DataFrame:
    out = df.copy()

    needed = [time_col, detector_col, host_col, label_col]
    out = out.dropna(subset=needed).copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out[time_col] = out[time_col].astype("int64")

    out["time_norm"] = pd.to_datetime(
        out[time_col], unit="s", utc=True, errors="coerce"
    )
    out["time_of_day"] = out["time_norm"].apply(time_of_day_bucket)
    out["time_epoch"] = (out["time_norm"].astype("int64") // 10**9).astype("int64")
    out["window_start"] = (out[time_col] // window_size_s) * window_size_s
    out["window_end"] = out["window_start"] + window_size_s

    out["detector_item"] = out[detector_col].astype(str)
    out["host_item"] = out[host_col].astype(str)
    out["detector_subtokens"] = out[detector_col].apply(extract_short_tokens)

    if signature_col in out.columns:
        out["signature_tokens"] = out[signature_col].apply(extract_signature_tokens)
    else:
        out["signature_tokens"] = [set() for _ in range(len(out))]

    def _label_window(labels: pd.Series) -> str:
        labels = set(labels.astype(str))
        has_benign = benign_label in labels
        has_attack = any(lbl != benign_label for lbl in labels)
        if has_attack and has_benign:
            return "mixed"
        elif has_attack:
            return "attack"
        else:
            return "benign"

    def _build_items(g: pd.DataFrame) -> set[str]:
        items = set(g["detector_item"]).union(set(g["host_item"]))
        for toks in g["detector_subtokens"]:
            items.update(toks)
        for toks in g["signature_tokens"]:
            items.update(toks)
        return items

    tx = (
        out.groupby(["window_start", "window_end"], sort=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n_alerts": len(g),
                    "items": _build_items(g),
                    "alert_labels": set(g[label_col].astype(str)),
                    "tx_label": _label_window(g[label_col]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return tx
