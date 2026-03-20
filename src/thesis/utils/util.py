import pandas as pd
from typing import List, Tuple, Optional

def attacks_per_period_report(
    df,
    period="week",                 # "hour" | "day" | "week"
    ts_col="timestamp",
    y_col="y",
    scenario_col=None,
    tz="UTC",
    week_start="MON",              # "MON" (ISO) or "SUN"
):
    """
    Returns a report of attacks/benign counts bucketed by period.
    - period: "hour", "day", or "week"
    - scenario_col: if provided, includes per-scenario breakdown
    - tz: timezone for bucketing (e.g., "UTC", "Europe/Amsterdam")
    - week_start: controls week buckets ("MON" or "SUN")
    """
    period = period.lower().strip()
    if period not in {"hour", "day", "week"}:
        raise ValueError("period must be one of: 'hour', 'day', 'week'")

    d = df.copy()

    # timestamps -> tz-aware
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col])
    if tz:
        d[ts_col] = d[ts_col].dt.tz_convert(tz)

    # labels
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[y_col])
    d[y_col] = d[y_col].astype(int)

    # bucket column
    if period == "hour":
        d["bucket"] = d[ts_col].dt.floor("h")
    elif period == "day":
        d["bucket"] = d[ts_col].dt.floor("D")
    else:  # week
        # Pandas uses W-SUN / W-MON etc. Week label is end-of-week by default, so use start_time
        anchor = "SUN" if week_start.upper() == "SUN" else "MON"
        d["bucket"] = d[ts_col].dt.to_period(f"W-{anchor}").dt.start_time
        # start_time loses tz; re-localize to same tz for consistency
        if tz:
            d["bucket"] = d["bucket"].dt.tz_localize(tz)

    # overall counts
    overall = (
        d.groupby("bucket")[y_col]
        .agg(n_total="count", n_attacks="sum")
        .reset_index()
        .sort_values("bucket")
    )
    overall["n_benign"] = overall["n_total"] - overall["n_attacks"]
    overall["attack_rate"] = overall["n_attacks"] / overall["n_total"]
    overall["has_both_classes"] = (overall["n_attacks"] > 0) & (overall["n_benign"] > 0)

    # per-scenario breakdown
    by_scenario = None
    if scenario_col is not None and scenario_col in d.columns:
        by_scenario = (
            d.groupby([scenario_col, "bucket"])[y_col]
            .agg(n_total="count", n_attacks="sum")
            .reset_index()
            .sort_values([scenario_col, "bucket"])
        )
        by_scenario["n_benign"] = by_scenario["n_total"] - by_scenario["n_attacks"]
        by_scenario["attack_rate"] = by_scenario["n_attacks"] / by_scenario["n_total"]
        by_scenario["has_both_classes"] = (by_scenario["n_attacks"] > 0) & (by_scenario["n_benign"] > 0)

    summary = {
        f"n_{period}s": int(len(overall)),
        f"{period}s_with_attacks": int((overall["n_attacks"] > 0).sum()),
        f"{period}s_with_both_classes": int(overall["has_both_classes"].sum()),
        f"{period}s_single_class": int((~overall["has_both_classes"]).sum()),
        f"min_attacks_per_{period}": int(overall["n_attacks"].min()) if len(overall) else 0,
        f"median_attacks_per_{period}": float(overall["n_attacks"].median()) if len(overall) else 0.0,
        f"max_attacks_per_{period}": int(overall["n_attacks"].max()) if len(overall) else 0,
    }

    return overall, by_scenario, summary


def make_time_windows(
    t: pd.Series,
    window_size: str = "3D",  # "3D", "12H", "6H"
    step_size: str = "3D",  # "1D", "6H"
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    align_to: Optional[
        str
    ] = "D",  # "D" for day, "h" for hour, or None. Rounds start and end times to clean boundaries before generating windows
    tz: str = "UTC",
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of (start_k, end_k) timestamps for half-open windows [start, end).
    """

    if start is None:
        start = t.min()
    if end is None:
        end = t.max()

    start = pd.Timestamp(start).tz_convert(tz)
    end = pd.Timestamp(end).tz_convert(tz)

    # Optional alignment
    if align_to is not None:
        start = start.floor(align_to)
        end = end.ceil(align_to)

    win = pd.Timedelta(window_size)
    step = pd.Timedelta(step_size)

    windows = []
    cur = start

    while cur < end:
        windows.append((cur, cur + win))
        cur += step

    return windows