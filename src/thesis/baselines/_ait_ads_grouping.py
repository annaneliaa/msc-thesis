"""
Materializes alertbert/deepcase groupings into the *same* on-disk cache
format/location `thesis.pipeline.pipeline.ingest_ait_alert_batch` already
produces for fixed_window/time_delta/cscas_grouping -- so nothing
downstream (`load_or_build_alert_groups`, `encode_and_cache_alert_groups`)
needs to know or care which of the 5 grouping methods produced the cache.

Why this exists rather than just passing grouping_mode="alertbert"/"deepcase"
through the standard pipeline: it can't. `process_alert_batch` only ever
forwards `window_size` as an extra kwarg (for fixed_window); calling
`group_alerts(alerts, method="alertbert")` without `delta`/`theta` (or
"deepcase" without `train_alerts`/`train_id`) raises immediately --
`GroupingConfig` has no fields for any of that. `baselines/grouping/
alertbert_sweep.py`/`deepcase_sweep.py` call `group_alerts_alertbert`/
`group_alerts_deepcase` directly for exactly this reason, but only ever
persist grouping-*quality* metrics (purity, size stats) -- never a labeled
AlertGroup cache. This module is the missing bridge: run the same grouping
calls, then feed the result through the same `CacheIngestor`/`TokenCache`
machinery `ingest_ait_alert_batch` uses internally.

Operating points below are the lowest-mean-`mixed_frac` (best purity,
averaged over the existing sweep's per-scenario rows) row in
`baselines/grouping/results/{alertbert,deepcase}.json` -- computed once and
hardcoded here, not re-swept. This deliberately diverges from
`grouping_comparison.ipynb`'s own "best" convention (highest `reduction`,
picked per scenario) -- that notebook is asking "which setting shrinks the
alert volume most", this module is asking "which setting produces the
purest groups to classify", a different question with a different answer.

Leakage guard: the pretrained AlertBERT checkpoint (mlm_1l_4h_16d_
original_1_60k) and the DeepCASE ContextBuilder were both trained on
shaw/wardbeck/wheeler/wilson (see baselines/grouping/_setup.py's
DEEPCASE_TRAIN_SCENARIOS and alertbert_sweep.py's module docstring) --
grouping those same 4 scenarios with either method for a *baseline result*
would be self-training leakage, which is exactly why the grouping-quality
sweep only ever evaluated on fox/russellmitchell. Baseline results for
alertbert/deepcase are therefore only valid for fox/harrison/
russellmitchell/santos.

Known, non-retriable DeepCASE failure mode: its ContextBuilder is trained
once on a closed vocabulary of event types seen in shaw/wardbeck/wheeler/
wilson (input_size/output_size fixed at training time). A held-out
scenario whose alerts contain an event type that never appeared in that
training corpus makes DeepCASE's own `context_builder.forward()` raise
ValueError("Expected N different input events, but received input event
... not in expected range") -- seen for real on 'santos'. This is a
structural vocabulary-coverage limitation, not a transient bug: it fails
identically on every retry, since it depends on santos's own alert
content, not randomness. Retraining the ContextBuilder on a broader/
different corpus would be a separate, deliberate decision, not something
this module should paper over silently.
"""

from __future__ import annotations

from pathlib import Path

from thesis.caching.cache import TokenCache
from thesis.caching.ingestor import CacheIngestor
from thesis.grouping.window_sweep import load_and_tokenize

ALERTBERT_BEST = {"delta": 1.5, "theta": 1024.0}  # mean mixed_frac ~= 0.000039
DEEPCASE_BEST = {"context_length": 2, "eps": 1.0}  # mean mixed_frac ~= 0.010367

# Both the AlertBERT checkpoint and the DeepCASE ContextBuilder were trained
# on these 4 -- grouping them with either method here would be leakage.
LEAKAGE_SCENARIOS = {"shaw", "wardbeck", "wheeler", "wilson"}

LEARNED_METHODS = {"alertbert", "deepcase"}


def materialize_learned_grouping(scenario: str, method: str, cache_dir: Path) -> None:
    """Populate cache_dir with an alertbert/deepcase grouping for `scenario`,
    in the same TokenCache format the standard ingestion pipeline uses.
    Raises on a leakage scenario rather than silently producing a
    misleading result. Skips (like ingest_ait_alert_batch does) if the
    cache is already populated."""
    if method not in LEARNED_METHODS:
        raise ValueError(
            f"materialize_learned_grouping only handles {LEARNED_METHODS}, got {method!r}"
        )
    if scenario in LEAKAGE_SCENARIOS:
        raise ValueError(
            f"{method} grouping on '{scenario}' would be self-training leakage -- "
            f"both the AlertBERT checkpoint and the DeepCASE ContextBuilder were "
            f"trained on {sorted(LEAKAGE_SCENARIOS)}. Use fixed_window/time_delta/"
            f"cscas_grouping for this scenario, or one of "
            f"fox/harrison/russellmitchell/santos for {method}."
        )

    group_store_dir = cache_dir / "groups"
    if group_store_dir.exists() and any(group_store_dir.glob("*.json")):
        print(f"  [skip] {method} group cache already populated at {group_store_dir}")
        return

    print(f"  Grouping '{scenario}' with {method} ({_describe(method)})...")
    alerts = load_and_tokenize(scenario)

    if method == "alertbert":
        from thesis.grouping.group_alerts import group_alerts_alertbert

        records = group_alerts_alertbert(alerts, **ALERTBERT_BEST, device="auto")
    else:  # deepcase
        from thesis.baselines.grouping._setup import (
            DEEPCASE_TRAIN_ID,
            load_deepcase_train_alerts,
        )
        from thesis.grouping.group_alerts import group_alerts_deepcase

        records = group_alerts_deepcase(
            alerts,
            load_deepcase_train_alerts(),
            DEEPCASE_TRAIN_ID,
            min_samples=5,
            threshold=0.2,
            seed=0,
            device="auto",
            **DEEPCASE_BEST,
        )

    cache = TokenCache(cache_dir=cache_dir)
    CacheIngestor(cache=cache).ingest_groups(alerts, records)
    print(f"  Cached {method} grouping for '{scenario}' -> {cache_dir}")


def _describe(method: str) -> str:
    if method == "alertbert":
        return f"delta={ALERTBERT_BEST['delta']}, theta={ALERTBERT_BEST['theta']}"
    return (
        f"context_length={DEEPCASE_BEST['context_length']}, eps={DEEPCASE_BEST['eps']}"
    )
