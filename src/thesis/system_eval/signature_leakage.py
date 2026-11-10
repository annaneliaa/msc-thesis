"""
Paper Check 3: do signature-derived items leak the label?

Every CSCAS signature is class-pure, which is why Base(5) drops SignatureID.
But category / ruleset / qualifier / CVE come from the signature text too: if
some of their values are near-pure, they're a SignatureID stand-in that every
miner can exploit. This module holds the pieces of that check:

  Step 1 (this file, read-only): a purity table over the signature-derived
  candidate fields on the training windows -- W0 (= W_src under
  "baseline_split") and pooled over every walk-forward training window --
  with the number of distinct SignatureIDs behind each value, plus a
  coverage summary of how many groups / attack groups fall under a flagged
  near-pure value.

  Steps 2-3 (seen-vs-unseen recall, the (a)/(b)/(c) ablation) build on the
  same SignatureID lookup and on rolling_walk_forward.fit_window.

SignatureID isn't carried on AlertGroup (ingest hashes it into group_id), so
`load_cscas_signature_lookup` rebuilds group_id from the raw CSV with the same
key (preprocessing.parsing.cscas_group_id) and maps it back.

Window scheme: "baseline_split" at the paper's granularity, identical to
rolling_walk_forward.py's -- step 0 trains on the CSCAS baseline train period,
step i>=1 on the i'th post-split window; the last window is test-only.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from thesis.config import GroupingConfig
from thesis.experiments._shared import (
    decide_threshold,
    fit_scored_model,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
)
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.mining.attribute_features import (
    CSCAS_SIGNATURE_DERIVED_FIELDS,
    _proto_mismatch,
    cve_age_years,
)
from thesis.paths import RESULTS_DIR, ROOT
from thesis.preprocessing.parsing import cscas_group_id
from thesis.preprocessing.suricata_tokenization import QUALIFIER_WORDS
from thesis.system_eval.rolling_walk_forward import fit_window
from thesis.system_eval.rule_precision_gate import (
    GATE_GS,
    gated_feature_names,
    rule_table,
)
from thesis.system_eval.temporal_decay import (
    WindowScheme,
    _resolve_baseline_split_idx,
    encode_target_window,
)

CSCAS_CSV_PATH = ROOT / "data" / "cscas" / "dataset-labeled-anon-ip.csv"
CSCAS_ALERT_GROUPS_CACHE_DIR = (
    ROOT / "artifacts" / "cache" / "cscas" / "groups" / "cscas_pregrouped"
)
CSCAS_BASELINE_SPLIT_TIME = "2022-01-26 06:23:21+02:00"
PAPER_GRANULARITY = 0.05

# Thresholds from the Check 3 spec.
MIN_GROUPS = 50
MIN_PURITY = 0.99

# CVE age buckets (years between the CVE's year and the alert). -1 = no CVE.
_CVE_AGE_EDGES = [0, 2, 5, 10, np.inf]
_CVE_AGE_LABELS = ["0-2y", "2-5y", "5-10y", "10y+"]

# Fields of the purity frame, i.e. the signature-derived candidate items as
# the miner sees them (cve_age_years bucketed -- the miner thresholds it).
PURITY_FIELDS: tuple[str, ...] = (
    "category",
    "ruleset",
    "cve_present",
    "cve_age_bucket",
    "proto_mismatch",
    *(f"qualifier_{w}" for w in sorted(QUALIFIER_WORDS)),
)
# Base(5) field reported alongside, but not a signature-text item: a
# per-signature rate, so a reviewer may read it as a SignatureID fingerprint
# too. Same in every ablation arm; diagnostic only.
DIAGNOSTIC_FIELDS: tuple[str, ...] = ("smpd_bin",)
_SMPD_N_BINS = 20


# --- SignatureID lookup ---------------------------------------------------


def load_cscas_signature_lookup(csv_path: Path = CSCAS_CSV_PATH) -> pd.Series:
    """group_id -> SignatureID, rebuilt from the raw CSCAS CSV with the same
    key ingest uses (cscas_group_id; IntIP "-1"/"" -> None, exactly as
    IncomingSuricataGroup.from_row does). Duplicate keys can only ever map to
    the same SignatureID (it's part of the key), so they're dropped."""
    # Default read_csv dtype/NA handling + str(), exactly as ingest's
    # pd.read_csv(...).to_dict("records") -> IncomingSuricataGroup.from_row.
    df = pd.read_csv(csv_path, usecols=["Timestamp", "SignatureID", "ExtIP", "IntIP"])
    int_ip = df["IntIP"].map(lambda v: str(v).strip())
    int_ip = int_ip.where(~int_ip.isin(["-1", ""]), None)
    ids = [
        cscas_group_id(int(s), str(e), i, str(t))
        for s, e, i, t in zip(df["SignatureID"], df["ExtIP"], int_ip, df["Timestamp"])
    ]
    lookup = pd.Series(df["SignatureID"].astype(int).to_numpy(), index=ids)
    return lookup[~lookup.index.duplicated()]


def signature_ids_for(groups: list, lookup: pd.Series) -> np.ndarray:
    """SignatureID per group (in order). Raises if any group is missing from
    the lookup -- a miss means the key reconstruction drifted from ingest."""
    sig = lookup.reindex([g.group_id for g in groups])
    n_miss = int(sig.isna().sum())
    if n_miss:
        raise ValueError(
            f"{n_miss}/{len(groups)} groups have no SignatureID in the CSV lookup "
            "-- cscas_group_id no longer matches how ingest built group_id."
        )
    return sig.astype(int).to_numpy()


# --- Windows --------------------------------------------------------------


def build_baseline_split_scheme(
    alert_groups: list, split_time: str = CSCAS_BASELINE_SPLIT_TIME
) -> WindowScheme:
    """The same "baseline_split" WindowScheme rolling_walk_forward.py uses."""
    split_idx = _resolve_baseline_split_idx(alert_groups, split_time)
    return WindowScheme("baseline_split", len(alert_groups), split_idx)


def training_window_bounds(
    scheme: WindowScheme, gran: float = PAPER_GRANULARITY
) -> list[tuple[int, int]]:
    """[start, end) of every walk-forward *training* window Wi, i = 0..n-2
    (the last window is only ever a test window). Mirrors
    rolling_walk_forward._step_bounds."""
    n = scheme.n_windows(gran)
    return [
        scheme.source_bounds(gran) if i == 0 else scheme.target_bounds(gran, i)
        for i in range(n - 1)
    ]


# --- Purity ---------------------------------------------------------------


def signature_item_frame(groups: list, signature_ids: np.ndarray) -> pd.DataFrame:
    """One row per labeled group: label, signature_id and the signature-derived
    candidate items. Values computed with the same helpers
    compute_candidate_attribute_features uses (cve_age_years, _proto_mismatch,
    QUALIFIER_WORDS), so they match what the miner sees; computed directly
    rather than through it to skip its ~40 similarity columns per group."""
    labels, mask = labels_and_mask(groups)
    rows = []
    for g in groups:
        qualifiers = g.qualifiers or set()
        age = cve_age_years(g.cve_refs or set(), g.start_ts)
        row = {
            "category": g.category or "",
            "ruleset": g.ruleset or "",
            "cve_present": bool(g.cve_refs),
            "cve_age": -1.0 if age is None else float(age),
            "proto_mismatch": _proto_mismatch(g.category, g.proto),
            "signature_matches_per_day": float(g.signature_matches_per_day or 0.0),
        }
        for w in QUALIFIER_WORDS:
            row[f"qualifier_{w}"] = w in qualifiers
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["cve_age_bucket"] = pd.cut(
        frame["cve_age"], bins=_CVE_AGE_EDGES, labels=_CVE_AGE_LABELS, right=False
    ).astype(object)
    frame.loc[frame["cve_age"] < 0, "cve_age_bucket"] = "no_cve"
    frame["label"] = labels
    frame["signature_id"] = signature_ids
    return frame.loc[mask].reset_index(drop=True)


def _with_smpd_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantile bins of signature_matches_per_day, fitted on `frame` itself."""
    out = frame.copy()
    out["smpd_bin"] = pd.qcut(
        out["signature_matches_per_day"], q=_SMPD_N_BINS, duplicates="drop"
    ).astype(str)
    return out


def purity_table(
    frame: pd.DataFrame,
    fields: tuple[str, ...] = PURITY_FIELDS + DIAGNOSTIC_FIELDS,
    min_groups: int = MIN_GROUPS,
    min_purity: float = MIN_PURITY,
) -> pd.DataFrame:
    """One row per (field, value): n_groups, n_attack, attack_rate,
    purity = max(rate, 1 - rate), the pure direction, lift over the window's
    base attack rate, and n_signatures behind the value. flagged = at least
    `min_groups` groups and purity >= `min_purity`.

    Note on direction: at a ~1.5% base rate, any value with attack rate <= 1%
    reaches purity 0.99 on the benign side, so benign-pure flags are cheap.
    The attack-pure ones are what a miner can turn into a SignatureID proxy."""
    if "smpd_bin" in fields and "smpd_bin" not in frame:
        frame = _with_smpd_bins(frame)
    base_rate = float(frame["label"].mean())
    parts = []
    for field in fields:
        agg = (
            frame.groupby(field, dropna=False, observed=True)
            .agg(
                n_groups=("label", "size"),
                n_attack=("label", "sum"),
                n_signatures=("signature_id", "nunique"),
            )
            .reset_index()
            .rename(columns={field: "value"})
        )
        agg.insert(0, "field", field)
        parts.append(agg)
    table = pd.concat(parts, ignore_index=True)
    table["value"] = table["value"].astype(str)
    table["n_attack"] = table["n_attack"].astype(int)
    table["attack_rate"] = table["n_attack"] / table["n_groups"]
    table["purity"] = np.maximum(table["attack_rate"], 1 - table["attack_rate"])
    table["direction"] = np.where(table["attack_rate"] >= 0.5, "attack", "benign")
    table["lift"] = table["attack_rate"] / base_rate if base_rate else np.nan
    table["flagged"] = (table["n_groups"] >= min_groups) & (
        table["purity"] >= min_purity
    )
    table["diagnostic"] = table["field"].isin(DIAGNOSTIC_FIELDS)
    return table


def coverage_summary(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Share of all groups and of attack groups under at least one flagged
    value, per direction (attack-pure / benign-pure / either) -- union over
    the signature-derived fields only (diagnostic fields excluded).
    Also reports how many of those groups' signatures are single-signature
    values (n_signatures == 1: effectively SignatureID itself)."""
    flagged = table[table["flagged"] & ~table["diagnostic"]]
    is_atk = frame["label"].to_numpy() == 1
    n, n_atk = len(frame), int(is_atk.sum())

    def _mask(rows: pd.DataFrame) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        for field, value in zip(rows["field"], rows["value"]):
            m |= frame[field].astype(str).to_numpy() == value
        return m

    out = []
    for direction in ("attack", "benign", "either"):
        rows = (
            flagged
            if direction == "either"
            else flagged[flagged["direction"] == direction]
        )
        m = _mask(rows)
        m1 = _mask(rows[rows["n_signatures"] == 1])
        out.append(
            {
                "direction": direction,
                "n_flagged_values": len(rows),
                "n_groups_covered": int(m.sum()),
                "share_groups": m.sum() / n if n else np.nan,
                "n_attack_covered": int((m & is_atk).sum()),
                "share_attack_groups": (m & is_atk).sum() / n_atk if n_atk else np.nan,
                "share_attack_groups_single_sig": (
                    (m1 & is_atk).sum() / n_atk if n_atk else np.nan
                ),
                "n_signatures_covered": int(frame.loc[m, "signature_id"].nunique()),
            }
        )
    return pd.DataFrame(out)


def run_purity_step(
    alert_groups: list,
    signature_ids: np.ndarray,
    gran: float = PAPER_GRANULARITY,
    scheme: WindowScheme | None = None,
) -> dict[str, pd.DataFrame]:
    """Step 1 end to end: purity table + coverage on W0 and on the pooled
    training windows. Returns {"w0_table", "w0_coverage", "pooled_table",
    "pooled_coverage", "windows"}."""
    scheme = scheme or build_baseline_split_scheme(alert_groups)
    bounds = training_window_bounds(scheme, gran)
    w0_start, w0_end = bounds[0]
    pooled_end = bounds[-1][1]  # windows are contiguous from 0

    w0 = signature_item_frame(
        alert_groups[w0_start:w0_end], signature_ids[w0_start:w0_end]
    )
    pooled = signature_item_frame(alert_groups[:pooled_end], signature_ids[:pooled_end])

    w0_table, pooled_table = purity_table(w0), purity_table(pooled)
    windows = pd.DataFrame(
        [
            {"step_i": i, "start": s, "end": e, "n_groups": e - s}
            for i, (s, e) in enumerate(bounds)
        ]
    )
    return {
        "w0_table": w0_table,
        "w0_coverage": coverage_summary(w0, w0_table),
        "pooled_table": pooled_table,
        "pooled_coverage": coverage_summary(pooled, pooled_table),
        "windows": windows,
    }


# --- Walk-forward ablation (Check 3 steps 2-3, Check 2 step 3) ---------------
#
# An arm = (mining variant, precision gate). All arms: RF, class-weighted,
# Base(5) base columns.
#   a        Base(5), no mining                      (feature_set "baseline")
#   b        Base(5) + mining on all items           (feature_set "symbolic")
#   c        Base(5) + mining without signature-derived items
#            (extra_exclude_fields=CSCAS_SIGNATURE_DERIVED_FIELDS)
#   b_g<g>   b with the Check 2 precision gate at g (rule_precision_gate) --
#            same mined schema as b, failing symbolic columns dropped before
#            the RF fit.
#
# Per step i (train Wi, test Wi+1): each mining variant is mined + encoded
# once and shared by every arm built on it; then one RF per (arm, seed)
# (mining is deterministic and cached, so seeds measure classifier variance
# only). Seen/unseen is per attack group in Wi+1: its SignatureID appeared in
# Wi ("seen") -- plus a stricter "seen_prior" (anywhere before Wi+1). Each
# symbolic variant also logs every mined rule's attack rate on Wi and Wi+1
# (rules.csv, rule_precision_gate.rule_table).


@dataclass(frozen=True)
class ArmSpec:
    feature_set: str  # "baseline" | "symbolic"
    extra_exclude: frozenset[str] | None = None
    gate_g: float | None = None
    label: str = ""

    @property
    def variant(self) -> tuple:
        return (self.feature_set, self.extra_exclude)


ARMS: dict[str, ArmSpec] = {
    "a": ArmSpec("baseline", label="Base(5)"),
    "b": ArmSpec("symbolic", label="Base(5) + mining (all items)"),
    "c": ArmSpec(
        "symbolic",
        CSCAS_SIGNATURE_DERIVED_FIELDS,
        label="Base(5) + mining (no signature items)",
    ),
    **{
        f"b_g{g:g}": ArmSpec(
            "symbolic", gate_g=g, label=f"Base(5) + mining, gate g={g:g}"
        )
        for g in GATE_GS
    },
}
ARM_LABELS = {k: v.label for k, v in ARMS.items()}


@dataclass
class SignatureAblationConfig:
    gran: float = PAPER_GRANULARITY
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    arms: tuple[str, ...] = ("a", "b", "c")
    model: str = "rf"
    mining_setting: str = "gr3_md1_mda4_rounds2"
    mining_settings_path: Path = (
        ROOT / "src/thesis/configs/monitor_eda_mining_setting.yaml"
    )
    threshold_mode: str = "fixed"
    calibrated_recall_target: float = 0.9
    split_time: str = CSCAS_BASELINE_SPLIT_TIME
    # None = all steps; an int caps the walk (smoke tests).
    max_steps: int | None = None
    force_remine: bool = False
    out_dir: Path | None = None
    # Output folder name: artifacts/experiments/<experiment>/ and
    # results/paper/<experiment with "-">/.
    experiment: str = "signature_leakage"
    log_rules: bool = True


def _window_days(rows: list) -> float:
    ts = [g.start_ts for g in rows if g.start_ts is not None]
    return (max(ts) - min(ts)) / 86400 if len(ts) > 1 else np.nan


def _step_metric_row(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    seen: np.ndarray,
    seen_prior: np.ndarray,
    n_days: float,
) -> dict:
    m = metrics_at_threshold(y, proba, threshold)
    pred = proba >= threshold
    atk = y == 1
    row = {
        **m,
        "pr_auc": float(average_precision_score(y, proba)) if atk.any() else np.nan,
        "n_days": n_days,
        "fp_per_day": m["fp"] / n_days if n_days and n_days > 0 else np.nan,
    }
    for tag, s in (("seen", seen), ("seen_prior", seen_prior)):
        for part, sel in ((tag, atk & s), (f"un{tag}", atk & ~s)):
            n = int(sel.sum())
            row[f"n_attack_{part}"] = n
            row[f"tp_{part}"] = int((sel & pred).sum())
            row[f"recall_{part}"] = row[f"tp_{part}"] / n if n else np.nan
    return row


def run_signature_ablation(config: SignatureAblationConfig) -> Path:
    """Walk-forward over `config.arms` x seeds. Writes, under a timestamped
    dir: per_step_results.csv (arm x seed x step), predictions.csv.gz (every
    attack row and every false positive of the test windows, with
    group_id / signature_id / seen flags / proba), rules.csv (every mined
    rule per step, train/next attack rate + gate verdicts), windows.csv,
    config.json, ablation_summary.csv. per_step_results and rules are
    rewritten after every step, so a crashed run keeps what it finished."""
    for arm in config.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")

    ctx = load_scenario_context(
        scenario="cscas",
        cache_dir=CSCAS_ALERT_GROUPS_CACHE_DIR,
        grouping=GroupingConfig(mode=CSCAS_PREGROUPED_METHOD),
        alerts_json_path=None,
        mining_settings_path=Path(config.mining_settings_path),
    )
    groups = ctx.alert_groups
    print("Building SignatureID lookup...")
    sig_ids = signature_ids_for(groups, load_cscas_signature_lookup())
    scheme = build_baseline_split_scheme(groups, config.split_time)
    bounds = training_window_bounds(scheme, config.gran)
    n_steps = (
        len(bounds) if config.max_steps is None else min(len(bounds), config.max_steps)
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = ROOT / "artifacts" / "experiments" / config.experiment
    out_dir = (Path(config.out_dir) if config.out_dir else base_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str)
    )
    pd.DataFrame(
        [
            {
                "step_i": i,
                "train_start": s,
                "train_end": e,
                "test_start": scheme.target_bounds(config.gran, i + 1)[0],
                "test_end": scheme.target_bounds(config.gran, i + 1)[1],
            }
            for i, (s, e) in enumerate(bounds)
        ]
    ).to_csv(out_dir / "windows.csv", index=False)
    print(
        f"Output → {out_dir}  ({n_steps} steps x {len(config.arms)} arms x {len(config.seeds)} seeds)"
    )

    # Arms grouped by mining variant, in first-appearance order.
    variants: dict[tuple, list[str]] = {}
    for arm in config.arms:
        variants.setdefault(ARMS[arm].variant, []).append(arm)

    step_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    rule_parts: list[pd.DataFrame] = []

    for i in range(n_steps):
        tr_start, tr_end = bounds[i]
        te_start, te_end = scheme.target_bounds(config.gran, i + 1)
        test_rows = groups[te_start:te_end]
        _, te_mask = labels_and_mask(test_rows)
        te_sig = sig_ids[te_start:te_end][te_mask]
        te_gid = np.array([g.group_id for g in test_rows])[te_mask]
        seen = np.isin(te_sig, np.unique(sig_ids[tr_start:tr_end]))
        seen_prior = np.isin(te_sig, np.unique(sig_ids[:te_start]))
        n_days = _window_days(test_rows)
        tr_labels, tr_mask = labels_and_mask(groups[tr_start:tr_end])
        y_fit = tr_labels[tr_mask].astype(int)

        for (feature_set, extra_excl), variant_arms in variants.items():
            cfg = ShortlistedConfig(
                feature_set=feature_set,
                mining_setting=config.mining_setting
                if feature_set == "symbolic"
                else None,
                granularity=config.gran,
                model=config.model,
            )
            t0 = time.perf_counter()
            fit = fit_window(
                cfg=cfg,
                scenario="cscas",
                alert_groups=groups,
                alert_groups_path=ctx.alert_groups_path,
                n_total=ctx.n_total,
                win_idx=i,
                base_schema=ctx.base_schema,
                mining_settings_by_name=ctx.mining_settings_by_name,
                mining_settings_path=ctx.mining_settings_path,
                threshold_mode=config.threshold_mode,
                calibrated_recall_target=config.calibrated_recall_target,
                scheme=scheme,
                force_remine=config.force_remine,
                extra_exclude_fields=extra_excl,
                model_seed=config.seeds[0],
            )
            if fit is None:
                print(
                    f"  [warn] step {i} {variant_arms}: fit_window returned None -- skipped"
                )
                continue
            prep_s = time.perf_counter() - t0
            X_te, y_te, _ = encode_target_window(
                groups, scheme, config.gran, i + 1, fit.schema
            )
            base_cols = list(fit.schema.base.features) if fit.schema.base else []

            if config.log_rules and fit.schema.symbolic is not None:
                rt = rule_table(fit.schema, fit.X_fit, y_fit, X_te, y_te)
                rt.insert(0, "step_i", i)
                rt.insert(
                    0,
                    "variant",
                    variant_arms[0]
                    if len(variant_arms) == 1
                    else "+".join(variant_arms),
                )
                rule_parts.append(rt)

            for arm in variant_arms:
                spec = ARMS[arm]
                if spec.gate_g is None:
                    cols = list(fit.X_fit.columns)
                else:
                    cols = base_cols + gated_feature_names(
                        fit.schema, fit.X_fit, y_fit, spec.gate_g
                    )
                X_fit_a, X_te_a = fit.X_fit[cols], X_te[cols]
                n_symbolic = len(cols) - len(base_cols) if fit.schema.symbolic else 0

                for k, seed in enumerate(config.seeds):
                    if k == 0 and spec.gate_g is None:
                        model, thr = fit.model, fit.threshold
                    else:
                        model = fit_scored_model(
                            config.model, X_fit_a, y_fit, seed=seed
                        )
                        thr = decide_threshold(
                            y_fit,
                            model.predict_proba(X_fit_a)[:, 1],
                            config.threshold_mode,
                            config.calibrated_recall_target,
                            model=model,
                        )
                    proba = model.predict_proba(X_te_a)[:, 1]
                    step_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "step_i": i,
                            "n_test": len(y_te),
                            "n_attack": int(y_te.sum()),
                            "n_features": X_te_a.shape[1],
                            "n_symbolic": n_symbolic,
                            "mining_cache_hit": fit.cache_hit,
                            "threshold": thr,
                            "prep_wall_s": prep_s,
                            **_step_metric_row(
                                y_te, proba, thr, seen, seen_prior, n_days
                            ),
                        }
                    )
                    keep = (y_te == 1) | (proba >= thr)
                    pred_parts.append(
                        pd.DataFrame(
                            {
                                "arm": arm,
                                "seed": seed,
                                "step_i": i,
                                "group_id": te_gid[keep],
                                "signature_id": te_sig[keep],
                                "y": y_te[keep],
                                "proba": proba[keep].astype(np.float32),
                                "pred": (proba[keep] >= thr),
                                "seen": seen[keep],
                                "seen_prior": seen_prior[keep],
                            }
                        )
                    )
                last = step_rows[-1]
                print(
                    f"  step {i:2d} arm {arm}: f1={last['f1']:.3f} pr_auc={last['pr_auc']:.3f} "
                    f"recall seen/unseen={last['recall_seen']:.3f}/{last['recall_unseen']:.3f} "
                    f"(n_unseen={last['n_attack_unseen']}, {n_symbolic} symbolic, "
                    f"{'cache' if fit.cache_hit else 'mined'}, {prep_s:.0f}s)"
                )

        pd.DataFrame(step_rows).to_csv(out_dir / "per_step_results.csv", index=False)
        if rule_parts:
            pd.concat(rule_parts, ignore_index=True).to_csv(
                out_dir / "rules.csv", index=False
            )

    pd.concat(pred_parts, ignore_index=True).to_csv(
        out_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    summary = summarize_ablation(pd.DataFrame(step_rows))
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    (out_dir / "summary.txt").write_text(summary.to_string(index=False) + "\n")
    print(summary.to_string(index=False))

    if config.out_dir is None and config.max_steps is None:
        mirror = RESULTS_DIR / "paper" / config.experiment.replace("_", "-") / ts
        try:
            shutil.copytree(out_dir, mirror, dirs_exist_ok=True)
            print(f"Mirrored → {mirror}")
        except OSError as exc:
            print(f"[warn] could not mirror to {mirror}: {exc}")
    return out_dir


SUMMARY_METRICS = ["f1", "precision", "recall", "pr_auc", "auc", "fp_per_day"]


def summarize_ablation(per_step: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: per seed, metrics are averaged over steps and
    seen/unseen recall is pooled over steps (sum tp / sum attacks -- unseen
    attacks per step can be few); then mean and std over seeds."""
    if per_step.empty:
        return pd.DataFrame()
    per_seed = per_step.groupby(["arm", "seed"])[SUMMARY_METRICS].mean()
    for part in ("seen", "unseen", "seen_prior", "unseen_prior"):
        g = per_step.groupby(["arm", "seed"])[[f"tp_{part}", f"n_attack_{part}"]].sum()
        per_seed[f"recall_{part}"] = g[f"tp_{part}"] / g[f"n_attack_{part}"].replace(
            0, np.nan
        )
        per_seed[f"n_attack_{part}"] = g[f"n_attack_{part}"]
    agg = per_seed.groupby("arm").agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.drop(
        columns=[
            c for c in agg.columns if c.startswith("n_attack_") and c.endswith("_std")
        ]
    )
    agg.insert(0, "arm_label", agg.index.map(ARM_LABELS))
    agg["n_seeds"] = per_seed.groupby("arm").size()
    return agg.reset_index()
