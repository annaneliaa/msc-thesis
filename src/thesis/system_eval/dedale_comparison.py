"""
Mining comparison on DEDALE (Lanvin & Majorczyk, ESORICS 2025 Workshops), on
the labelled "DEDALE Sigma" alert set from the CATS repo (Uetz et al.,
arXiv 2609.02465): Sigma rules run on DEDALE's Windows event logs, labels
reviewed by hand. 1,819 alerts over 27 days, 51 attack (FINDINGS.md,
2026-11-22).

Same treatment as the AIT-ADS comparison (mining_comparison.run_step, reused
as is): the same item table, precision gate, condition + firing dedupe,
budget K, class-weighted RF at 0.5, per-group test scores and the paired
bootstrap over test groups. Only loading, grouping and the split are
DEDALE-specific, and they live here so mining_comparison.py is untouched.

Alerts -> AIT-ADS-style tokens, so the existing item table, base features
(hour_of_day, n_alerts, n_sigs, n_hosts, n_shorts) and look-back sequences
work unchanged:
  - host:<hostname>                     ("host" field)
  - short:S-<Tactic>-<RuleTitle>        ("short"; short_cat = S-<Tactic>,
                                         the rule's first ATT&CK tactic)
  - sig:<content token>                 ("sig"): rule level, every ATT&CK
    tactic and technique of the rule, event id, log channel, the process
    image and parent image (basenames) and the user kind (system / account /
    none). The user and host names themselves are not sig tokens (identities,
    not content).

Split: one chronological cut (`cut`). The default, 2025-01-07 00:00 UTC, is
the only cut with attack groups on both sides that doesn't split an attack
burst: training holds the infection stage (CLIENT2, 06/01), test the
PrintNightmare lateral movement and the collection tunnel (CLIENT1, 08/01 and
10/01). The dataset's recommended weeks 1-2 / 3-4 cut ("weeks") has no
training attacks; the 70/30 chronological cut ("70_30") has no test attacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator

import importlib

import pandas as pd

from thesis.mining.temporal.lookback import HostAlertIndex, attach_lookback_steps
from thesis.paths import ROOT
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.mining_comparison import (
    SCORES_FILE,
    ComparisonConfig,
    DatasetSpec,
    Step,
    _window_days,
    ait_ads_base_schema,
    relabel_mixed_as_attack,
    run_step,
    write_bootstrap_summary,
    write_summary,
)
from thesis.system_eval.mining_mini_experiment import (
    TEMPORAL,
    TOKEN_ARMS,
    _ait_short_codes,
    _labels,
    _repo_commit,
    labelled,
    lookback_token_fields,
)

# thesis.grouping re-exports a function named group_alerts, shadowing the module.
_G = importlib.import_module("thesis.grouping.group_alerts")

DATASET = "dedale"
DEDALE_SIGMA_PATH = ROOT / "data" / "dedale" / "dedale_sigma.jsonl"

CUTS = {
    "attack_week": "2025-01-07T00:00:00Z",
    "weeks": "2025-01-06T00:00:00Z",
}

# grouping name -> (repo grouping function, its time parameter in seconds).
# The group method is prefixed "dedale_" so it can't collide with another
# dataset's method prefix in the base-feature encoder.
GROUPINGS: dict[str, Callable] = {
    "cscas_grouping": lambda al, s: _G.group_alerts_cscas_by_group(al),
    "time_delta": lambda al, s: _G.group_alerts_time_delta_by_group(al, s),
    "time_delta_host": lambda al, s: _G.group_alerts_time_delta_host_by_group(al, s),
    "fixed_window": lambda al, s: _G.group_alerts_fixed_window_by_group(al, int(s)),
    "fixed_window_host": lambda al, s: _G.group_alerts_fixed_window_host_by_group(
        al, int(s)
    ),
}
DEFAULT_SECONDS = {
    "cscas_grouping": None,
    "time_delta": _G.TIME_DELTA_SECONDS,
    "time_delta_host": _G.TIME_DELTA_SECONDS,
    "fixed_window": _G.FIXED_WINDOW_SECONDS,
    "fixed_window_host": _G.FIXED_WINDOW_SECONDS,
}


@dataclass
class DedaleConfig:
    grouping: str = "cscas_grouping"
    # Grouping time parameter (time_delta: gap, fixed_window: window size);
    # None = the repo default. cscas_grouping keeps its own 60 s / 300 s.
    grouping_seconds: float | None = None
    # A preset name from CUTS, "70_30" (first 70% of groups train), or an
    # ISO timestamp.
    cut: str = "attack_week"
    path: Path = DEDALE_SIGMA_PATH


# --- Alerts -> tokens ---------------------------------------------------------


@dataclass
class DedaleAlert:
    """One Sigma alert, in the shape the repo's grouping functions read."""

    alert_id: str
    ts: float
    host: str | None
    signature: str | None  # the Sigma rule title
    short: str
    sig_tokens: frozenset[str]
    misuse: bool


def _camel(text: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^0-9A-Za-z]+", text) if w)


def _basename(path) -> str | None:
    if not path or not isinstance(path, str):
        return None
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower() or None


def _user_kind(user) -> str:
    if not user or not isinstance(user, str):
        return "none"
    u = user.upper()
    if "SYSTEM" in u or "NT AUTHORITY" in u or u.endswith("$"):
        return "system"
    return "account"


def short_code(rule_title: str, tactics: list[str]) -> str:
    """S-<Tactic>-<RuleTitle>, no inner hyphens (attribute_features._short_parts
    splits on them: short_src "S", short_cat "S-<Tactic>")."""
    tactic = _camel(tactics[0]) if tactics else "None"
    return f"S-{tactic}-{_camel(rule_title)}"


def rule_tactics(tags) -> list[str]:
    """ATT&CK tactics from the rule's raw tags ("attack.defense-evasion"), in
    tag order. Not features.rule_attacktactics: the CATS normalizer's regex
    stops at the hyphen ("defense-evasion" -> "defense")."""
    out = []
    for t in tags or ():
        t = t.lower()
        if t.startswith("attack.") and not re.match(r"attack\.t\d", t):
            out.append(t[len("attack.") :])
    return out


def alert_sig_tokens(features: dict, log: dict, tactics: list[str]) -> frozenset[str]:
    tokens = {f"level_{features.get('rule_level') or 'none'}"}
    tokens |= {f"tactic_{_camel(t)}" for t in tactics}
    tokens |= {f"tech_{t.lower()}" for t in features.get("rule_attacktechniques") or ()}
    if log.get("EventID") is not None:
        tokens.add(f"eid_{log['EventID']}")
    channel = log.get("Channel")
    if channel:
        tokens.add(f"chan_{_camel(channel.split('/')[0])}")
    image = _basename(log.get("Image"))
    if image:
        tokens.add(f"img_{image}")
    parent = _basename(log.get("ParentImage"))
    if parent:
        tokens.add(f"parent_{parent}")
    tokens.add(f"user_{_user_kind(features.get('username'))}")
    return frozenset(tokens)


@lru_cache(maxsize=2)
def load_dedale_alerts(path: Path = DEDALE_SIGMA_PATH) -> tuple[DedaleAlert, ...]:
    alerts = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            feat, log = r["features"], r["full_alert"]["log"]
            tactics = rule_tactics(r["full_alert"].get("tags"))
            ts = pd.Timestamp(feat["timestamp"]).timestamp()
            alerts.append(
                DedaleAlert(
                    alert_id=r["metadata"]["alert_id"],
                    ts=ts,
                    host=(feat.get("hostname") or "").lower() or None,
                    signature=feat["rule_name"],
                    short=short_code(feat["rule_name"], tactics),
                    sig_tokens=alert_sig_tokens(feat, log, tactics),
                    misuse=bool(r["metadata"]["misuse"]),
                )
            )
    return tuple(sorted(alerts, key=lambda a: a.ts))


def alert_items(a: DedaleAlert) -> set[str]:
    items = {f"short:{a.short}"} | {f"sig:{t}" for t in a.sig_tokens}
    if a.host:
        items.add(f"host:{a.host}")
    return items


# --- Groups -------------------------------------------------------------------


def build_dedale_groups(config: DedaleConfig) -> list[AlertGroup]:
    """Every DEDALE alert group (attack / benign / mixed), in time order."""
    if config.grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {config.grouping!r}")
    alerts = list(load_dedale_alerts(Path(config.path)))
    seconds = (
        config.grouping_seconds
        if config.grouping_seconds is not None
        else DEFAULT_SECONDS[config.grouping]
    )
    by_group = GROUPINGS[config.grouping](alerts, seconds)
    method = f"{DATASET}_{config.grouping}"
    groups = []
    for gid, members in by_group.items():
        members = sorted(members, key=lambda a: a.ts)
        per_alert = [alert_items(a) for a in members]
        n_att = sum(a.misuse for a in members)
        label = "attack" if n_att == len(members) else ("mixed" if n_att else "benign")
        groups.append(
            AlertGroup(
                alert_group_id=f"{DATASET}:{gid}",
                group_id=f"{DATASET}:{gid}",
                method=method,
                start_ts=int(members[0].ts),
                end_ts=int(members[-1].ts),
                n_alerts=len(members),
                group_label=label,
                raw_items=set().union(*per_alert),
                sorted_items=per_alert,
                alert_ids=[a.alert_id for a in members],
                alert_labels={"attack" if a.misuse else "benign" for a in members},
            )
        )
    return sorted(groups, key=lambda g: (g.start_ts, g.group_id))


def raw_alert_frame(path: Path = DEDALE_SIGMA_PATH) -> pd.DataFrame:
    """time / host / short / signature per alert, for HostAlertIndex."""
    alerts = load_dedale_alerts(Path(path))
    return pd.DataFrame(
        {
            "time": [int(a.ts) for a in alerts],
            "host": [a.host or "" for a in alerts],
            "short": [a.short for a in alerts],
            "signature": [a.signature or "" for a in alerts],
        }
    )


def attach_dedale_lookback(
    groups: list[AlertGroup], token_arm: str, window_minutes: int, path: Path
) -> None:
    """Per-host look-back steps (mining/temporal/lookback.py) from the DEDALE
    alerts, with the token arm's look-back fields. Small, so not cached."""
    index = HostAlertIndex(raw_alert_frame(path), lookback_token_fields(token_arm))
    attach_lookback_steps(
        groups, alerts=None, window_minutes=window_minutes, index=index
    )


# --- Split and spec -----------------------------------------------------------


def split_groups(
    groups: list[AlertGroup], cut: str
) -> tuple[list[AlertGroup], list[AlertGroup], str]:
    """(train, test, cut description). `groups` are labelled, in time order."""
    if cut == "70_30":
        n = int(len(groups) * 0.7)
        return groups[:n], groups[n:], "first 70% of groups"
    ts = pd.Timestamp(CUTS.get(cut, cut))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    t = ts.timestamp()
    return (
        [g for g in groups if g.start_ts < t],
        [g for g in groups if g.start_ts >= t],
        ts.isoformat(),
    )


def dedale_spec(config: ComparisonConfig) -> DatasetSpec:
    include, exclude = TOKEN_ARMS[config.token_arm]
    return DatasetSpec(
        dataset=DATASET,
        base_schema=ait_ads_base_schema(config.base_exclude_features),
        # The group's rule short codes, like AIT-ADS under every grouping.
        signature_of=_ait_short_codes,
        exclude_fields=set(exclude) | set(config.extra_exclude_fields),
        include_fields=set(include),
        # Labels are per alert and hand-reviewed (no attack time windows), so
        # the benign side uses the same g as the attack side.
        benign_gate_g=None,
        attach_steps=lambda groups: None,
    )


def dedale_steps(
    config: ComparisonConfig, dconfig: DedaleConfig
) -> tuple[
    Callable[[], Iterator[tuple[Step, DatasetSpec]]], pd.DataFrame, pd.DataFrame
]:
    groups = build_dedale_groups(dconfig)
    n_mixed = sum(g.group_label == "mixed" for g in groups)
    if config.mixed_as_attack:
        relabel_mixed_as_attack(groups)
        print(f"  {n_mixed} mixed group(s) labelled attack", flush=True)
    elif n_mixed:
        print(f"  {n_mixed} mixed group(s) dropped", flush=True)
    groups = labelled(groups)
    tr, te, cut_desc = split_groups(groups, dconfig.cut)
    row = {
        "step_i": 0,
        "scenario": DATASET,
        "grouping": dconfig.grouping,
        "grouping_seconds": dconfig.grouping_seconds,
        "cut": cut_desc,
        "n_train": len(tr),
        "n_train_attack": int(_labels(tr).sum()) if tr else 0,
        "n_test": len(te),
        "n_test_attack": int(_labels(te).sum()) if te else 0,
        "test_days": _window_days(te) if te else 0.0,
        "n_mixed": n_mixed,
    }
    skipped = pd.DataFrame()
    plan = [row]
    if row["n_train_attack"] == 0 or row["n_test_attack"] == 0:
        side = "training" if row["n_train_attack"] == 0 else "test"
        print(f"  [skip] no attack groups on the {side} side of {cut_desc}")
        skipped = pd.DataFrame([{**row, "reason": f"no {side} attacks"}])
        plan = []

    def steps():
        for r in plan:
            if any(m in TEMPORAL for m in config.methods):
                attach_dedale_lookback(
                    groups,
                    config.token_arm,
                    config.mining.lookback_minutes,
                    Path(dconfig.path),
                )
            yield Step(0, tr, te, r["test_days"], scenario=DATASET), dedale_spec(config)

    return steps, pd.DataFrame(plan), skipped


# --- Run ----------------------------------------------------------------------


def run_dedale_comparison(config: ComparisonConfig, dconfig: DedaleConfig) -> Path:
    """mining_comparison.run_comparison's loop on the DEDALE step. Writes the
    same files (per_step_results.csv, features.csv, windows.csv, config.json,
    summary.csv, helps_hurts.csv, test_scores.csv.gz, bootstrap.csv/.txt)."""
    if config.stage not in ("all", "mine", "train"):
        raise ValueError(f"unknown stage {config.stage!r}")
    if config.stage in ("mine", "train") and config.mining_cache_dir is None:
        raise ValueError(f"stage {config.stage!r} needs a mining_cache_dir")
    from thesis.system_eval.mining_comparison import default_out_dir

    out_dir = (
        Path(config.out_dir) if config.out_dir else default_out_dir(config.experiment)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dedale": asdict(dconfig),
                "repo_commit": _repo_commit(),
            },
            indent=2,
            default=str,
        )
    )
    steps, windows, skipped = dedale_steps(config, dconfig)
    windows.to_csv(out_dir / "windows.csv", index=False)
    if not skipped.empty:
        skipped.to_csv(out_dir / "skipped.csv", index=False)
    if windows.empty:
        print(f"No steps to run → {out_dir}")
        return out_dir
    print(
        f"Output → {out_dir}  (1 step x "
        f"{len(config.methods) + config.include_no_mining} arms x "
        f"{len(config.seeds)} seeds)"
    )
    rows: list[dict] = []
    feats: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] | None = [] if config.save_test_scores else None
    for step, spec in steps():
        print(
            f"[{DATASET} {dconfig.grouping}] train {len(step.train)} "
            f"({int(_labels(step.train).sum())} attack), test {len(step.test)} "
            f"({int(_labels(step.test).sum())} attack)"
        )
        r, f = run_step(config, step, spec, scores)
        rows += r
        feats += f
    if config.stage == "mine":
        print(f"Mining cached → {config.mining_cache_dir}")
        return out_dir
    pd.DataFrame(rows).to_csv(out_dir / "per_step_results.csv", index=False)
    if feats:
        pd.concat(feats, ignore_index=True).to_csv(
            out_dir / "features.csv", index=False
        )
    if scores:
        pd.concat(scores, ignore_index=True).to_csv(out_dir / SCORES_FILE, index=False)
    write_summary(out_dir, pd.DataFrame(rows), pd.concat(feats) if feats else None)
    if scores:
        write_bootstrap_summary(out_dir, pd.concat(scores, ignore_index=True))
    return out_dir


def flag_all_f1(groups: list[AlertGroup]) -> float:
    """F1 of flagging every group attack (the reference next to every F1)."""
    y = _labels(groups)
    if not y.any():
        return float("nan")
    p = y.mean()
    return float(2 * p / (1 + p))
