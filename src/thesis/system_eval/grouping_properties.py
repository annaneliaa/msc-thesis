"""
Grouping × mining, step 1 (GM_PLAN.md): describe what each grouping does to
the groups, without any mining, so the targeted runs can pick groupings that
differ in one property at a time.

Per (scenario, grouping), over the labelled (attack / benign) groups:

  counts       n_groups, n_attack, base_rate (over the labelled groups)
  reduction    1 - groups / alerts (all groups), to place each grouping
               setting on the thesis's reduction-vs-purity curve (Fig. 5.6)
  label purity mixed_group_share / mixed_alert_share: groups (and the alerts
               in them) whose alert labels mix benign ("false_positive") and
               attack, over ALL groups -- these are labelled "mixed" and
               dropped from mining and training. Grouping can't remove the
               label noise already in the time-window alert labels, but it
               decides how many mixed groups that noise produces; coarser
               groupings (higher reduction) produce more (thesis §5.2).
  content      ambiguous_attack_share: attack groups whose alert codes all
  ambiguity    also occur in benign groups of the same scenario -- the
               attack groups no content-based feature can separate (the
               label noise the comparison runs into).
  composition  alerts_per_group (mean / median), codes_per_group (mean),
               single_code_share, multi_host_share
  order        ordered_share: groups with >= 2 time-ordered alert steps
               (sorted_items) -- what a within-group sequence miner needs;
               duration_median_s
  unseen       unseen_attack_share (LOSO view): attack groups with an alert
               code never seen in an attack group of another scenario, under
               the same grouping.

Codes are the `short:` items (one per alert name; see
mining_mini_experiment._ait_short_codes); hosts the `host:` items.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis.schemas.groups import AlertGroup

BENIGN_ALERT_LABEL = "false_positive"


def _codes(g: AlertGroup) -> frozenset[str]:
    return frozenset(i[6:] for i in (g.raw_items or ()) if i.startswith("short:"))


def _hosts(g: AlertGroup) -> frozenset[str]:
    return frozenset(i[5:] for i in (g.raw_items or ()) if i.startswith("host:"))


def labelled_groups(groups: list[AlertGroup]) -> list[AlertGroup]:
    return [g for g in groups if g.group_label in ("attack", "benign")]


def group_properties(
    groups: list[AlertGroup], other_attack_codes: set[str] | None = None
) -> dict:
    """The properties above for one (scenario, grouping). `other_attack_codes`:
    codes seen in attack groups of the other scenarios (for unseen_attack_share);
    None skips it."""
    gs = labelled_groups(groups)
    if not gs:
        return {"n_groups": 0}
    attack = [g for g in gs if g.group_label == "attack"]
    benign = [g for g in gs if g.group_label == "benign"]
    benign_codes = set().union(*(_codes(g) for g in benign)) if benign else set()
    codes = [_codes(g) for g in gs]
    n_alerts = np.array([g.n_alerts for g in gs])
    durations = np.array(
        [(g.end_ts - g.start_ts) if g.end_ts is not None else 0 for g in gs]
    )
    # Purity over ALL groups: a group whose alerts mix benign and attack labels
    # is labelled "mixed" and dropped by labelled_groups (and by the
    # comparison's labelled()), so it must be counted before that filter.
    all_gs = list(groups)
    mixed = [g.group_label == "mixed" for g in all_gs]
    alerts_all = sum(g.n_alerts for g in all_gs)
    row = {
        "n_groups_all": len(all_gs),
        "n_alerts": alerts_all,
        "reduction": 1 - len(all_gs) / alerts_all if alerts_all else np.nan,
        "mixed_group_share": float(np.mean(mixed)),
        "mixed_alert_share": (
            sum(g.n_alerts for g in all_gs if g.group_label == "mixed") / alerts_all
            if alerts_all
            else np.nan
        ),
        "n_groups": len(gs),
        "n_attack": len(attack),
        "base_rate": len(attack) / len(gs),
        "ambiguous_attack_share": (
            float(
                np.mean([bool(_codes(g)) and _codes(g) <= benign_codes for g in attack])
            )
            if attack
            else np.nan
        ),
        "alerts_per_group_mean": float(n_alerts.mean()),
        "alerts_per_group_median": float(np.median(n_alerts)),
        "codes_per_group_mean": float(np.mean([len(c) for c in codes])),
        "single_code_share": float(np.mean([len(c) == 1 for c in codes])),
        "multi_host_share": float(np.mean([len(_hosts(g)) > 1 for g in gs])),
        "ordered_share": float(np.mean([len(g.sorted_items or ()) >= 2 for g in gs])),
        "duration_median_s": float(np.median(durations)),
    }
    if other_attack_codes is not None:
        row["unseen_attack_share"] = (
            float(np.mean([bool(_codes(g) - other_attack_codes) for g in attack]))
            if attack
            else np.nan
        )
    return row


def properties_table(
    groups_by: dict[tuple[str, str], list[AlertGroup]],
) -> pd.DataFrame:
    """One row per (scenario, grouping) in `groups_by`. Unseen shares compare
    each scenario against the other scenarios under the same grouping."""
    attack_codes: dict[tuple[str, str], set[str]] = {}
    for key, gs in groups_by.items():
        attack_codes[key] = set().union(
            *(_codes(g) for g in labelled_groups(gs) if g.group_label == "attack")
        )
    rows = []
    for (scenario, grouping), gs in groups_by.items():
        others = [k for k in groups_by if k[1] == grouping and k[0] != scenario]
        other_codes = (
            set().union(*(attack_codes[k] for k in others)) if others else None
        )
        rows.append(
            {
                "scenario": scenario,
                "grouping": grouping,
                **group_properties(gs, other_codes),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_grouping(table: pd.DataFrame) -> pd.DataFrame:
    """Mean over scenarios per grouping (plus how many scenarios it covers)."""
    num = table.drop(columns=["scenario"]).groupby("grouping")
    out = num.mean(numeric_only=True)
    out.insert(0, "n_scenarios", table.groupby("grouping")["scenario"].nunique())
    return out


def load_groups_by(
    scenarios: list[str], groupings: list[str], cache_dir: Path
) -> dict[tuple[str, str], list[AlertGroup]]:
    """Cached groups for every (scenario, grouping) whose cache exists."""
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    out = {}
    for s in scenarios:
        for m in groupings:
            d = Path(cache_dir) / s / "groups" / m
            if not (d / "alert_groups" / "alert_groups_raw.json").exists():
                continue
            out[(s, m)] = load_or_build_alert_groups(s, d)
    return out
