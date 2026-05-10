"""
Post-mining token abstraction.

Replaces raw alert tokens (e.g. "short:S-Tls-Hnd") with controlled semantic
abstractions (e.g. "tls_invalid_handshake"), then removes hierarchical
redundancy so that parent concepts are dropped whenever a child concept is
already present in the same pattern.

Abstraction levels (index into each map entry's value list):
  0 = mid-level / most specific  (default, recommended)
  1 = coarse / most general

Tokens that have no entry in the map are kept as-is unless keep_unmapped=False.
Repeat-encoded suffixes (__repeat_2, __repeat_3_4, __repeat_5_plus) are
preserved through abstraction: "short:S-Tls-Hnd__repeat_2" →
"tls_invalid_handshake__repeat_2".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPEAT_SUFFIXES = ("__repeat_2", "__repeat_3_4", "__repeat_5_plus")

_MAIL_HOST = "host:mail_host"
_MULTIPLE_MAIL_HOSTS = "host:multiple_mail_hosts"


def _is_mail_host(item: str) -> bool:
    return item.startswith("host:") and item.split("host:", 1)[1].endswith("_mail")


def abstract_mail_hosts(items: set[str]) -> set[str]:
    """
    Replace host:<name>_mail tokens with host:mail_host.

    Adds host:multiple_mail_hosts when 2+ distinct mail host names are present.
    """
    mail_hosts = {item for item in items if _is_mail_host(item)}
    if not mail_hosts:
        return items.copy()
    result = (items - mail_hosts) | {_MAIL_HOST}
    if len(mail_hosts) >= 2:
        result.add(_MULTIPLE_MAIL_HOSTS)
    return result


def _abstract_mail_hosts_pattern(pattern: tuple[str, ...]) -> tuple[str, ...]:
    """Apply mail host abstraction to a flat itemset pattern (tuple of tokens)."""
    mail_hosts = {t for t in pattern if _is_mail_host(t)}
    if not mail_hosts:
        return pattern
    rest = [t for t in pattern if t not in mail_hosts]
    rest.append(_MAIL_HOST)
    if len(mail_hosts) >= 2:
        rest.append(_MULTIPLE_MAIL_HOSTS)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in rest:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return tuple(deduped)


def abstract_mail_hosts_mined_df(
    df: pd.DataFrame,
    column: str = "itemset",
) -> pd.DataFrame:
    """
    Apply mail host abstraction to mined itemset patterns post-mining.

    Updates k column; collapses identical patterns (keeps highest |support_diff|).
    """
    if df.empty or column not in df.columns:
        return df.copy()

    out = df.copy()
    out[column] = out[column].apply(_abstract_mail_hosts_pattern)

    if "k" in out.columns:
        out["k"] = out[column].apply(len)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=[column])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=[column]).reset_index(drop=True)

    return out


def abstract_mail_hosts_or_clauses_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mail host abstraction to OR-clause patterns post-mining.

    Each AND-clause is abstracted independently; duplicate clauses within a row
    are dropped.
    """
    if df.empty or "clauses" not in df.columns:
        return df.copy()

    out = df.copy()

    def _abstract_clauses(
        clauses: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        seen: set[tuple[str, ...]] = set()
        result: list[tuple[str, ...]] = []
        for clause in clauses:
            abstracted = _abstract_mail_hosts_pattern(clause)
            if abstracted and abstracted not in seen:
                seen.add(abstracted)
                result.append(abstracted)
        return tuple(result)

    out["clauses"] = out["clauses"].apply(_abstract_clauses)
    out = out[out["clauses"].apply(len) > 0].reset_index(drop=True)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=["clauses"])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=["clauses"]).reset_index(drop=True)

    return out


def abstract_mail_hosts_item_sequence_df(
    df: pd.DataFrame,
    column: str = "sequence",
) -> pd.DataFrame:
    """
    Apply mail host abstraction to mined item-level sequence patterns.

    Each step is a single token; host:*_mail → host:mail_host.
    Updates sequence_str and k columns; collapses duplicate patterns.
    """
    if df.empty or column not in df.columns:
        return df.copy()

    out = df.copy()
    out[column] = out[column].apply(
        lambda seq: tuple(_MAIL_HOST if _is_mail_host(t) else t for t in seq)
    )

    if "sequence_str" in out.columns:
        out["sequence_str"] = out[column].apply(lambda x: " -> ".join(x))
    if "k" in out.columns:
        out["k"] = out[column].apply(len)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=[column])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=[column]).reset_index(drop=True)

    return out


def abstract_mail_hosts_itemset_sequence_df(
    df: pd.DataFrame,
    column: str = "sequence",
) -> pd.DataFrame:
    """
    Apply mail host abstraction to mined itemset-level sequence patterns.

    Each step is a frozenset; abstraction (including multiple_mail_hosts detection)
    is applied independently per step.
    """
    if df.empty or column not in df.columns:
        return df.copy()

    out = df.copy()
    out[column] = out[column].apply(
        lambda seq: tuple(frozenset(abstract_mail_hosts(set(step))) for step in seq)
    )

    if "k" in out.columns:
        out["k"] = out[column].apply(len)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=[column])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=[column]).reset_index(drop=True)

    return out


def load_abstraction_map(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_ancestor_index(abstraction_map: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    For each abstracted token, return the set of tokens that are its ancestors
    (i.e., more general concepts in the same hierarchy chain).

    Given map entry "short:X" -> ["specific", "general"]:
    - ancestors["specific"] includes "general"
    """
    ancestors: dict[str, set[str]] = {}
    for levels in abstraction_map.values():
        for i, token in enumerate(levels):
            if token not in ancestors:
                ancestors[token] = set()
            for j in range(i + 1, len(levels)):
                ancestors[token].add(levels[j])
    return ancestors


def _split_repeat_suffix(token: str) -> tuple[str, str]:
    for suffix in _REPEAT_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)], suffix
    return token, ""


def _remove_hierarchical_redundancy(
    tokens: list[str],
    ancestor_index: dict[str, set[str]],
) -> list[str]:
    """
    Drop tokens that are ancestors of another token in the same collection.

    Repeat suffixes are handled: tls_invalid_handshake__repeat_2 + tls__repeat_2
    → drops tls__repeat_2.  Mixed-suffix pairs (different repeat buckets) are
    not treated as hierarchically redundant.
    """
    token_set = set(tokens)
    to_remove: set[str] = set()
    for token in token_set:
        base, suffix = _split_repeat_suffix(token)
        for anc_base in ancestor_index.get(base, set()):
            anc_token = anc_base + suffix
            if anc_token in token_set:
                to_remove.add(anc_token)
    return [t for t in tokens if t not in to_remove]


def _abstract_tokens(
    tokens: tuple[str, ...],
    abstraction_map: dict[str, list[str]],
    ancestor_index: dict[str, set[str]],
    level: int,
    keep_unmapped: bool,
) -> tuple[str, ...]:
    """Map each token to its abstracted form, deduplicate, remove hierarchy."""
    mapped: list[str] = []
    for token in tokens:
        base, suffix = _split_repeat_suffix(token)
        levels = abstraction_map.get(base)
        if levels is not None:
            idx = min(level, len(levels) - 1)
            mapped.append(levels[idx] + suffix)
        elif keep_unmapped:
            mapped.append(token)

    seen: set[str] = set()
    deduped: list[str] = []
    for t in mapped:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return tuple(_remove_hierarchical_redundancy(deduped, ancestor_index))


def abstract_mined_df(
    df: pd.DataFrame,
    abstraction_map: dict[str, list[str]],
    level: int = 0,
    column: str = "itemset",
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Replace raw tokens in all patterns with their abstracted forms.

    - Updates 'k' column if present.
    - Drops patterns that become empty.
    - Collapses rows that become identical (keeps highest |support_diff|).

    Returns a new DataFrame; does not modify df in place.
    """
    if df.empty or column not in df.columns:
        return df.copy()

    ancestor_index = _build_ancestor_index(abstraction_map)
    out = df.copy()

    out[column] = out[column].apply(
        lambda tokens: _abstract_tokens(
            tokens, abstraction_map, ancestor_index, level, keep_unmapped
        )
    )

    if "k" in out.columns:
        out["k"] = out[column].apply(len)

    out = out[out[column].apply(len) > 0].reset_index(drop=True)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=[column])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=[column]).reset_index(drop=True)

    return out


def abstract_or_clauses_df(
    df: pd.DataFrame,
    abstraction_map: dict[str, list[str]],
    level: int = 0,
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Apply token abstraction to OR-clause patterns.

    Each row's 'clauses' column is a tuple[tuple[str, ...], ...]. Each
    AND-clause is abstracted independently; duplicate clauses within a row are
    dropped.  Rows whose clause set becomes identical after abstraction are
    collapsed (highest |support_diff| wins).
    """
    if df.empty or "clauses" not in df.columns:
        return df.copy()

    ancestor_index = _build_ancestor_index(abstraction_map)
    out = df.copy()

    def _abstract_clauses(
        clauses: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        seen: set[tuple[str, ...]] = set()
        result: list[tuple[str, ...]] = []
        for clause in clauses:
            abstracted = _abstract_tokens(
                clause, abstraction_map, ancestor_index, level, keep_unmapped
            )
            if abstracted and abstracted not in seen:
                seen.add(abstracted)
                result.append(abstracted)
        return tuple(result)

    out["clauses"] = out["clauses"].apply(_abstract_clauses)

    out = out[out["clauses"].apply(len) > 0].reset_index(drop=True)

    if "support_diff" in out.columns:
        out = (
            out.assign(_abs_diff=out["support_diff"].abs())
            .sort_values("_abs_diff", ascending=False)
            .drop_duplicates(subset=["clauses"])
            .drop(columns=["_abs_diff"])
            .reset_index(drop=True)
        )
    else:
        out = out.drop_duplicates(subset=["clauses"]).reset_index(drop=True)

    return out
