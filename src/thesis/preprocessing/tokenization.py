from __future__ import annotations

import re
from thesis.schemas.preprocessing import ParsedAlert, TokenizedAlert
import pandas as pd

SIGNATURE_TOKEN_WHITELIST = {
    "access",
    "apache",
    "authentication",
    "code",
    "database",
    "directory",
    "dns",
    "domain",
    "entropy",
    "failed",
    "forbidden",
    "handshake",
    "invalid",
    "login",
    "message",
    "parameter",
    "query",
    "request",
    "server",
    "status",
    "success",
    "suspicious",
    "tls",
    "update",
    "url",
    "user",
}

SIGNATURE_TOKEN_BLACKLIST = {
    # vendor / product / engine names
    "wazuh",
    "suricata",
    "aminer",
    "clamav",
    "dovecot",
    "pam",
    "apache2",
    # generic / noisy meta words
    "alert",
    "alerts",
    "info",
    "event",
    "events",
    "log",
    "logs",
    "attempt",
    "attempts",
    "new",
    "same",
    "source",
    # short / noisy protocol crumbs
    "id",
    "ip",
    "et",
    "tld",
    "biz",
    # generic filler words
    "the",
    "and",
    "or",
    "to",
    "for",
    "from",
    "in",
    "of",
    "on",
    "by",
    "with",
    "a",
    "an",
}


def normalize_text(text: str) -> str:
    """
    Lowercase and collapse non-alphanumeric separators.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_signature_tokens(
    signature: str,
    whitelist: set[str] = SIGNATURE_TOKEN_WHITELIST,
    blacklist: set[str] = SIGNATURE_TOKEN_BLACKLIST,
) -> set[str]:
    """
    Extract whitelisted semantic tokens from a raw alert signature string.

    Example:
        'Wazuh: ClamAV database update'
        -> {'database', 'update'}
    """
    if pd.isna(signature):
        return set()

    text = str(signature).lower()

    # split into alphabetic tokens only
    raw_tokens = re.findall(r"[a-z]+", text)

    tokens = {tok for tok in raw_tokens if tok in whitelist and tok not in blacklist}

    return tokens


def tokenize_name_to_signature_substrings(name: str | None) -> set[str]:
    """
    Extract simple mining substrings from alert name.
    """
    if not name:
        return set()

    cleaned = normalize_text(name)
    tokens = extract_signature_tokens(cleaned)
    return {f"sig:{tok}" for tok in tokens}


def build_feature_tokens(alert: ParsedAlert) -> set[str]:
    """
    Tokens for grouping of alerts.
    """
    tokens: set[str] = set()

    if alert.short:
        tokens.add(f"short:{alert.short}")
    if alert.host:
        tokens.add(f"host:{alert.host}")
    if alert.name:
        tokens.add(f"name:{normalize_text(alert.name)}")
        tokens |= tokenize_name_to_signature_substrings(alert.name)

    # if alert.ip:
    #     tokens.add(f"ip:{alert.ip}")

    return tokens


def tokenize_alert(alert: ParsedAlert) -> TokenizedAlert:
    """
    Convert a ParsedAlert into a TokenizedAlert.
    """
    tokens = build_feature_tokens(alert)

    return TokenizedAlert(
        alert_id=alert.alert_id,
        ts=alert.ts,
        time_norm=alert.time_norm,
        name=alert.name,
        ip=alert.ip,
        host=alert.host,
        short=alert.short,
        time_label=alert.time_label,
        event_label=alert.event_label,
        tokens=tokens,
        raw=alert.raw.copy() if alert.raw else {},
    )
