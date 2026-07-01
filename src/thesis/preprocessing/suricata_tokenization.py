from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from thesis.preprocessing.tokenization import normalize_text

# --- Constants ---

# Words that carry no discriminating signal in Suricata rule descriptions.
SIGNATURE_NOISE_WORDS = {
    # conjunctions / prepositions
    "the",
    "a",
    "an",
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
    "via",
    "at",
    "as",
    "is",
    "are",
    "was",
    "using",
    # match-variant suffixes (M1, M2, M3 …)
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    # generic structural noise in rule text
    "inbound",
    "outbound",
    "attempt",
    "attempts",
    "request",
    "requests",
    "response",
    "responses",
    "detected",
    "related",
    "activity",
    # attribution / group labels that don't describe the threat
    "group",
    "poc",
}

# 2-char tokens that are domain-meaningful abbreviations and should be kept.
# Everything else at length 2 is dropped (e.g. "dd", "ed", "un", "hx").
SHORT_TOKEN_WHITELIST = {
    "ua",  # user agent
    "wp",  # WordPress
    "ip",  # IP address
    "ms",  # Microsoft
    "id",  # identifier
    "rce",  # remote code execution
}

# Words kept as unigrams but excluded from bigram generation.
# These are prose-filler words in rule names that don't form informative pairs.
BIGRAM_SKIP_WORDS = {
    "closing",
    "plus",
    "line",
    "comment",
    "string",
    "general",
    "generic",
    "common",
    "various",
    "multiple",
    "large",
    "small",
    "long",
    "version",
    "style",
    "type",
    "format",
    "named",
    "called",
    "known",
    "based",
}

# Words that signal detection confidence — isolated so callers can test
# whether including them improves or hurts downstream mining.
QUALIFIER_WORDS = {
    "possible",
    "suspected",
    "observed",
    "suspicious",
    "likely",
    "attempted",
}

CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d+\b", re.IGNORECASE)
BRACKETED_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
RULESET_CATEGORY_PATTERN = re.compile(r"^(ET|ETPRO|GPL)\s+(\S+)\s+(.*)", re.DOTALL)


# --- Data structures ---


@dataclass
class TokenizedSignature:
    signature_id: int
    ruleset: str  # ET / ETPRO / GPL  — separate categorical, not in tokens
    category: str  # EXPLOIT / MALWARE / WEB_SERVER / etc.
    description: str  # raw description after ruleset+category stripped
    cve_refs: set[str] = field(default_factory=set)  # {"CVE-2021-44228", …}
    qualifiers: set[str] = field(default_factory=set)  # {"possible", "observed", …}
    tokens: set[str] = field(default_factory=set)  # unigrams + bigrams from description


# --- Parsing helpers ---


def parse_signature_text(text: str) -> tuple[str, str, str]:
    """
    Split 'ET EXPLOIT Some description' into (ruleset, category, description).
    Returns ("", "", text) when format is not recognised.
    """
    if pd.isna(text):
        return "", "", ""
    text = str(text).strip()
    # Strip attribution tags like [PT OPEN] or [401TRG] before parsing
    text = BRACKETED_TAG_PATTERN.sub("", text).strip()
    m = RULESET_CATEGORY_PATTERN.match(text)
    if not m:
        return "", "", text
    return m.group(1).upper(), m.group(2).upper(), m.group(3).strip()


def extract_cve_refs(text: str) -> set[str]:
    """Extract CVE identifiers, normalised to uppercase."""
    return {m.upper() for m in CVE_PATTERN.findall(text)}


def description_to_tokens(
    description: str,
    noise_words: set[str] = SIGNATURE_NOISE_WORDS,
    qualifier_words: set[str] = QUALIFIER_WORDS,
    short_whitelist: set[str] = SHORT_TOKEN_WHITELIST,
    bigram_skip_words: set[str] = BIGRAM_SKIP_WORDS,
    include_qualifiers: bool = True,
    include_bigrams: bool = True,
) -> tuple[set[str], set[str]]:
    """
    Tokenize a description string into (token_set, qualifier_set).

    CVE refs should be extracted before calling this — they are stripped here
    rather than tokenised, since they are handled separately.
    """
    # Remove CVE patterns before lowercasing so they don't produce noisy tokens
    desc = CVE_PATTERN.sub(" ", description)
    # Normalise: lowercase, collapse non-alphanumeric to spaces
    desc = normalize_text(desc)

    raw_tokens = desc.split()

    quals: set[str] = set()
    content_tokens: list[str] = []
    for tok in raw_tokens:
        if tok in qualifier_words:
            quals.add(tok)
        elif tok in noise_words or tok.isdigit() or len(tok) <= 1:
            pass
        elif len(tok) == 2 and tok not in short_whitelist:
            pass  # drop 2-char fragments not in the domain whitelist
        else:
            content_tokens.append(tok)

    token_set: set[str] = set(content_tokens)

    if include_qualifiers:
        token_set |= {f"qual:{q}" for q in quals}

    if include_bigrams:
        # Deduplicate consecutive identical tokens (e.g. ms-sql → ms sql sql → ms sql),
        # then exclude prose-filler words so only content terms form bigram pairs.
        bigram_candidates: list[str] = []
        prev: str | None = None
        for tok in content_tokens:
            if tok == prev:
                continue  # skip consecutive duplicate (artefact of hyphen splitting)
            prev = tok
            if tok not in bigram_skip_words:
                bigram_candidates.append(tok)
        for a, b in zip(bigram_candidates, bigram_candidates[1:]):
            token_set.add(f"{a}_{b}")

    return token_set, quals


# --- Main entry points ---


def tokenize_signature(
    signature_id: int,
    text: str,
    include_qualifiers: bool = True,
    include_bigrams: bool = True,
) -> TokenizedSignature:
    ruleset, category, description = parse_signature_text(text)
    cve_refs = extract_cve_refs(description)
    tokens, qualifiers = description_to_tokens(
        description,
        include_qualifiers=include_qualifiers,
        include_bigrams=include_bigrams,
    )
    # Category is a strong semantic signal; include it as a typed token
    if category:
        tokens.add(f"cat:{category}")
    # CVE refs as typed tokens alongside the structured cve_refs field
    tokens |= {f"cve:{c}" for c in cve_refs}

    return TokenizedSignature(
        signature_id=signature_id,
        ruleset=ruleset,
        category=category,
        description=description,
        cve_refs=cve_refs,
        qualifiers=qualifiers,
        tokens=tokens,
    )


def tokenize_signatures_df(
    df: pd.DataFrame,
    id_col: str = "SignatureID",
    text_col: str = "SignatureText",
    include_qualifiers: bool = True,
    include_bigrams: bool = True,
) -> list[TokenizedSignature]:
    """Tokenize every row of a DataFrame with SignatureID and SignatureText columns."""
    return [
        tokenize_signature(
            signature_id=int(row[id_col]),
            text=row[text_col],
            include_qualifiers=include_qualifiers,
            include_bigrams=include_bigrams,
        )
        for _, row in df.iterrows()
    ]
