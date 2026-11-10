"""MCTSExtent (discriminative sequential patterns, label used *during* the
search) as a comparison miner, running the authors' code (vendor.py).

Input: the per-host look-back sequences (AlertGroup.context_steps), the same
sequences PrefixSpan mines. MCTSExtent returns, per target class, the top_k
non-redundant patterns by WRAcc (theta = non-redundancy threshold). It runs
once per class to get attack- and benign-leaning patterns (plan §4).

Patterns are sequences of itemsets, so they're written as itemset_sequence
features (steps joined with STEP_ITEM_SEP); the encoder checks them in order
with the same subset-in-later-step rule upstream's is_subsequence uses, and
n_attack / n_benign are counted with that rule here.

Reproducibility: upstream stops on a wall-clock budget or an iteration cap.
`iterations` is the cap and `time_budget` defaults to effectively unlimited,
so a seeded run does the same work on any machine; pass a time_budget to
bound runtime instead (then results depend on machine speed).
"""

from __future__ import annotations

import contextlib
import io
import random
import warnings
from typing import Sequence

import pandas as pd

from thesis.mining.common.patterns import PATTERN_COLUMNS
from thesis.mining.common.util import itemset_sequence_contains_pattern
from thesis.mining.temporal.mctsextent.vendor import load_launch_mcts
from thesis.mining.temporal.prefixspan.miner import labelled_sequences
from thesis.schemas.features import join_step
from thesis.schemas.groups import AlertGroup

METHOD = "mctsextent"
UPSTREAM_DEFAULT_THETA = 0.8  # general/conf.py THETA


def mine_mctsextent(
    alert_groups: Sequence[AlertGroup],
    top_k: int = 25,
    theta: float = UPSTREAM_DEFAULT_THETA,
    iterations: int = 1000,
    time_budget: float = 10**9,
    random_state: int = 0,
    classes: tuple[str, ...] = ("attack", "benign"),
    quality_measure: str = "WRAcc",
) -> pd.DataFrame:
    launch_mcts = load_launch_mcts()
    seqs = labelled_sequences(alert_groups)
    # Integer-encode items (sorted, so the encoding is stable): upstream does
    # the same in get_patterns, and int hashing keeps set order reproducible.
    vocab = sorted({i for ss in seqs.values() for s in ss for step in s for i in step})
    enc = {item: n for n, item in enumerate(vocab)}
    dec = dict(enumerate(vocab))
    data = [
        [label, *({enc[i] for i in step} for step in s)]
        for label in ("attack", "benign")
        for s in seqs[label]
    ]
    all_seqs = [[set(step) for step in row[1:]] for row in data]
    is_attack = [row[0] == "attack" for row in data]

    rows = []
    for label in classes:
        if not seqs[label]:
            continue
        random.seed(random_state)
        # Upstream prints progress and uses the deprecated datetime.utcnow().
        with (
            contextlib.redirect_stdout(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            results = launch_mcts(
                [list(r) for r in data],  # upstream may mutate its input
                label,
                time_budget=time_budget,
                top_k=top_k,
                theta=theta,
                iterations_limit=iterations,
                quality_measure=quality_measure,
            )
        for quality, pattern in results:
            if not pattern:
                continue
            fires = [itemset_sequence_contains_pattern(s, pattern) for s in all_seqs]
            n_attack = sum(f and a for f, a in zip(fires, is_attack))
            n_benign = sum(f and not a for f, a in zip(fires, is_attack))
            rows.append(
                {
                    "items": tuple(join_step(dec[i] for i in step) for step in pattern),
                    "mining_type": "itemset_sequence",
                    "source_label": label,
                    "n_attack": n_attack,
                    "n_benign": n_benign,
                    "quality": float(quality),
                }
            )
    return pd.DataFrame(rows, columns=[*PATTERN_COLUMNS, "quality"])
