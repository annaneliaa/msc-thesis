"""Brute-force equivalence tests for the frequency-based miners (Eclat,
item PrefixSpan, itemset PrefixSpan): on small random databases, each miner
must return exactly the patterns a full enumeration finds, with the same
support counts, using the same matching rule the cross-label support code
(and the symbolic encoder) uses."""

import itertools
import random

import pandas as pd
import pytest

from thesis.mining.atemporal.eclat.eclat_mining import run_eclat
from thesis.mining.common.util import (
    add_confidence_scores,
    itemset_sequence_contains_pattern,
    min_support_count,
    sequence_contains_pattern,
)
from thesis.mining.temporal.prefixspan.prefixspan_mining import (
    run_itemset_prefixspan,
    run_prefixspan,
)

ITEMS = "abcd"


@pytest.mark.parametrize(
    "min_support,n,expected",
    [(0.05, 30, 2), (0.05, 100, 5), (0.2, 10, 2), (0.0, 10, 1), (0.01, 5, 1)],
)
def test_min_support_count_rounds_up(min_support, n, expected):
    assert min_support_count(min_support, n) == expected


def test_eclat_runs_without_run_dir():
    df = run_eclat([frozenset("ab")] * 3, min_support=0.5)
    assert set(df["itemset"]) == {("a",), ("b",), ("a", "b")}


def test_eclat_threshold_is_inclusive_and_rounds_up():
    # 1 of 30 groups is 3.3% < 5%: must not pass; 2 of 30 (6.7%) must.
    tx = [frozenset("a")] + [frozenset("bc")] * 2 + [frozenset("d")] * 27
    found = set(run_eclat(tx, min_support=0.05)["itemset"])
    assert ("a",) not in found
    assert ("b", "c") in found


@pytest.mark.parametrize("seed", range(5))
def test_eclat_matches_brute_force(seed):
    rng = random.Random(seed)
    for _ in range(40):
        txs = [
            frozenset(x for x in ITEMS if rng.random() < 0.5)
            for _ in range(rng.randint(1, 15))
        ]
        ms = rng.choice([0.1, 0.2, 0.3])
        df = run_eclat(txs, ms, max_len=None)
        got = dict(zip(df["itemset"], df["support_count"]))
        mc = min_support_count(ms, len(txs))
        expected = {}
        for k in range(1, len(ITEMS) + 1):
            for s in itertools.combinations(ITEMS, k):
                c = sum(set(s) <= t for t in txs)
                if c >= mc:
                    expected[s] = c
        assert got == expected


@pytest.mark.parametrize("seed", range(5))
def test_item_prefixspan_matches_brute_force(seed):
    rng = random.Random(seed)
    for _ in range(40):
        seqs = [
            [{x for x in ITEMS if rng.random() < 0.4} for _ in range(rng.randint(1, 5))]
            for _ in range(rng.randint(1, 10))
        ]
        df = run_prefixspan(seqs, 0.2, max_len=3)
        got = dict(zip(df["sequence"], df["support_count"]))
        mc = min_support_count(0.2, len(seqs))
        expected = {}
        for k in range(1, 4):
            for p in itertools.product(ITEMS, repeat=k):
                c = sum(sequence_contains_pattern(s, p) for s in seqs)
                if c >= mc:
                    expected[p] = c
        assert got == expected


def test_itemset_prefixspan_finds_i_extension_beyond_first_match():
    # <{a}, {a,c}>: the first alert holding {a} after alert 0 is {a} alone
    # (can't grow to {a,c}); the later {a,c} alert can. The old projection
    # kept only the first match and missed this pattern.
    seqs = [[{"a"}, {"a"}, {"a", "c"}]]
    df = run_itemset_prefixspan(seqs, 1.0, max_len=2)
    found = set(df["sequence"])
    assert (frozenset("a"), frozenset("ac")) in found
    assert (frozenset("ac"),) in found


@pytest.mark.parametrize("seed", range(5))
def test_itemset_prefixspan_matches_brute_force(seed):
    rng = random.Random(seed)
    items = "abc"
    steps = [
        frozenset(s)
        for k in range(1, len(items) + 1)
        for s in itertools.combinations(items, k)
    ]
    for _ in range(40):
        seqs = [
            [{x for x in items if rng.random() < 0.5} for _ in range(rng.randint(1, 4))]
            for _ in range(rng.randint(1, 8))
        ]
        df = run_itemset_prefixspan(seqs, 0.2, max_len=2)
        got = {tuple(p): c for p, c in zip(df["sequence"], df["support_count"])}
        mc = min_support_count(0.2, len(seqs))
        expected = {}
        for k in (1, 2):
            for p in itertools.product(steps, repeat=k):
                c = sum(itemset_sequence_contains_pattern(s, p) for s in seqs)
                if c >= mc:
                    expected[p] = c
        assert got == expected


def test_precision_is_class_rate_inside_pattern():
    # 30% of 150 attacks vs 1% of 10,000 benign: balanced confidence says
    # ~0.97 attack, but only 45 of 145 groups where it fires are attacks.
    df = pd.DataFrame(
        {
            "support_benign": [0.01],
            "support_attack": [0.30],
            "count_benign": [100],
            "count_attack": [45],
        }
    )
    out = add_confidence_scores(df)
    assert out["confidence_attack"].iloc[0] == pytest.approx(0.30 / 0.31)
    assert out["precision_attack"].iloc[0] == pytest.approx(45 / 145)
    assert out["precision_benign"].iloc[0] == pytest.approx(100 / 145)
