from __future__ import annotations

import numpy as np

from thesis.system_eval.deepcase_comparison import Attended, CscasEvents, Vocab
from thesis.system_eval.deepcase_synthetic_drift import (
    Injection,
    detection_summary,
    inject,
    run_injection,
)

PARAMS = {"context_length": 2, "threshold": 0.2, "eps": 0.1}


def _events(n_train=1200, n_test=4000, seed=0):
    """Sigs 1,2 attack; 3,4,5 benign (5 = minority). Hosts h0..h19."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    sig = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.1, 0.4, 0.35, 0.05])
    return CscasEvents(
        ts=np.arange(n, dtype=np.int64) * 600,
        signature=sig.astype(np.int64),
        machine=np.array([f"h{i % 20}" for i in range(n)], dtype=object),
        label=np.isin(sig, [1, 2]).astype(np.int8),
        group_id=np.array([f"g{i}" for i in range(n)], dtype=object),
        split_idx=n_train,
    )


def _provider_factory(ev):
    def provide(n_train: int, vocab: Vocab, tag: str) -> Attended:
        ids = vocab.encode(ev.signature)
        ctx = np.stack([np.full(len(ids), vocab.no_event_id), ids], axis=1)
        att = np.tile(np.array([0.0, 1.0], dtype=np.float32), (len(ids), 1))
        return Attended(ctx, ids, np.ones(len(ids)), att)

    return provide


def test_inject_keeps_labels_and_prefix():
    ev = _events()
    start = ev.split_idx + 2000
    for kind in (
        "attack_mimics_benign",
        "benign_mimics_attack",
        "new_signature",
        "context_shift",
        "benign_mix_shift",
    ):
        out = inject(ev, Injection(kind, 0.5), start, seed=1)
        assert (out.label == ev.label).all()
        assert (out.signature[:start] == ev.signature[:start]).all()
        assert (out.machine[:start] == ev.machine[:start]).all()
        assert (ev.signature == _events().signature).all()  # input not mutated


def test_injection_effects():
    ev = _events()
    start = ev.split_idx
    att = ev.label[start:] == 1
    m = inject(ev, Injection("attack_mimics_benign", 1.0), start)
    assert np.isin(m.signature[start:][att], [3, 4, 5]).all()  # benign sigs
    b = inject(ev, Injection("benign_mimics_attack", 1.0), start)
    assert np.isin(b.signature[start:][~att], [1, 2]).all()
    mix = inject(ev, Injection("benign_mix_shift", 1.0), start)
    assert (mix.signature[start:][~att] == 5).all()  # the minority benign sig
    new = inject(ev, Injection("new_signature", 0.3), start)
    share = (new.signature[start:] > 5).mean()
    assert 0.25 < share < 0.35
    ctx = inject(ev, Injection("context_shift", 1.0), start, seed=3)
    assert (ctx.machine[start:] != ev.machine[start:]).mean() > 0.8


def test_end_to_end_detection_summary():
    ev = _events()
    frames = [
        run_injection(ev, _provider_factory, PARAMS, inj, 3, 0.25, 10, 5, 0, 0.5)
        for inj in (
            Injection("none", 0.0),
            Injection("attack_mimics_benign", 0.5),
            Injection("new_signature", 0.3),
        )
    ]
    import pandas as pd

    pw = pd.concat(frames, ignore_index=True)
    assert set(pw.injection) == {
        "none",
        "attack_mimics_benign@0.5",
        "new_signature@0.3",
    }
    s = detection_summary(pw).set_index("injection")
    # Attacks now carry benign signatures: auto-labelled benign -> recall drops,
    # benign clusters' precision and attack coverage drop.
    assert s.loc["attack_mimics_benign@0.5", "d_recall_vs_clean"] < -0.2
    assert s.loc["attack_mimics_benign@0.5", "pattern_validity_detect"] == 1.0
    # New signatures -> unknown events -> the native signal fires.
    assert s.loc["new_signature@0.3", "native_unknown_event_detect"] == 1.0
    # Nothing fires on the clean run.
    assert s.loc["none", "pattern_validity_detect"] == 0.0
    assert s.loc["none", "native_any_detect"] == 0.0
