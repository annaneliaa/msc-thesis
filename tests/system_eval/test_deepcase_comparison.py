import numpy as np
import pandas as pd
from deepcase.preprocessing import Preprocessor

from thesis.system_eval.deepcase_comparison import (
    CODE_LOW_CONFIDENCE,
    CODE_NOT_IN_TRAINING,
    CODE_TOO_FAR,
    Attended,
    Vocab,
    build_context,
    cluster_rows,
    decision_metrics,
    predict_codes_and_clusters,
    fit_interpreter,
    horizon_bounds,
    predict_codes,
    simulated_analyst,
    vectorize,
)


def test_build_context_matches_deepcase_preprocessor():
    rng = np.random.default_rng(0)
    n = 400
    ts = np.sort(rng.integers(0, 5 * 86400, n))  # distinct enough; ties allowed
    ts = ts + np.arange(n)  # strictly increasing so tie order can't differ
    machine = rng.choice(["a", "b", "c"], n).astype(object)
    sig = rng.choice([11, 22, 33, 44], n)
    L, timeout = 5, 3600.0

    df = pd.DataFrame({"timestamp": ts, "machine": machine, "event": sig})
    ctx_ref, ev_ref, _, mapping = Preprocessor(length=L, timeout=timeout).sequence(df)

    vocab = Vocab.from_signatures(sig)
    ids = vocab.encode(sig)
    ctx = build_context(ids, machine, ts, L, timeout, vocab.no_event_id)

    # Translate the reference ids (its own mapping; NO_EVENT last) to ours.
    inv = {
        k: (
            vocab.no_event_id
            if v == -1337
            else int(np.searchsorted(vocab.signatures, v))
        )
        for k, v in mapping.items()
    }
    ref = np.vectorize(inv.get)(ctx_ref.numpy())
    np.testing.assert_array_equal(ctx, ref)
    np.testing.assert_array_equal(ids, np.vectorize(inv.get)(ev_ref.numpy()))


def test_build_context_no_host_gets_no_context():
    ts = np.arange(4)
    ctx = build_context(
        np.array([0, 1, 0, 1]),
        np.array([None, None, "a", "a"], dtype=object),
        ts,
        2,
        10,
        9,
    )
    assert (ctx[:2] == 9).all()
    assert ctx[3].tolist() == [9, 0]


def test_vocab_unknown():
    v = Vocab.from_signatures(np.array([5, 3, 5]))
    assert v.encode(np.array([3, 5, 7])).tolist() == [0, 1, v.unknown_id]
    assert v.size == 4


def test_simulated_analyst_conservative_rule_and_budget():
    clusters = np.array([0] * 20 + [1] * 3 + [-1, -1])
    label = np.zeros(25, dtype=np.int8)
    label[0:20] = 1  # cluster 0 all attack
    mask = np.array([True] * 24 + [False])
    res = simulated_analyst(clusters, mask, label, 10, np.random.default_rng(0))
    assert res.cluster_label == {0: 1, 1: 0}
    assert res.stats["n_cluster_samples"] == 13  # 10 + min(10, 3)
    assert res.stats["n_outliers"] == 2
    assert res.stats["n_outliers_low_confidence"] == 1
    assert res.stats["budget_b0"] == 15


def test_interpreter_roundtrip_codes():
    # Two event types; event 0 has a tight cluster, event 1 only noise.
    size = 4
    X = np.array([[2, 3]] * 6 + [[3, 2]] * 2)
    A = np.array([[0.5, 0.5]] * 6 + [[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    y = np.array([0] * 6 + [1, 1])
    conf = np.ones(8)
    tr = Attended(X, y, conf, A)
    vec, mask = vectorize(tr, 0.2, size)
    clusters = cluster_rows(vec, mask, y, eps=0.1, min_samples=5)
    assert (clusters[:6] == clusters[0]).all() and clusters[0] != -1
    assert (clusters[6:] == -1).all()
    label = np.array([1] + [0] * 7, dtype=np.int8)
    analyst = simulated_analyst(clusters, mask, label, 10, np.random.default_rng(0))
    fitted = fit_interpreter(vec, mask, y, clusters, analyst.cluster_label, 0.1)

    Xt = np.array([[2, 3], [2, 3], [2, 3], [3, 2], [2, 3]])
    At = np.array(
        [[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32
    )
    yt = np.array([0, 0, 0, 1, 2])  # 2 = unknown id here
    ct = np.array([1.0, 1.0, 0.05, 1.0, 1.0])
    vt, mt = vectorize(Attended(Xt, yt, ct, At), 0.2, size)
    codes = predict_codes(fitted, vt, mt, yt, unknown_id=2)
    assert codes.tolist() == [
        1,
        CODE_TOO_FAR,
        CODE_LOW_CONFIDENCE,
        CODE_NOT_IN_TRAINING,
        CODE_NOT_IN_TRAINING,
    ]
    # The matched training cluster is kept for the automatic decision only.
    codes2, cl = predict_codes_and_clusters(fitted, vt, mt, yt, unknown_id=2)
    assert codes2.tolist() == codes.tolist()
    assert cl.tolist() == [clusters[0], -1, -1, -1, -1]


def test_decision_metrics_manual_counts_as_escalated():
    label = np.array([1, 1, 1, 0, 0, 0])
    codes = np.array([1, 0, -1, 1, 0, -3])
    m = decision_metrics(label, codes, days=2.0)
    assert m["coverage"] == 4 / 6
    assert m["missed_attacks"] == 1
    assert m["analyst_items"] == 4  # 2 flagged + 2 manual
    assert m["covered_recall"] == 0.5
    assert m["e2e_recall"] == 2 / 3
    assert m["analyst_items_per_day"] == 2.0


def test_horizon_bounds_cover_all_rows():
    b = horizon_bounds(1_255_792, 0.1)
    assert len(b) == 10
    assert b[0][1] == 0 and b[-1][2] == 1_255_792
    assert all(b[i][2] == b[i + 1][1] for i in range(len(b) - 1))
