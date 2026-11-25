from __future__ import annotations

import numpy as np
import pytest

from thesis.system_eval.deepcase_comparison import Attended, CscasEvents, Vocab
from thesis.system_eval.deepcase_updates import UpdateConfig, run_seed

PARAMS = {"context_length": 2, "threshold": 0.2, "eps": 0.1}
NEW_SIG = 9  # attack signature first seen in test window 2


def _events():
    """600 W0 rows (sigs 1,2 attack; 3,4 benign) + 4 test windows of 250.
    From window 2 on: 50 rows of a new attack signature, and 30 sig-3 rows
    whose context differs from training ("too far" for a frozen model)."""
    sig, lab, far = [], [], []
    for s_, l_ in ((1, 1), (2, 1), (3, 0), (4, 0)):
        sig += [s_] * 150
        lab += [l_] * 150
        far += [False] * 150
    for w in range(4):
        part = [1] * 40 + [2] * 40 + [3] * 60 + [4] * 60
        part_far = [False] * 200
        if w >= 1:
            part += [NEW_SIG] * 50
            part_far += [False] * 50
            part[80:110] = [3] * 30
            part_far[80:110] = [True] * 30
        else:
            part += [4] * 50
            part_far += [False] * 50
        sig += part
        lab += [1 if s_ in (1, 2, NEW_SIG) else 0 for s_ in part]
        far += part_far
    n = len(sig)
    rng = np.random.default_rng(0)
    order = np.argsort(rng.random(600))  # shuffle W0 rows a bit
    sig_a, lab_a = np.array(sig), np.array(lab, dtype=np.int8)
    sig_a[:600], lab_a[:600] = sig_a[:600][order], lab_a[:600][order]
    ev = CscasEvents(
        ts=np.arange(n, dtype=np.int64) * 600,
        signature=sig_a.astype(np.int64),
        machine=np.array([None] * n, dtype=object),
        label=lab_a,
        group_id=np.array([f"g{i}" for i in range(n)], dtype=object),
        split_idx=600,
    )
    return ev, np.array(far)


def _provider(ev, far):
    calls = []

    def provide(n_train: int, vocab: Vocab, tag: str) -> Attended:
        calls.append((n_train, tag))
        ids = vocab.encode(ev.signature)
        ctx = np.stack([np.full(len(ids), vocab.no_event_id), ids], axis=1)
        ctx[far, 1] = vocab.no_event_id  # different context -> different vector
        att = np.tile(np.array([0.0, 1.0], dtype=np.float32), (len(ids), 1))
        return Attended(ctx, ids, np.ones(len(ids)), att)

    return provide, calls


def _run(policy, action="REMINE_AND_RETRAIN", audit="auto"):
    ev, far = _events()
    provide, calls = _provider(ev, far)
    cfg = UpdateConfig(
        base_run="unused",
        policy=policy,
        schedule_action=action,
        audit=audit,
        audit_rate=0.5,
    )
    rows, events, per_row = run_seed(ev, provide, PARAMS, cfg, 10, 5, 0.25, seed=0)
    return rows, events, per_row, calls


def test_never_is_frozen_and_sends_new_signature_to_analyst():
    rows, events, per_row, calls = _run("never")
    assert events == [] and calls == [(600, "w0")]
    late = per_row[per_row.horizon >= rows[1]["horizon"]]
    new = late[late.group_id.isin([f"g{i}" for i in range(600, 1600)])]
    assert (new.code == -2).any()  # unknown signature -> manual queue
    assert len({r["budget_labels"] for r in rows}) == 1  # B0 only, no audit
    assert all(r["n_audit_labels"] == 0 for r in rows)


def test_retrain_only_adds_clusters_for_too_far_events():
    rows, events, per_row, calls = _run("schedule", action="RETRAIN_ONLY")
    assert calls == [(600, "w0")]  # ContextBuilder never retrained
    assert any(e["n_new_clusters"] > 0 for e in events)
    # After the first update the too-far sig-3 rows are decided automatically.
    last = per_row[per_row.horizon == per_row.horizon.max()]
    assert (last.code == -3).sum() == 0
    # New signature still unknown to the frozen vocabulary.
    assert (last.code == -2).sum() > 0


def test_remine_retrains_and_covers_new_signature():
    rows, events, per_row, calls = _run("schedule")
    assert len(calls) == 1 + len(events)
    assert all(tag.startswith("upd_h") for _, tag in calls[1:])
    last = per_row[per_row.horizon == per_row.horizon.max()]
    assert (last.code < 0).sum() == 0  # everything covered after updating
    assert rows[-1]["all_e2e_recall"] == pytest.approx(1.0)
    assert all(e["extra_labels"] >= 0 for e in events)
    assert rows[-1]["budget_labels"] == (
        rows[-1]["budget_b0"] + rows[-1]["budget_audit"] + rows[-1]["budget_update"]
    )


def test_audit_runs_for_combined_or_when_forced():
    rows, *_ = _run("combined")
    assert sum(r["n_audit_labels"] for r in rows) > 0
    rows, *_ = _run("schedule", audit="on")
    assert sum(r["n_audit_labels"] for r in rows) > 0
    rows, *_ = _run("schedule")
    assert sum(r["n_audit_labels"] for r in rows) == 0


def test_combined_first_window_is_reference_and_logs_signals():
    rows, *_ = _run("combined")
    assert rows[0]["signals_reference_window"] and rows[0]["action"] == "NO_ACTION"
    assert {"mon_elevated", "mon_cov_attack", "mon_fb_precision"} <= set(rows[1])


def test_no_remine_downgrades_to_add_clusters():
    ev, far = _events()
    provide, calls = _provider(ev, far)
    cfg = UpdateConfig(base_run="unused", policy="schedule", allow_remine=False)
    rows, events, _ = run_seed(ev, provide, PARAMS, cfg, 10, 5, 0.25, seed=0)
    assert calls == [(600, "w0")]  # ContextBuilder never retrained
    assert {e["action"] for e in events} == {"RETRAIN_ONLY"}
    assert {r["requested_action"] for r in rows} == {"REMINE_AND_RETRAIN"}


def test_native_policy_acts_on_new_event_types():
    ev, far = _events()
    provide, _ = _provider(ev, far)
    cfg = UpdateConfig(base_run="unused", policy="native", allow_remine=False)
    rows, events, _ = run_seed(ev, provide, PARAMS, cfg, 10, 5, 0.25, seed=0)
    assert rows[0]["action"] == "NO_ACTION"  # reference window
    # The new signature appears in window 2: unknown-event share rises.
    assert rows[1]["native_unknown_event_elevated"]
    assert rows[1]["requested_action"] == "REMINE_AND_RETRAIN"
    assert rows[1]["action"] == "RETRAIN_ONLY"
    assert len(events) >= 1
