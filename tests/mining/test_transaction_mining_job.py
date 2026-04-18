import pytest
from pathlib import Path

from thesis.mining.mining_transaction_job import run_transaction_eclat_job
from thesis.schemas.mining import MiningTransaction


# -------------------------
# helpers
# -------------------------


def make_tx(i, items, label):
    return MiningTransaction(
        transaction_id=i,
        items=set(items),
        tx_label=label,
        window_start=0,
        window_end=1,
        n_alerts=len(items),
    )


# -------------------------
# tests
# -------------------------


def test_run_transaction_eclat_job_happy_path():
    txs = [
        make_tx(1, ["a", "b"], "benign"),
        make_tx(2, ["a"], "benign"),
        make_tx(3, ["b"], "attack"),
    ]

    run_dir = run_transaction_eclat_job(
        transactions=txs,
        scenario_name="test_scenario",
        min_support=0.1,
    )

    assert isinstance(run_dir, Path)


def test_run_transaction_eclat_job_empty_transactions():
    run_dir = run_transaction_eclat_job(
        transactions=[],
        scenario_name="empty_case",
    )

    assert isinstance(run_dir, Path)


def test_run_transaction_eclat_job_non_binary_labels_raises():
    txs = [
        make_tx(1, ["a"], "benign"),
        make_tx(2, ["b"], "attack"),
        make_tx(3, ["c"], "weird_label"),
    ]

    with pytest.raises(ValueError):
        run_transaction_eclat_job(
            transactions=txs,
            scenario_name="bad_labels",
        )


def test_run_transaction_eclat_job_only_target_label():
    txs = [
        make_tx(1, ["a", "b"], "benign"),
        make_tx(2, ["a"], "benign"),
    ]

    # should not crash even if no "other" label exists
    run_dir = run_transaction_eclat_job(
        transactions=txs,
        scenario_name="only_target",
    )

    assert isinstance(run_dir, Path)
