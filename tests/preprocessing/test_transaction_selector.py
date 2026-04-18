import math

from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.transaction_selector import (
    select_transactions,
    select_transactions_from_response,
)
from thesis.schemas.cache import CacheQuery, CacheResponse, WindowCacheEntry
from thesis.schemas.preprocessing import WindowTransaction

SCENARIO = "test_scenario"


def make_window_entry(
    *,
    window_id: int,
    start_ts: int,
    end_ts: int,
    alert_ids: list[str] | None = None,
    items: set[str] | None = None,
    hosts: set[str] | None = None,
    signatures: set[str] | None = None,
    closed: bool = False,
    alert_labels: set[str] | None = None,
    tx_label: str | None = None,
) -> WindowCacheEntry:
    return WindowCacheEntry(
        window_id=window_id,
        start_ts=start_ts,
        end_ts=end_ts,
        alert_ids=alert_ids or [],
        items=items or set(),
        hosts=hosts or set(),
        signatures=signatures or set(),
        closed=closed,
        alert_labels=alert_labels,
        tx_label=tx_label,
    )


def test_cache_query_returns_matching_windows_only(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    w1 = make_window_entry(
        window_id=100,
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        items={"short:A", "host:h1"},
        closed=False,
    )
    w2 = make_window_entry(
        window_id=101,
        start_ts=202,
        end_ts=203,
        alert_ids=["a2"],
        items={"short:B", "host:h2"},
        closed=False,
    )
    w3 = make_window_entry(
        window_id=102,
        start_ts=204,
        end_ts=205,
        alert_ids=["a3"],
        items={"short:C", "host:h3"},
        closed=False,
    )

    cache.write_window_entry(w1)
    cache.write_window_entry(w2)
    cache.write_window_entry(w3)

    response = cache.query(
        CacheQuery(
            min_window_id=101,
            max_window_id=102,
            only_closed=False,
        )
    )

    assert isinstance(response, CacheResponse)
    assert [w.window_id for w in response.windows] == [101, 102]


def test_cache_query_returns_only_closed_windows_when_requested(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    open_window = make_window_entry(
        window_id=100,
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        items={"short:A"},
        closed=False,
    )
    closed_window = make_window_entry(
        window_id=101,
        start_ts=202,
        end_ts=203,
        alert_ids=["a2"],
        items={"short:B"},
        closed=True,
    )

    cache.write_window_entry(open_window)
    cache.write_window_entry(closed_window)

    response = cache.query(CacheQuery(only_closed=True))

    assert [w.window_id for w in response.windows] == [101]


def test_cache_query_returns_empty_response_when_no_windows_match(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    w1 = make_window_entry(
        window_id=100,
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        items={"short:A"},
        closed=False,
    )
    cache.write_window_entry(w1)

    response = cache.query(
        CacheQuery(
            min_window_id=200,
            max_window_id=300,
            only_closed=False,
        )
    )

    assert isinstance(response, CacheResponse)
    assert response.windows == []


def test_select_transactions_from_response_without_labels():
    response = CacheResponse(
        windows=[
            make_window_entry(
                window_id=100,
                start_ts=200,
                end_ts=201,
                alert_ids=["a1", "a2"],
                items={"short:A", "host:h1", "sig:update"},
                closed=False,
                alert_labels=None,
                tx_label=None,
            )
        ]
    )

    transactions = select_transactions_from_response(
        response=response,
        retention_windows=None,
        decay_factor=1.0,
    )

    assert len(transactions) == 1

    tx = transactions[0]
    assert isinstance(tx, WindowTransaction)
    assert tx.window_id == 100
    assert tx.window_start == 200
    assert tx.window_end == 201
    assert tx.n_alerts == 2
    assert tx.items == {"short:A", "host:h1", "sig:update"}
    assert tx.alert_labels is None
    assert tx.tx_label is None
    assert tx.weight == 1.0


def test_select_transactions_from_response_with_labels():
    response = CacheResponse(
        windows=[
            make_window_entry(
                window_id=100,
                start_ts=200,
                end_ts=201,
                alert_ids=["a1", "a2", "a3"],
                items={"short:A", "host:h1", "sig:update"},
                closed=False,
                alert_labels={"benign", "false_positive"},
                tx_label="benign",
            )
        ]
    )

    transactions = select_transactions_from_response(
        response=response,
        retention_windows=None,
        decay_factor=1.0,
    )

    assert len(transactions) == 1

    tx = transactions[0]
    assert tx.window_id == 100
    assert tx.window_start == 200
    assert tx.window_end == 201
    assert tx.n_alerts == 3
    assert tx.items == {"short:A", "host:h1", "sig:update"}
    assert tx.alert_labels == {"benign", "false_positive"}
    assert tx.tx_label == "benign"
    assert tx.weight == 1.0


def test_select_transactions_applies_retention():
    response = CacheResponse(
        windows=[
            make_window_entry(
                window_id=100,
                start_ts=200,
                end_ts=201,
                alert_ids=["a1"],
                items={"short:A"},
                closed=False,
            ),
            make_window_entry(
                window_id=101,
                start_ts=202,
                end_ts=203,
                alert_ids=["a2"],
                items={"short:B"},
                closed=False,
            ),
            make_window_entry(
                window_id=102,
                start_ts=204,
                end_ts=205,
                alert_ids=["a3"],
                items={"short:C"},
                closed=False,
            ),
        ]
    )

    transactions = select_transactions_from_response(
        response=response,
        retention_windows=2,
        decay_factor=1.0,
    )

    assert [tx.window_id for tx in transactions] == [101, 102]


def test_select_transactions_applies_decay():
    response = CacheResponse(
        windows=[
            make_window_entry(
                window_id=100,
                start_ts=200,
                end_ts=201,
                alert_ids=["a1"],
                items={"short:A"},
                closed=False,
            ),
            make_window_entry(
                window_id=101,
                start_ts=202,
                end_ts=203,
                alert_ids=["a2"],
                items={"short:B"},
                closed=False,
            ),
            make_window_entry(
                window_id=102,
                start_ts=204,
                end_ts=205,
                alert_ids=["a3"],
                items={"short:C"},
                closed=False,
            ),
        ]
    )

    transactions = select_transactions_from_response(
        response=response,
        retention_windows=None,
        decay_factor=0.5,
    )

    assert [tx.window_id for tx in transactions] == [100, 101, 102]

    # latest window_id = 102
    # weights: 0.5^(2), 0.5^(1), 0.5^(0)
    assert math.isclose(transactions[0].weight, 0.25)
    assert math.isclose(transactions[1].weight, 0.5)
    assert math.isclose(transactions[2].weight, 1.0)


def test_select_transactions_returns_empty_list_for_empty_response():
    response = CacheResponse(windows=[])

    transactions = select_transactions_from_response(
        response=response,
        retention_windows=None,
        decay_factor=1.0,
    )

    assert transactions == []


def test_select_transactions_queries_cache_and_returns_transactions(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    w1 = make_window_entry(
        window_id=100,
        start_ts=200,
        end_ts=201,
        alert_ids=["a1", "a2"],
        items={"short:A", "sig:update"},
        closed=False,
        alert_labels={"false_positive"},
        tx_label="benign",
    )
    w2 = make_window_entry(
        window_id=101,
        start_ts=202,
        end_ts=203,
        alert_ids=["a3"],
        items={"short:B", "sig:invalid"},
        closed=False,
        alert_labels=None,
        tx_label=None,
    )

    cache.write_window_entry(w1)
    cache.write_window_entry(w2)

    transactions = select_transactions(
        cache=cache,
        query=CacheQuery(min_window_id=100, max_window_id=101, only_closed=False),
        retention_windows=None,
        decay_factor=1.0,
    )

    assert [tx.window_id for tx in transactions] == [100, 101]

    tx1 = transactions[0]
    assert tx1.window_start == 200
    assert tx1.window_end == 201
    assert tx1.n_alerts == 2
    assert tx1.items == {"short:A", "sig:update"}
    assert tx1.alert_labels == {"false_positive"}
    assert tx1.tx_label == "benign"
    assert tx1.weight == 1.0

    tx2 = transactions[1]
    assert tx2.window_start == 202
    assert tx2.window_end == 203
    assert tx2.n_alerts == 1
    assert tx2.items == {"short:B", "sig:invalid"}
    assert tx2.alert_labels is None
    assert tx2.tx_label is None
    assert tx2.weight == 1.0
