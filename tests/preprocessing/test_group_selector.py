from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.group_selector import (
    create_cache_query,
    select_group_snapshots,
    select_group_snapshots_from_response,
)
from thesis.schemas.cache import CacheQuery, CacheResponse, GroupCacheEntry
from thesis.schemas.preprocessing import GroupSnapshot

SCENARIO = "test_scenario"


def make_group_entry(
    *,
    group_id: str,
    method: str = "fixed_window",
    status: str = "closed",
    start_ts: int,
    end_ts: int,
    last_update_ts: int | None = None,
    alert_ids: list[str] | None = None,
    n_alerts: int = 0,
    items: set[str] | None = None,
    alert_labels: set[str] | None = None,
    group_label: str | None = None,
    version: int = 1,
) -> GroupCacheEntry:
    return GroupCacheEntry(
        group_id=group_id,
        method=method,
        status=status,
        start_ts=start_ts,
        end_ts=end_ts,
        last_update_ts=last_update_ts if last_update_ts is not None else end_ts,
        alert_ids=alert_ids or [],
        n_alerts=n_alerts,
        items=items or set(),
        alert_labels=alert_labels,
        group_label=group_label,
        version=version,
    )


def test_create_cache_query_builds_expected_defaults():
    query = create_cache_query()

    assert isinstance(query, CacheQuery)
    assert query.allowed_methods == {"fixed_window", "alertbert"}
    assert query.only_closed is True
    assert query.allowed_statuses == {"closed"}
    assert query.min_start_ts is None
    assert query.max_end_ts is None
    assert query.limit is None


def test_create_cache_query_overrides_defaults():
    query = create_cache_query(
        allowed_methods={"fixed_window"},
        only_closed=False,
        allowed_statuses={"open", "closed"},
        min_start_ts=100,
        max_end_ts=200,
        limit=5,
    )

    assert query.allowed_methods == {"fixed_window"}
    assert query.only_closed is False
    assert query.allowed_statuses == {"open", "closed"}
    assert query.min_start_ts == 100
    assert query.max_end_ts == 200
    assert query.limit == 5


def test_cache_query_returns_matching_groups_only(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    g1 = make_group_entry(
        group_id="g1",
        method="fixed_window",
        status="closed",
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        n_alerts=1,
        items={"short:A"},
    )
    g2 = make_group_entry(
        group_id="g2",
        method="alertbert",
        status="closed",
        start_ts=202,
        end_ts=203,
        alert_ids=["a2"],
        n_alerts=1,
        items={"short:B"},
    )
    g3 = make_group_entry(
        group_id="g3",
        method="other_method",
        status="closed",
        start_ts=204,
        end_ts=205,
        alert_ids=["a3"],
        n_alerts=1,
        items={"short:C"},
    )

    cache.write_group_entry(g1)
    cache.write_group_entry(g2)
    cache.write_group_entry(g3)

    response = cache.query(
        CacheQuery(
            allowed_methods={"fixed_window", "alertbert"},
            only_closed=True,
            allowed_statuses={"closed"},
        )
    )

    assert isinstance(response, CacheResponse)
    assert [g.group_id for g in response.groups] == ["g1", "g2"]


def test_cache_query_returns_only_closed_groups_when_requested(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    open_group = make_group_entry(
        group_id="g1",
        status="open",
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        n_alerts=1,
        items={"short:A"},
    )
    closed_group = make_group_entry(
        group_id="g2",
        status="closed",
        start_ts=202,
        end_ts=203,
        alert_ids=["a2"],
        n_alerts=1,
        items={"short:B"},
    )

    cache.write_group_entry(open_group)
    cache.write_group_entry(closed_group)

    response = cache.query(
        CacheQuery(
            allowed_methods={"fixed_window", "alertbert"},
            only_closed=True,
            allowed_statuses={"closed"},
        )
    )

    assert [g.group_id for g in response.groups] == ["g2"]


def test_cache_query_returns_empty_response_when_no_groups_match(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    g1 = make_group_entry(
        group_id="g1",
        method="other_method",
        status="open",
        start_ts=200,
        end_ts=201,
        alert_ids=["a1"],
        n_alerts=1,
        items={"short:A"},
    )
    cache.write_group_entry(g1)

    response = cache.query(
        CacheQuery(
            allowed_methods={"fixed_window"},
            only_closed=True,
            allowed_statuses={"closed"},
        )
    )

    assert isinstance(response, CacheResponse)
    assert response.groups == []


def test_select_group_snapshots_from_response_without_labels():
    response = CacheResponse(
        groups=[
            make_group_entry(
                group_id="g1",
                method="fixed_window",
                status="closed",
                start_ts=200,
                end_ts=201,
                alert_ids=["a1", "a2"],
                n_alerts=2,
                items={"short:A", "host:h1", "sig:update"},
                alert_labels=None,
                group_label=None,
                version=1,
            )
        ]
    )

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=None,
        require_closed=True,
    )

    assert len(snapshots) == 1

    snap = snapshots[0]
    assert isinstance(snap, GroupSnapshot)
    assert snap.group_id == "g1"
    assert snap.method == "fixed_window"
    assert snap.version == 1
    assert snap.start_ts == 200
    assert snap.end_ts == 201
    assert snap.alert_ids == ["a1", "a2"]
    assert snap.n_alerts == 2
    assert snap.items == {"short:A", "host:h1", "sig:update"}
    assert snap.alert_labels is None
    assert snap.group_label is None
    assert snap.status == "closed"


def test_select_group_snapshots_from_response_with_labels():
    response = CacheResponse(
        groups=[
            make_group_entry(
                group_id="g1",
                method="alertbert",
                status="closed",
                start_ts=200,
                end_ts=201,
                alert_ids=["a1", "a2", "a3"],
                n_alerts=3,
                items={"short:A", "host:h1", "sig:update"},
                alert_labels={"benign", "false_positive"},
                group_label="benign",
                version=2,
            )
        ]
    )

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=None,
        require_closed=True,
    )

    assert len(snapshots) == 1

    snap = snapshots[0]
    assert snap.group_id == "g1"
    assert snap.method == "alertbert"
    assert snap.version == 2
    assert snap.start_ts == 200
    assert snap.end_ts == 201
    assert snap.alert_ids == ["a1", "a2", "a3"]
    assert snap.n_alerts == 3
    assert snap.items == {"short:A", "host:h1", "sig:update"}
    assert snap.alert_labels == {"benign", "false_positive"}
    assert snap.group_label == "benign"
    assert snap.status == "closed"


def test_select_group_snapshots_from_response_applies_limit_to_latest_groups():
    response = CacheResponse(
        groups=[
            make_group_entry(
                group_id="g1",
                start_ts=200,
                end_ts=201,
                alert_ids=["a1"],
                n_alerts=1,
                items={"short:A"},
            ),
            make_group_entry(
                group_id="g2",
                start_ts=202,
                end_ts=203,
                alert_ids=["a2"],
                n_alerts=1,
                items={"short:B"},
            ),
            make_group_entry(
                group_id="g3",
                start_ts=204,
                end_ts=205,
                alert_ids=["a3"],
                n_alerts=1,
                items={"short:C"},
            ),
        ]
    )

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=2,
        require_closed=True,
    )

    assert [snap.group_id for snap in snapshots] == ["g2", "g3"]


def test_select_group_snapshots_from_response_filters_open_groups_when_required():
    response = CacheResponse(
        groups=[
            make_group_entry(
                group_id="g1",
                status="open",
                start_ts=200,
                end_ts=201,
                alert_ids=["a1"],
                n_alerts=1,
                items={"short:A"},
            ),
            make_group_entry(
                group_id="g2",
                status="closed",
                start_ts=202,
                end_ts=203,
                alert_ids=["a2"],
                n_alerts=1,
                items={"short:B"},
            ),
        ]
    )

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=None,
        require_closed=True,
    )

    assert [snap.group_id for snap in snapshots] == ["g2"]


def test_select_group_snapshots_from_response_keeps_open_groups_when_not_required():
    response = CacheResponse(
        groups=[
            make_group_entry(
                group_id="g1",
                status="open",
                start_ts=200,
                end_ts=201,
                alert_ids=["a1"],
                n_alerts=1,
                items={"short:A"},
            ),
            make_group_entry(
                group_id="g2",
                status="closed",
                start_ts=202,
                end_ts=203,
                alert_ids=["a2"],
                n_alerts=1,
                items={"short:B"},
            ),
        ]
    )

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=None,
        require_closed=False,
    )

    assert [snap.group_id for snap in snapshots] == ["g1", "g2"]


def test_select_group_snapshots_from_response_returns_empty_list_for_empty_response():
    response = CacheResponse(groups=[])

    snapshots = select_group_snapshots_from_response(
        response=response,
        limit=None,
        require_closed=True,
    )

    assert snapshots == []


def test_select_group_snapshots_queries_cache_and_returns_snapshots(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    g1 = make_group_entry(
        group_id="g1",
        method="fixed_window",
        status="closed",
        start_ts=200,
        end_ts=201,
        alert_ids=["a1", "a2"],
        n_alerts=2,
        items={"short:A", "sig:update"},
        alert_labels={"false_positive"},
        group_label="benign",
        version=1,
    )
    g2 = make_group_entry(
        group_id="g2",
        method="alertbert",
        status="closed",
        start_ts=202,
        end_ts=203,
        alert_ids=["a3"],
        n_alerts=1,
        items={"short:B", "sig:invalid"},
        alert_labels=None,
        group_label=None,
        version=1,
    )

    cache.write_group_entry(g1)
    cache.write_group_entry(g2)

    snapshots = select_group_snapshots(
        cache=cache,
        allowed_methods={"fixed_window", "alertbert"},
        limit=None,
        min_start_ts=None,
        max_end_ts=None,
        require_closed=True,
    )

    assert [snap.group_id for snap in snapshots] == ["g1", "g2"]

    s1 = snapshots[0]
    assert s1.method == "fixed_window"
    assert s1.version == 1
    assert s1.start_ts == 200
    assert s1.end_ts == 201
    assert s1.alert_ids == ["a1", "a2"]
    assert s1.n_alerts == 2
    assert s1.items == {"short:A", "sig:update"}
    assert s1.alert_labels == {"false_positive"}
    assert s1.group_label == "benign"
    assert s1.status == "closed"

    s2 = snapshots[1]
    assert s2.method == "alertbert"
    assert s2.version == 1
    assert s2.start_ts == 202
    assert s2.end_ts == 203
    assert s2.alert_ids == ["a3"]
    assert s2.n_alerts == 1
    assert s2.items == {"short:B", "sig:invalid"}
    assert s2.alert_labels is None
    assert s2.group_label is None
    assert s2.status == "closed"
