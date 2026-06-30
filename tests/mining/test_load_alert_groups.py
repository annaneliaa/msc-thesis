import json
from pathlib import Path

import pandas as pd
import pytest

from thesis.mining.load_mining_alert_groups import (
    build_tidsets,
    load_and_prepare_mining_alert_groups,
    load_mining_alert_groups_from_cache,
    prepare_alert_groups,
)
from thesis.schemas.mining import MiningAlertGroup


def make_tx(
    alert_group_id=1,
    items=None,
    group_label="benign",
    window_start=0,
    window_end=1,
    n_alerts=1,
    alert_labels=None,
    weight=1.0,
):
    return MiningAlertGroup(
        alert_group_id=alert_group_id,
        window_start=window_start,
        window_end=window_end,
        n_alerts=n_alerts,
        items=set(items or []),
        group_label=group_label,
        alert_labels=alert_labels,
        weight=weight,
    )


def write_cache_file(tmp_path: Path, payload: list[dict]) -> Path:
    cache_dir = tmp_path / "artifacts" / "cache" / "alert_groups"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "alert_groups.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_load_mining_alert_groups_from_cache_happy_path(tmp_path):
    path = write_cache_file(
        tmp_path,
        [
            {
                "alert_group_id": 821106976,
                "window_start": 1642213952,
                "window_end": 1642213953,
                "n_alerts": 2,
                "abs_items": ["host:mail", "sig:tls"],
                "group_label": "benign",
                "alert_labels": ["false_positive"],
                "weight": 1.0,
            }
        ],
    )

    out = load_mining_alert_groups_from_cache(path)

    assert len(out) == 1
    tx = out[0]
    assert isinstance(tx, MiningAlertGroup)
    assert tx.alert_group_id == 821106976
    assert tx.window_start == 1642213952
    assert tx.window_end == 1642213953
    assert tx.n_alerts == 2
    assert tx.items == {"host:mail", "sig:tls"}
    assert tx.group_label == "benign"
    assert tx.alert_labels == {"false_positive"}
    assert tx.weight == 1.0


def test_load_mining_alert_groups_from_cache_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        load_mining_alert_groups_from_cache(missing)


def test_load_mining_alert_groups_from_cache_empty_list(tmp_path):
    path = write_cache_file(tmp_path, [])

    out = load_mining_alert_groups_from_cache(path)

    assert out == []


def test_load_mining_alert_groups_from_cache_none_alert_labels(tmp_path):
    path = write_cache_file(
        tmp_path,
        [
            {
                "alert_group_id": 1,
                "window_start": 10,
                "window_end": 11,
                "n_alerts": 1,
                "abs_items": ["a", "b"],
                "group_label": "benign",
                "alert_labels": None,
                "weight": 1.0,
            }
        ],
    )

    out = load_mining_alert_groups_from_cache(path)

    assert len(out) == 1
    assert out[0].alert_labels is None


def test_load_mining_alert_groups_from_cache_defaults_weight_when_missing(tmp_path):
    path = write_cache_file(
        tmp_path,
        [
            {
                "alert_group_id": 1,
                "window_start": 10,
                "window_end": 11,
                "n_alerts": 1,
                "abs_items": ["a"],
                "group_label": "benign",
                "alert_labels": None,
            }
        ],
    )

    out = load_mining_alert_groups_from_cache(path)

    assert len(out) == 1
    assert out[0].weight == 1.0


def test_prepare_alert_groups_keeps_only_non_empty_and_labeled(tmp_path):
    txs = [
        make_tx(alert_group_id=1, items={"a", "b"}, group_label="benign"),
        make_tx(alert_group_id=2, items=set(), group_label="benign"),
        make_tx(alert_group_id=3, items={"x"}, group_label=None),
        make_tx(alert_group_id=4, items={"y"}, group_label="attack"),
    ]

    prepared = prepare_alert_groups(txs)

    assert len(prepared) == 2
    assert [tx.alert_group_id for tx in prepared] == [1, 4]


def test_prepare_alert_groups_strips_whitespace_and_drops_blank_items():
    txs = [
        make_tx(
            alert_group_id=1,
            items={" a ", "b", "   ", ""},
            group_label="benign",
        )
    ]

    prepared = prepare_alert_groups(txs)

    assert len(prepared) == 1
    assert prepared[0].items == {"a", "b"}


def test_prepare_alert_groups_drops_alert_group_if_items_become_empty_after_cleaning():
    txs = [
        make_tx(
            alert_group_id=1,
            items={"   ", ""},
            group_label="benign",
        )
    ]

    prepared = prepare_alert_groups(txs)

    assert prepared == []


def test_prepare_alert_groups_copies_alert_labels_as_set():
    txs = [
        make_tx(
            alert_group_id=1,
            items={"a"},
            group_label="benign",
            alert_labels={"fp", "benign_context"},
        )
    ]

    prepared = prepare_alert_groups(txs)

    assert len(prepared) == 1
    assert prepared[0].alert_labels == {"fp", "benign_context"}
    assert isinstance(prepared[0].alert_labels, set)


def test_prepare_alert_groups_writes_prepared_csv(tmp_path):
    txs = [
        make_tx(
            alert_group_id=1,
            items={"b", "a"},
            group_label="benign",
            alert_labels={"fp"},
            weight=0.8,
        )
    ]

    prepared = prepare_alert_groups(txs, run_dir=tmp_path)

    assert len(prepared) == 1

    csv_path = tmp_path / "prepared_alert_groups.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.loc[0, "alert_group_id"] == 1
    assert df.loc[0, "basket_size"] == 2
    assert df.loc[0, "group_label"] == "benign"
    assert df.loc[0, "weight"] == 0.8


def test_load_and_prepare_mining_alert_groups_integration(tmp_path):
    path = write_cache_file(
        tmp_path,
        [
            {
                "alert_group_id": 1,
                "window_start": 10,
                "window_end": 11,
                "n_alerts": 1,
                "abs_items": [" a ", "b", "   "],
                "group_label": "benign",
                "alert_labels": ["fp"],
                "weight": 1.0,
            },
            {
                "alert_group_id": 2,
                "window_start": 12,
                "window_end": 13,
                "n_alerts": 1,
                "abs_items": [],
                "group_label": "benign",
                "alert_labels": None,
                "weight": 1.0,
            },
            {
                "alert_group_id": 3,
                "window_start": 14,
                "window_end": 15,
                "n_alerts": 1,
                "abs_items": ["x"],
                "group_label": None,
                "alert_labels": None,
                "weight": 1.0,
            },
        ],
    )

    prepared = load_and_prepare_mining_alert_groups(path=path, run_dir=tmp_path)

    assert len(prepared) == 1
    assert prepared[0].alert_group_id == 1
    assert prepared[0].items == {"a", "b"}

    assert (tmp_path / "prepared_alert_groups.csv").exists()


def test_build_tidsets_happy_path(tmp_path):
    alert_groups = [
        frozenset({"a", "b"}),
        frozenset({"b", "c"}),
        frozenset({"a"}),
    ]

    tidsets = build_tidsets(alert_groups, run_dir=tmp_path)

    assert tidsets == {
        "a": {0, 2},
        "b": {0, 1},
        "c": {1},
    }

    json_path = tmp_path / "tidsets.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload == {
        "a": [0, 2],
        "b": [0, 1],
        "c": [1],
    }


def test_build_tidsets_empty_alert_groups(tmp_path):
    tidsets = build_tidsets([], run_dir=tmp_path)

    assert tidsets == {}

    json_path = tmp_path / "tidsets.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload == {}


def test_build_tidsets_duplicate_items_in_same_basket_do_not_duplicate_tid(tmp_path):
    alert_groups = [
        frozenset(["a", "a", "b"]),
        frozenset(["a"]),
    ]

    tidsets = build_tidsets(alert_groups, run_dir=tmp_path)

    assert tidsets["a"] == {0, 1}
    assert tidsets["b"] == {0}


def test_load_mining_alert_groups_from_cache_preserves_none_optionals(tmp_path):
    path = write_cache_file(
        tmp_path,
        [
            {
                "alert_group_id": 1,
                "window_start": None,
                "window_end": None,
                "n_alerts": None,
                "abs_items": ["a"],
                "group_label": "benign",
                "alert_labels": None,
                "weight": 1.0,
            }
        ],
    )

    out = load_mining_alert_groups_from_cache(path)

    assert len(out) == 1
    assert out[0].window_start is None
    assert out[0].window_end is None
    assert out[0].n_alerts is None
