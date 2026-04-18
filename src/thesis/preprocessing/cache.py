from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict
from typing import Iterable

from thesis.schemas.cache import (
    AlertCacheEntry,
    WindowCacheEntry,
    CacheQuery,
    CacheResponse,
)


class TokenCache:
    def __init__(self, cache_dir: Path, scenario: str) -> None:
        self.cache_dir = cache_dir
        self.alert_store_dir = cache_dir / scenario / "alerts"
        self.window_store_dir = cache_dir / scenario / "windows"

        self.alert_store_dir.mkdir(parents=True, exist_ok=True)
        self.window_store_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # serialization helpers
    # -------------------------

    @staticmethod
    def _alert_to_payload(entry: AlertCacheEntry) -> dict:
        payload = asdict(entry)
        payload["repr_tokens"] = sorted(payload["repr_tokens"])
        payload["mining_tokens"] = sorted(payload["mining_tokens"])
        return payload

    @staticmethod
    def _alert_from_payload(payload: dict) -> AlertCacheEntry:
        payload = dict(payload)
        payload["repr_tokens"] = set(payload["repr_tokens"])
        payload["mining_tokens"] = set(payload["mining_tokens"])
        return AlertCacheEntry(**payload)

    @staticmethod
    def _window_to_payload(entry: WindowCacheEntry) -> dict:
        payload = asdict(entry)
        payload["items"] = sorted(payload["items"])
        payload["hosts"] = sorted(payload["hosts"])
        payload["signatures"] = sorted(payload["signatures"])

        if payload["alert_labels"] is not None:
            payload["alert_labels"] = sorted(payload["alert_labels"])

        return payload

    @staticmethod
    def _window_from_payload(payload: dict) -> WindowCacheEntry:
        payload = dict(payload)
        payload["items"] = set(payload["items"])
        payload["hosts"] = set(payload["hosts"])
        payload["signatures"] = set(payload["signatures"])

        if payload.get("alert_labels") is not None:
            payload["alert_labels"] = set(payload["alert_labels"])

        return WindowCacheEntry(**payload)

    # -------------------------
    # single-entry writes
    # -------------------------

    def write_alert_entry(self, entry: AlertCacheEntry) -> None:
        path = self.alert_store_dir / f"{entry.alert_id}.json"
        payload = self._alert_to_payload(entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def write_window_entry(self, entry: WindowCacheEntry) -> None:
        path = self.window_store_dir / f"{entry.window_id}.json"
        payload = self._window_to_payload(entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    # -------------------------
    # batch writes
    # -------------------------

    def write_alert_batch(
        self,
        entries: Iterable[AlertCacheEntry],
        batch_name: str,
    ) -> Path:
        path = self.alert_store_dir / f"{batch_name}.json"
        payload = [self._alert_to_payload(entry) for entry in entries]

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        return path

    def write_window_batch(
        self,
        entries: Iterable[WindowCacheEntry],
        batch_name: str,
    ) -> Path:
        path = self.window_store_dir / f"{batch_name}.json"
        payload = [self._window_to_payload(entry) for entry in entries]

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        return path

    # -------------------------
    # single-entry reads
    # -------------------------

    def read_alert_entry(self, alert_id: str) -> AlertCacheEntry | None:
        path = self.alert_store_dir / f"{alert_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        return self._alert_from_payload(payload)

    def read_window_entry(self, window_id: int) -> WindowCacheEntry | None:
        path = self.window_store_dir / f"{window_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        return self._window_from_payload(payload)

    # -------------------------
    # batch reads
    # -------------------------

    def read_alert_batch(self, batch_name: str) -> list[AlertCacheEntry]:
        path = self.alert_store_dir / f"{batch_name}.json"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            payloads = json.load(f)

        return [self._alert_from_payload(payload) for payload in payloads]

    def read_window_batch(self, batch_name: str) -> list[WindowCacheEntry]:
        path = self.window_store_dir / f"{batch_name}.json"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            payloads = json.load(f)

        return [self._window_from_payload(payload) for payload in payloads]

    # -------------------------
    # listing helpers
    # -------------------------

    def list_window_ids(self) -> list[int]:
        window_ids: list[int] = []

        for path in self.window_store_dir.glob("*.json"):
            try:
                window_ids.append(int(path.stem))
            except ValueError:
                continue

        return sorted(window_ids)

    def list_alert_batch_names(self) -> list[str]:
        return sorted(path.stem for path in self.alert_store_dir.glob("*.json"))

    def list_window_batch_names(self) -> list[str]:
        return sorted(path.stem for path in self.window_store_dir.glob("*.json"))

    # -------------------------
    # query
    # -------------------------

    def query(self, query: CacheQuery) -> CacheResponse:
        windows: list[WindowCacheEntry] = []

        for window_id in self.list_window_ids():
            if query.min_window_id is not None and window_id < query.min_window_id:
                continue
            if query.max_window_id is not None and window_id > query.max_window_id:
                continue

            window = self.read_window_entry(window_id)
            if window is None:
                continue

            if query.only_closed and not window.closed:
                continue

            windows.append(window)

        return CacheResponse(windows=windows)

    def query_from_batches(self, query: CacheQuery) -> CacheResponse:
        windows: list[WindowCacheEntry] = []

        for batch_name in self.list_window_batch_names():
            for window in self.read_window_batch(batch_name):
                if (
                    query.min_window_id is not None
                    and window.window_id < query.min_window_id
                ):
                    continue
                if (
                    query.max_window_id is not None
                    and window.window_id > query.max_window_id
                ):
                    continue
                if query.only_closed and not window.closed:
                    continue
                windows.append(window)

        windows.sort(key=lambda w: w.window_id)
        return CacheResponse(windows=windows)
