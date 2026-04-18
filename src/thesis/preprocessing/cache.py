from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict

from thesis.schemas.cache import (
    AlertCacheEntry,
    WindowCacheEntry,
    CacheQuery,
    CacheResponse,
)


class TokenCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.alert_store_dir = cache_dir / "alerts"
        self.window_store_dir = cache_dir / "windows"

        self.alert_store_dir.mkdir(parents=True, exist_ok=True)
        self.window_store_dir.mkdir(parents=True, exist_ok=True)

    def write_alert_entry(self, entry: AlertCacheEntry) -> None:
        path = self.alert_store_dir / f"{entry.alert_id}.json"
        payload = asdict(entry)

        payload["repr_tokens"] = sorted(payload["repr_tokens"])
        payload["mining_tokens"] = sorted(payload["mining_tokens"])

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def write_window_entry(self, entry: WindowCacheEntry) -> None:
        path = self.window_store_dir / f"{entry.window_id}.json"
        payload = asdict(entry)

        payload["items"] = sorted(payload["items"])
        payload["hosts"] = sorted(payload["hosts"])
        payload["signatures"] = sorted(payload["signatures"])

        if payload["alert_labels"] is not None:
            payload["alert_labels"] = sorted(payload["alert_labels"])

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def read_alert_entry(self, alert_id: str) -> AlertCacheEntry | None:
        path = self.alert_store_dir / f"{alert_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["repr_tokens"] = set(payload["repr_tokens"])
        payload["mining_tokens"] = set(payload["mining_tokens"])

        return AlertCacheEntry(**payload)

    def read_window_entry(self, window_id: int) -> WindowCacheEntry | None:
        path = self.window_store_dir / f"{window_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["items"] = set(payload["items"])
        payload["hosts"] = set(payload["hosts"])
        payload["signatures"] = set(payload["signatures"])

        if payload.get("alert_labels") is not None:
            payload["alert_labels"] = set(payload["alert_labels"])

        return WindowCacheEntry(**payload)

        return WindowCacheEntry(**payload)

    def list_window_ids(self) -> list[int]:
        window_ids: list[int] = []

        for path in self.window_store_dir.glob("*.json"):
            try:
                window_ids.append(int(path.stem))
            except ValueError:
                continue

        return sorted(window_ids)

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
