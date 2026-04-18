from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict

from thesis.schemas.cache import AlertCacheEntry, WindowCacheEntry


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

        return WindowCacheEntry(**payload)
