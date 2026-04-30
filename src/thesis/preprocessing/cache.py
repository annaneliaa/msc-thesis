from pathlib import Path
import json
from dataclasses import asdict
from typing import Iterable, Any

from thesis.schemas.cache import (
    AlertCacheEntry,
    GroupCacheEntry,
    CacheQuery,
    CacheResponse,
)


class TokenCache:
    def __init__(self, cache_dir: Path, scenario: str) -> None:
        self.cache_dir = cache_dir
        self.alert_store_dir = cache_dir / scenario / "alerts"
        self.group_store_dir = cache_dir / scenario / "groups"

        self.alert_store_dir.mkdir(parents=True, exist_ok=True)
        self.group_store_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # serialization helpers
    # -------------------------

    @staticmethod
    def _alert_to_payload(entry: AlertCacheEntry) -> dict:
        payload = asdict(entry)
        return payload

    @staticmethod
    def _alert_from_payload(payload: dict) -> AlertCacheEntry:
        payload = dict(payload)
        return AlertCacheEntry(**payload)

    @staticmethod
    def _group_to_payload(entry: GroupCacheEntry) -> dict:
        payload = asdict(entry)

        # simple set fields
        for key in (
            "items",
            "alert_labels",
            "alert_ips",
        ):
            if key in payload and payload[key] is not None:
                payload[key] = sorted(payload[key])

        # group_features_summary: dict[str, set[str]]
        if (
            "group_features_summary" in payload
            and payload["group_features_summary"] is not None
        ):
            payload["group_features_summary"] = {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in payload["group_features_summary"].items()
            }

        # embedding_centroid may be numpy array / tuple etc.; store as JSON list
        if (
            "embedding_centroid" in payload
            and payload["embedding_centroid"] is not None
        ):
            payload["embedding_centroid"] = list(payload["embedding_centroid"])

        if (
            "mining_token_sources" in payload
            and payload["mining_token_sources"] is not None
        ):
            normalized_sources: list[Any] = []
            for src in payload["mining_token_sources"]:
                if isinstance(src, set):
                    normalized_sources.append(sorted(src))
                elif isinstance(src, dict):
                    normalized_sources.append(
                        {
                            k: sorted(v) if isinstance(v, set) else v
                            for k, v in src.items()
                        }
                    )
                else:
                    normalized_sources.append(src)
            payload["mining_token_sources"] = normalized_sources

        return payload

    @staticmethod
    def _group_from_payload(payload: dict) -> GroupCacheEntry:
        payload = dict(payload)

        if payload.get("items") is not None:
            payload["items"] = set(payload["items"])

        if "sorted_items" not in payload:
            payload["sorted_items"] = []

        if payload.get("alert_labels") is not None:
            payload["alert_labels"] = set(payload["alert_labels"])

        if payload.get("alert_ips") is not None:
            payload["alert_ips"] = set(payload["alert_ips"])

        if payload.get("group_features_summary") is not None:
            payload["group_features_summary"] = {
                k: set(v) if isinstance(v, list) else v
                for k, v in payload["group_features_summary"].items()
            }

        if payload.get("mining_token_sources") is not None:
            restored_sources: list[Any] = []
            for src in payload["mining_token_sources"]:
                if isinstance(src, list):
                    restored_sources.append(set(src))
                elif isinstance(src, dict):
                    restored_sources.append(
                        {
                            k: set(v) if isinstance(v, list) else v
                            for k, v in src.items()
                        }
                    )
                else:
                    restored_sources.append(src)
            payload["mining_token_sources"] = restored_sources

        return GroupCacheEntry(**payload)

    # -------------------------
    # single-entry writes
    # -------------------------

    def write_alert_entry(self, entry: AlertCacheEntry) -> None:
        path = self.alert_store_dir / f"{entry.alert_id}.json"
        payload = self._alert_to_payload(entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def write_group_entry(self, entry: GroupCacheEntry) -> None:
        path = self.group_store_dir / f"{entry.group_id}.json"
        payload = self._group_to_payload(entry)

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

    def write_group_batch(
        self,
        entries: Iterable[GroupCacheEntry],
        batch_name: str,
    ) -> Path:
        path = self.group_store_dir / f"{batch_name}.json"
        payload = [self._group_to_payload(entry) for entry in entries]

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

    def read_group_entry(self, group_id: str) -> GroupCacheEntry | None:
        path = self.group_store_dir / f"{group_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        return self._group_from_payload(payload)

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

    def read_group_batch(self, batch_name: str) -> list[GroupCacheEntry]:
        path = self.group_store_dir / f"{batch_name}.json"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            payloads = json.load(f)

        return [self._group_from_payload(payload) for payload in payloads]

    # -------------------------
    # listing helpers
    # -------------------------

    def list_group_ids(self) -> list[str]:
        return sorted(path.stem for path in self.group_store_dir.glob("*.json"))

    def list_alert_batch_names(self) -> list[str]:
        return sorted(path.stem for path in self.alert_store_dir.glob("*.json"))

    def list_group_batch_names(self) -> list[str]:
        return sorted(path.stem for path in self.group_store_dir.glob("*.json"))

    # -------------------------
    # query
    # -------------------------
    def query(self, query: CacheQuery) -> CacheResponse:
        groups: list[GroupCacheEntry] = []

        for group_id in self.list_group_ids():
            try:
                group = self.read_group_entry(group_id)
            except Exception as e:
                print(f"Failed to read group file: {group_id}.json -> {e}")
                raise

            if group is None:
                continue

            if query.only_closed and group.status != "closed":
                continue

            if (
                query.allowed_methods is not None
                and group.method not in query.allowed_methods
            ):
                continue

            if query.min_start_ts is not None and group.start_ts < query.min_start_ts:
                continue

            if query.max_end_ts is not None and group.end_ts > query.max_end_ts:
                continue

            if (
                query.allowed_statuses is not None
                and group.status not in query.allowed_statuses
            ):
                continue

            groups.append(group)

        groups.sort(key=lambda g: (g.end_ts, g.start_ts, g.group_id))
        print("Cache query returned {} groups".format(len(groups)))

        return CacheResponse(groups=groups)

    def query_from_batches(self, query: CacheQuery) -> CacheResponse:
        groups: list[GroupCacheEntry] = []

        for batch_name in self.list_group_batch_names():
            for group in self.read_group_batch(batch_name):
                if hasattr(query, "only_closed") and query.only_closed:
                    if (
                        getattr(group, "closed", False) is False
                        and getattr(group, "status", None) != "closed"
                    ):
                        continue

                if (
                    hasattr(query, "allowed_methods")
                    and query.allowed_methods is not None
                ):
                    if group.method not in query.allowed_methods:
                        continue

                if hasattr(query, "min_start_ts") and query.min_start_ts is not None:
                    if group.start_ts < query.min_start_ts:
                        continue

                if hasattr(query, "max_end_ts") and query.max_end_ts is not None:
                    if group.end_ts > query.max_end_ts:
                        continue

                if (
                    hasattr(query, "allowed_statuses")
                    and query.allowed_statuses is not None
                ):
                    if group.status not in query.allowed_statuses:
                        continue

                groups.append(group)

        groups.sort(key=lambda g: (g.end_ts, g.start_ts, g.group_id))
        return CacheResponse(groups=groups)
