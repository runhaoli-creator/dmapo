"""Response caching for the DMAPO judge API.

Supports an in-process LRU or a Redis backend, selected by
``DMAPO_CACHE_BACKEND``. Keys hash ``(prompt, candidate, judge_set)`` so
identical (prompt, candidate) pairs scored under the same judge configuration
are served from cache on the second call.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    @property
    def backend_name(self) -> str: ...


def make_key(prompt: str, candidate: str, judges: list[str] | None, process: bool) -> str:
    payload = json.dumps(
        {"p": prompt, "c": candidate, "j": sorted(judges or []), "pc": process},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LRUMemoryCache:
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    @property
    def backend_name(self) -> str:
        return "memory"


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 3600):
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis backend requires redis-py") from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value, ex=self._ttl)

    @property
    def backend_name(self) -> str:
        return "redis"


def build_cache_from_env() -> CacheBackend:
    backend = os.getenv("DMAPO_CACHE_BACKEND", "memory").lower()
    if backend == "redis":
        url = os.getenv("DMAPO_REDIS_URL", "redis://localhost:6379/1")
        ttl = int(os.getenv("DMAPO_REDIS_TTL", "3600"))
        return RedisCache(url, ttl_seconds=ttl)
    return LRUMemoryCache(capacity=int(os.getenv("DMAPO_LRU_CAPACITY", "4096")))
