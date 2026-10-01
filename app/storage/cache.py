"""Exact-match response cache (Redis), checked after routing and before any model call.

An identical request is answered from the cache with no model call: the same
messages, the same requested model and forced tier, the same model the router
picked (so a threshold or privacy change never serves an answer from the wrong
tier), the same generation parameters and the same privacy mode. Keys are a
SHA-256 of all that, so no prompt text is ever readable in Redis. Only successful
answers are stored, with a TTL (routing.yaml `cache.ttl_s`).

The cache is an optimisation, never a dependency: every Redis error or timeout is
logged and counted, and the request carries on as a miss.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.config import CacheConfig, Tier
from app.router.engine import RouteResult
from app.schemas import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("app.cache")

KEY_VERSION = 1  # bump when the key material or the stored entry changes shape
# Request fields that change the answer. Unknown OpenAI fields are dropped on parsing
# (they are never forwarded either), so they cannot change it.
_GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "response_format",
    "tools",
    "tool_choice",
)


class CacheError(Exception):
    """The cache backend failed or timed out."""


class CacheBackend(ABC):
    name: Literal["redis", "memory"]

    @abstractmethod
    async def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl_s: int) -> None: ...

    @abstractmethod
    async def ping(self) -> None: ...

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections."""


class RedisBackend(CacheBackend):
    name = "redis"

    def __init__(self, url: str, timeout_s: float) -> None:
        self._client = redis.from_url(
            url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s
        )
        self._timeout = timeout_s

    async def _run(self, operation: str, call: Any) -> Any:  # noqa: ANN401 - redis replies
        try:
            return await asyncio.wait_for(call, self._timeout)
        except (RedisError, OSError, TimeoutError) as exc:
            # Never echo the URL: it may carry a password.
            raise CacheError(f"redis {operation} failed: {type(exc).__name__}") from exc

    async def get(self, key: str) -> bytes | None:
        return await self._run("get", self._client.get(key))

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._run("set", self._client.set(key, value, ex=ttl_s))

    async def ping(self) -> None:
        await self._run("ping", self._client.ping())

    async def close(self) -> None:
        await self._client.aclose()


class MemoryBackend(CacheBackend):
    """In-process backend with TTLs, for tests and demos without Redis."""

    name = "memory"

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.entries: dict[str, tuple[float, bytes]] = {}

    async def get(self, key: str) -> bytes | None:
        entry = self.entries.get(key)
        if entry is None or entry[0] <= self._clock():
            self.entries.pop(key, None)
            return None
        return entry[1]

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        self.entries[key] = (self._clock() + ttl_s, value)

    async def ping(self) -> None:
        return None


def cache_key(
    body: ChatCompletionRequest,
    result: RouteResult,
    forced_tier: Tier | None,
    privacy_mode: bool,
    prefix: str,
) -> str:
    """Deterministic key for the effective request. Hash only: no prompt text in the key."""
    model = result.decision.model
    material = {
        "v": KEY_VERSION,
        "messages": [m.model_dump(exclude_none=True) for m in body.messages],
        "requested_model": body.model,
        "forced_tier": forced_tier,
        "model_id": model.id,
        "model": model.model,
        "params": {f: getattr(body, f) for f in _GENERATION_FIELDS},
        "privacy_mode": privacy_mode,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return prefix + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class CachedAnswer:
    response: ChatCompletionResponse  # without the router block
    model_id: str  # the models.yaml id that produced it
    input_tokens: int
    output_tokens: int
    stored_at: float  # unix time

    def dumps(self) -> bytes:
        return json.dumps(
            {
                "v": KEY_VERSION,
                "response": self.response.model_dump(mode="json", exclude={"router"}),
                "model_id": self.model_id,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "stored_at": self.stored_at,
            }
        ).encode()

    @classmethod
    def loads(cls, raw: bytes) -> "CachedAnswer | None":
        try:
            data = json.loads(raw)
            if data.get("v") != KEY_VERSION:
                return None
            return cls(
                response=ChatCompletionResponse.model_validate(data["response"]),
                model_id=data["model_id"],
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
                stored_at=float(data["stored_at"]),
            )
        except (ValueError, KeyError, TypeError):
            return None  # a corrupt or foreign entry is a miss, not an error page


Lookup = Literal["hit", "miss", "error", "bypass"]


class ResponseCache:
    """Lookup and store with error isolation. `on_event(kind, outcome)` feeds the metrics."""

    def __init__(
        self,
        backend: CacheBackend | None,
        config: CacheConfig,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self.backend = backend if config.enabled else None
        self.config = config
        self._on_event = on_event or (lambda kind, outcome: None)

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    @property
    def status_name(self) -> Literal["redis", "memory", "disabled"]:
        return self.backend.name if self.backend is not None else "disabled"

    def applies_to(self, result: RouteResult) -> bool:
        """Whether this request may be cached at all."""
        return self.enabled and (self.config.store_pii or not result.pii_detected)

    async def lookup(self, key: str) -> tuple[Lookup, CachedAnswer | None]:
        assert self.backend is not None
        try:
            raw = await self.backend.get(key)
        except CacheError as exc:
            self._on_event("lookup", "error")
            logger.warning("cache_unavailable", extra={"operation": "get", "error": str(exc)})
            return "error", None
        entry = CachedAnswer.loads(raw) if raw is not None else None
        outcome: Lookup = "hit" if entry is not None else "miss"
        self._on_event("lookup", outcome)
        return outcome, entry

    def bypassed(self) -> None:
        self._on_event("lookup", "bypass")

    async def store(self, key: str, entry: CachedAnswer) -> None:
        """Write one entry; failures are logged and counted, never raised."""
        assert self.backend is not None
        try:
            await self.backend.set(key, entry.dumps(), self.config.ttl_s)
        except CacheError as exc:
            self._on_event("store", "error")
            logger.warning("cache_unavailable", extra={"operation": "set", "error": str(exc)})
            return
        self._on_event("store", "ok")

    async def ping(self) -> None:
        if self.backend is not None:
            await self.backend.ping()

    async def close(self) -> None:
        if self.backend is not None:
            await self.backend.close()
