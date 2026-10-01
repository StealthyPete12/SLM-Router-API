"""What gets stored for each request, and the interface every store implements.

Prompts are never stored in full: only a PII-masked preview and a SHA-256 hash.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

FeedbackStatus = Literal["recorded", "updated"]


class StorageError(Exception):
    """The store could not be reached or rejected the operation."""


@dataclass(frozen=True)
class RequestRecord:
    """One row of the `requests` table (infra/postgres/init.sql)."""

    id: str
    client: str
    task_type: str
    complexity_score: int
    threshold: int
    signals: dict[str, int]
    tier: str  # the tier that answered (or was being called when it failed)
    reason: str
    model_id: str
    model: str | None
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    savings_usd: float | None
    cache_hit: bool
    fallback_from: str | None
    fallback_cause: str | None
    pii_detected: bool
    status: Literal["ok", "error"]
    status_code: int
    error: str | None
    prompt_preview: str
    prompt_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Stats:
    """Aggregates for GET /v1/stats over successful requests in the window."""

    requests: int = 0
    ok: int = 0
    local: int = 0  # answered locally, no fallback
    premium: int = 0  # routed to and answered by premium
    fallbacks: int = 0  # local failed, premium answered
    cache_hits: int = 0
    cost_usd: float | None = None
    baseline_cost_usd: float | None = None  # over requests whose savings are known
    savings_usd: float | None = None
    p95_ms: float | None = None
    p95_local_ms: float | None = None
    p95_premium_ms: float | None = None
    feedback_up: int = 0
    feedback_down: int = 0


class Store(ABC):
    backend: Literal["postgres", "memory"]

    async def start(self) -> None:  # noqa: B027 - optional hook
        """Open connections. Must not raise: a store that is down is retried later."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections."""

    @abstractmethod
    async def ping(self) -> None:
        """Raise StorageError when the store cannot be used."""

    @abstractmethod
    async def record_request(self, record: RequestRecord) -> None: ...

    @abstractmethod
    async def request_exists(self, request_id: str) -> bool: ...

    @abstractmethod
    async def save_feedback(
        self, request_id: str, rating: int, comment: str | None
    ) -> FeedbackStatus | None:
        """Store or replace the one rating of a request; None if the request is unknown."""

    @abstractmethod
    async def stats(self, since: datetime | None) -> Stats: ...
