"""Postgres store (asyncpg). The schema lives in infra/postgres/init.sql; no migration tool.

The pool is opened lazily: if Postgres is down at startup the API still serves
answers, logs each failed write, and reconnects on the next request.
"""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal

import asyncpg

from app.storage.base import FeedbackStatus, RequestRecord, Stats, StorageError, Store

logger = logging.getLogger("app.storage")

_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)

_COLUMNS = (
    "id",
    "created_at",
    "client",
    "task_type",
    "complexity_score",
    "threshold",
    "signals",
    "tier",
    "reason",
    "model_id",
    "model",
    "provider",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
    "baseline_cost_usd",
    "savings_usd",
    "cache_hit",
    "fallback_from",
    "fallback_cause",
    "pii_detected",
    "status",
    "status_code",
    "error",
    "prompt_preview",
    "prompt_hash",
)
_INSERT_REQUEST = (
    f"INSERT INTO requests ({', '.join(_COLUMNS)}) VALUES ("
    + ", ".join(f"${i}::jsonb" if c == "signals" else f"${i}" for i, c in enumerate(_COLUMNS, 1))
    + ")"
)

_UPSERT_FEEDBACK = """
INSERT INTO feedback (request_id, rating, comment) VALUES ($1, $2, $3)
ON CONFLICT (request_id) DO UPDATE
    SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, updated_at = now()
RETURNING (xmax = 0) AS inserted
"""

# Shares are over successful requests; a cache hit is counted once, as a cache hit.
# Per-tier p95 leaves cache hits out (no model ran); the overall p95 keeps them.
_STATS = """
WITH r AS (
    SELECT * FROM requests WHERE $1::timestamptz IS NULL OR created_at >= $1
), ok AS (
    SELECT * FROM r WHERE status = 'ok'
)
SELECT
    (SELECT count(*) FROM r) AS requests,
    count(*) AS ok,
    count(*) FILTER (WHERE NOT cache_hit AND fallback_from IS NULL AND tier = 'local') AS local,
    count(*) FILTER (WHERE NOT cache_hit AND fallback_from IS NULL AND tier = 'premium')
        AS premium,
    count(*) FILTER (WHERE NOT cache_hit AND fallback_from IS NOT NULL) AS fallbacks,
    count(*) FILTER (WHERE cache_hit) AS cache_hits,
    sum(cost_usd) AS cost_usd,
    sum(baseline_cost_usd) FILTER (WHERE savings_usd IS NOT NULL) AS baseline_cost_usd,
    sum(savings_usd) AS savings_usd,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
        FILTER (WHERE tier = 'local' AND NOT cache_hit) AS p95_local_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
        FILTER (WHERE tier = 'premium' AND NOT cache_hit) AS p95_premium_ms,
    (SELECT count(*) FROM feedback f
        WHERE f.rating = 1 AND ($1::timestamptz IS NULL OR f.created_at >= $1)) AS feedback_up,
    (SELECT count(*) FROM feedback f
        WHERE f.rating = -1 AND ($1::timestamptz IS NULL OR f.created_at >= $1)) AS feedback_down
FROM ok
"""


def _number(value: Decimal | float | None) -> float | None:
    return None if value is None else float(value)


class PostgresStore(Store):
    backend = "postgres"

    def __init__(self, dsn: str, timeout: float = 5.0) -> None:
        self._dsn = dsn
        self._timeout = timeout
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                try:
                    self._pool = await asyncpg.create_pool(
                        self._dsn,
                        min_size=1,
                        max_size=5,
                        timeout=self._timeout,
                        command_timeout=self._timeout * 2,
                    )
                except _ERRORS as exc:
                    # Never echo the DSN: it carries the password.
                    raise StorageError(f"cannot connect to Postgres: {type(exc).__name__}") from exc
        return self._pool

    async def start(self) -> None:
        try:
            await self._get_pool()
        except StorageError as exc:
            logger.warning("storage_unavailable_at_startup", extra={"error": str(exc)})

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> None:
        pool = await self._get_pool()
        try:
            await asyncio.wait_for(pool.fetchval("SELECT 1"), self._timeout)
        except _ERRORS as exc:
            raise StorageError(f"Postgres did not answer: {type(exc).__name__}") from exc

    async def record_request(self, record: RequestRecord) -> None:
        values = [getattr(record, c) for c in _COLUMNS]
        values[_COLUMNS.index("signals")] = json.dumps(record.signals)
        pool = await self._get_pool()
        try:
            await pool.execute(_INSERT_REQUEST, *values)
        except _ERRORS as exc:
            raise StorageError(f"insert into requests failed: {exc}") from exc

    async def request_exists(self, request_id: str) -> bool:
        pool = await self._get_pool()
        try:
            return bool(
                await pool.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM requests WHERE id = $1)", request_id
                )
            )
        except _ERRORS as exc:
            raise StorageError(f"request lookup failed: {exc}") from exc

    async def save_feedback(
        self, request_id: str, rating: int, comment: str | None
    ) -> FeedbackStatus | None:
        pool = await self._get_pool()
        try:
            inserted = await pool.fetchval(_UPSERT_FEEDBACK, request_id, rating, comment)
        except asyncpg.ForeignKeyViolationError:
            return None
        except _ERRORS as exc:
            raise StorageError(f"saving feedback failed: {exc}") from exc
        return "recorded" if inserted else "updated"

    async def stats(self, since: datetime | None) -> Stats:
        pool = await self._get_pool()
        try:
            row = await pool.fetchrow(_STATS, since)
        except _ERRORS as exc:
            raise StorageError(f"stats query failed: {exc}") from exc
        assert row is not None  # an aggregate query always returns one row
        return Stats(
            requests=row["requests"],
            ok=row["ok"],
            local=row["local"],
            premium=row["premium"],
            fallbacks=row["fallbacks"],
            cache_hits=row["cache_hits"],
            cost_usd=_number(row["cost_usd"]),
            baseline_cost_usd=_number(row["baseline_cost_usd"]),
            savings_usd=_number(row["savings_usd"]),
            p95_ms=_number(row["p95_ms"]),
            p95_local_ms=_number(row["p95_local_ms"]),
            p95_premium_ms=_number(row["p95_premium_ms"]),
            feedback_up=row["feedback_up"],
            feedback_down=row["feedback_down"],
        )
