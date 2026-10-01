"""Writes request rows in the background, so the response never waits on Postgres.

A client may send feedback the moment its answer arrives, before the row is
written; `wait_for` lets the feedback endpoint wait for that one write.
"""

import asyncio
import logging
from collections.abc import Callable

from app.storage.base import RequestRecord, StorageError, Store

logger = logging.getLogger("app.storage")


class Recorder:
    def __init__(self, store: Store, on_error: Callable[[str], None] | None = None) -> None:
        self.store = store
        self._on_error = on_error or (lambda operation: None)
        self._pending: dict[str, asyncio.Event] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, record: RequestRecord) -> None:
        """Schedule the write and return at once. Failures are logged, never raised."""
        done = asyncio.Event()
        self._pending[record.id] = done
        task = asyncio.create_task(self._write(record, done))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _write(self, record: RequestRecord, done: asyncio.Event) -> None:
        try:
            await self.store.record_request(record)
        except StorageError as exc:
            self._on_error("record_request")
            logger.error("storage_write_failed", extra={"request_id": record.id, "error": str(exc)})
        finally:
            done.set()
            self._pending.pop(record.id, None)

    async def wait_for(self, request_id: str, timeout: float = 5.0) -> None:
        """Wait until a pending write of this request has finished (or timed out)."""
        done = self._pending.get(request_id)
        if done is not None:
            try:
                await asyncio.wait_for(done.wait(), timeout)
            except TimeoutError:
                logger.warning("storage_write_slow", extra={"request_id": request_id})

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for every pending write; used before reading stats and at shutdown."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)
