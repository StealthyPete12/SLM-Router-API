"""Store and recorder units: percentile maths, share categories, background writes."""

import asyncio
from dataclasses import replace

import pytest

from app.storage import MemoryStore, Recorder, RequestRecord, StorageError
from app.storage.memory import percentile_cont

ROW = RequestRecord(
    id="req_1",
    client="api",
    task_type="qa",
    complexity_score=10,
    threshold=30,
    signals={"task_base": 10},
    tier="local",
    reason="score_below_threshold",
    model_id="phi3-mini",
    model="phi3:mini",
    provider="ollama",
    input_tokens=10,
    output_tokens=20,
    latency_ms=100,
    cost_usd=0.0,
    baseline_cost_usd=0.001,
    savings_usd=0.001,
    cache_hit=False,
    fallback_from=None,
    fallback_cause=None,
    pii_detected=False,
    status="ok",
    status_code=200,
    error=None,
    prompt_preview="What is the capital of France?",
    prompt_hash="0" * 64,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], None), ([5], 5), ([1, 2, 3, 4, 5], 4.8), (list(range(1, 101)), 95.05)],
)
def test_percentile_matches_postgres_percentile_cont(values, expected):
    result = percentile_cont(values, 0.95)
    assert result == (pytest.approx(expected) if expected is not None else None)


def test_memory_stats_categories():
    store = MemoryStore()
    rows = [
        ROW,
        replace(ROW, id="r2", tier="premium", reason="score_at_or_above_threshold", savings_usd=0),
        replace(ROW, id="r3", tier="premium", fallback_from="phi3-mini", savings_usd=0),
        replace(ROW, id="r4", cache_hit=True),
        replace(ROW, id="r5", status="error", status_code=502, latency_ms=None),
    ]

    async def run():
        for row in rows:
            await store.record_request(row)
        return await store.stats(None)

    stats = asyncio.run(run())
    assert (stats.requests, stats.ok) == (5, 4)
    assert (stats.local, stats.premium, stats.fallbacks, stats.cache_hits) == (1, 1, 1, 1)


class FailingStore(MemoryStore):
    async def record_request(self, record):
        raise StorageError("database is down")


def test_recorder_logs_and_counts_failed_writes():
    errors: list[str] = []

    async def run():
        recorder = Recorder(FailingStore(), on_error=errors.append)
        recorder.submit(ROW)
        await recorder.wait_for(ROW.id)
        await recorder.drain()

    asyncio.run(run())
    assert errors == ["record_request"]


def test_feedback_waits_for_the_pending_write():
    store = MemoryStore()

    async def run():
        recorder = Recorder(store)
        recorder.submit(ROW)
        # Not written yet: the task has not run.
        assert not await store.request_exists(ROW.id)
        await recorder.wait_for(ROW.id)
        return await store.save_feedback(ROW.id, 1, None)

    assert asyncio.run(run()) == "recorded"
