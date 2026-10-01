"""In-process store, used when DATABASE_URL is unset (tests and quick runs). Not persisted."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from app.storage.base import FeedbackStatus, RequestRecord, Stats, Store


def percentile_cont(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile, the same as Postgres percentile_cont."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _sum(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


@dataclass
class _Feedback:
    rating: int
    comment: str | None
    created_at: datetime


class MemoryStore(Store):
    backend = "memory"

    def __init__(self) -> None:
        self.requests: dict[str, RequestRecord] = {}
        self.feedback: dict[str, _Feedback] = {}

    async def ping(self) -> None:
        return None

    async def record_request(self, record: RequestRecord) -> None:
        self.requests[record.id] = record

    async def request_exists(self, request_id: str) -> bool:
        return request_id in self.requests

    async def save_feedback(
        self, request_id: str, rating: int, comment: str | None
    ) -> FeedbackStatus | None:
        if request_id not in self.requests:
            return None
        status: FeedbackStatus = "updated" if request_id in self.feedback else "recorded"
        self.feedback[request_id] = _Feedback(rating, comment, datetime.now(UTC))
        return status

    async def stats(self, since: datetime | None) -> Stats:
        rows = [r for r in self.requests.values() if since is None or r.created_at >= since]
        ok = [r for r in rows if r.status == "ok"]
        served = [r for r in ok if not r.cache_hit]
        with_savings = [r for r in ok if r.savings_usd is not None]

        def p95(tier: str | None) -> float | None:
            # Per-tier p95 leaves cache hits out (no model ran); the overall p95 keeps them.
            latencies = [
                float(r.latency_ms)
                for r in ok
                if r.latency_ms is not None
                and (tier is None or (r.tier == tier and not r.cache_hit))
            ]
            return percentile_cont(latencies, 0.95)

        votes = [
            f.rating
            for request_id, f in self.feedback.items()
            if since is None or f.created_at >= since
            if request_id in self.requests
        ]
        return Stats(
            requests=len(rows),
            ok=len(ok),
            local=sum(r.fallback_from is None and r.tier == "local" for r in served),
            premium=sum(r.fallback_from is None and r.tier == "premium" for r in served),
            fallbacks=sum(r.fallback_from is not None for r in served),
            cache_hits=len(ok) - len(served),
            cost_usd=_sum([r.cost_usd for r in ok]),
            baseline_cost_usd=_sum([r.baseline_cost_usd for r in with_savings]),
            savings_usd=_sum([r.savings_usd for r in with_savings]),
            p95_ms=p95(None),
            p95_local_ms=p95("local"),
            p95_premium_ms=p95("premium"),
            feedback_up=votes.count(1),
            feedback_down=votes.count(-1),
        )
