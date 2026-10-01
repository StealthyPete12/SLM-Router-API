"""One response shape for every provider: the router block, the stored row, the headers."""

import hashlib
import json
import re

from app.aggregator.costs import Costs
from app.aggregator.dispatch import Answer, AnswerFailed
from app.aggregator.guardrails import PiiScanner
from app.router.engine import RouteResult
from app.schemas import ChatCompletionRequest, RouterMetadata
from app.storage import RequestRecord

_WHITESPACE = re.compile(r"\s+")


def last_user_message(body: ChatCompletionRequest) -> str:
    return next((m.content or "" for m in reversed(body.messages) if m.role == "user"), "")


def prompt_preview(body: ChatCompletionRequest, scanner: PiiScanner, chars: int) -> str:
    """The last user message, PII-masked, on one line and cut to `chars` characters."""
    text = _WHITESPACE.sub(" ", scanner.mask(last_user_message(body))).strip()
    return text if len(text) <= chars else f"{text[: chars - 1]}…"


def prompt_hash(body: ChatCompletionRequest) -> str:
    """SHA-256 of every message, to spot repeats without storing the prompt."""
    messages = [[m.role, m.content] for m in body.messages]
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


def answered_metadata(
    result: RouteResult, answer: Answer, costs: Costs, cache_hit: bool = False
) -> RouterMetadata:
    meta = result.metadata(latency_ms=answer.latency_ms)
    update: dict[str, object] = {
        "cost_usd": costs.cost_usd,
        "baseline_cost_usd": costs.baseline_cost_usd,
        "savings_usd": costs.savings_usd,
        "cache_hit": cache_hit,
    }
    if answer.fallback_from is not None:
        update |= {
            "tier": answer.model.tier,
            "model_id": answer.model.id,
            "model": answer.model.model,
            "fallback_from": answer.fallback_from,
            "fallback_cause": answer.fallback_cause,
            "summary": (
                f"{meta.summary}; {answer.fallback_from} failed ({answer.fallback_cause}), "
                f"so {answer.model.id} answered"
            ),
        }
    return meta.model_copy(update=update)


def router_headers(meta: RouterMetadata) -> dict[str, str]:
    """The decision as headers, for clients that drop unknown JSON fields."""
    headers = {
        "X-Router-Tier": meta.tier,
        "X-Router-Model": meta.model_id,
        "X-Router-Score": str(meta.complexity_score),
    }
    if meta.fallback_from is not None:
        headers["X-Router-Fallback-From"] = meta.fallback_from
    return headers


def request_record(
    *,
    request_id: str,
    client: str,
    body: ChatCompletionRequest,
    result: RouteResult,
    scanner: PiiScanner,
    preview_chars: int,
    meta: RouterMetadata | None = None,
    answer: Answer | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    failure: AnswerFailed | None = None,
) -> RequestRecord:
    """The `requests` row for an answered request (meta + answer) or a failed one."""
    if answer is not None and meta is not None:
        model, latency_ms = answer.model, answer.latency_ms
        fallback_from, fallback_cause = answer.fallback_from, answer.fallback_cause
    else:
        assert failure is not None
        model, latency_ms = failure.model, failure.latency_ms
        fallback_from, fallback_cause = failure.fallback_from, None
        meta = result.metadata(latency_ms=latency_ms)
    return RequestRecord(
        id=request_id,
        client=client,
        task_type=meta.task_type,
        complexity_score=meta.complexity_score,
        threshold=meta.threshold,
        signals=meta.signals,
        tier=model.tier,
        reason=meta.reason,
        model_id=model.id,
        model=model.model,
        provider=model.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=meta.cost_usd,
        baseline_cost_usd=meta.baseline_cost_usd,
        savings_usd=meta.savings_usd,
        cache_hit=meta.cache_hit,
        fallback_from=fallback_from,
        fallback_cause=fallback_cause,
        pii_detected=result.pii_detected,
        status="ok" if failure is None else "error",
        status_code=200 if failure is None else failure.status_code,
        error=None if failure is None else scanner.mask(failure.message)[:1000],
        prompt_preview=prompt_preview(body, scanner, preview_chars),
        prompt_hash=prompt_hash(body),
    )
