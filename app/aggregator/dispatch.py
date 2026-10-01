"""Calls the routed model: one retry for premium, and the local-to-premium fallback.

- Local timeout, error or empty answer: retry once on the premium default and
  record `fallback_from`. Never for privacy-kept prompts, an explicit model ID or
  a forced local tier; those fail with an error that says why there was no fallback.
- Premium unreachable, timed out, 429 or 5xx: retried after a backoff
  (routing.yaml `retry`), then the error goes back to the client.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.aggregator.guardrails import is_empty_answer
from app.config import AppConfig, ModelConfig
from app.providers import (
    Provider,
    ProviderError,
    ProviderNotConfigured,
    ProviderRegistry,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.router.engine import RouteResult
from app.schemas import ChatCompletionResponse, FallbackCause

logger = logging.getLogger("app")

# Replaced in tests, so retry backoff costs no wall time.
sleep = asyncio.sleep


@dataclass(frozen=True)
class Answer:
    response: ChatCompletionResponse
    model: ModelConfig  # the model that answered
    latency_ms: int
    fallback_from: str | None = None
    fallback_cause: FallbackCause | None = None


class AnswerFailed(Exception):
    """No usable answer. status_code is what the client gets."""

    def __init__(
        self,
        status_code: int,
        message: str,
        model: ModelConfig,
        cause: FallbackCause | None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.model = model  # the model being called when it failed
        self.cause = cause
        self.retryable = retryable
        self.fallback_from: str | None = None
        self.latency_ms: int | None = None


class Dispatcher:
    def __init__(self, config: AppConfig, providers: ProviderRegistry) -> None:
        self._config = config
        self._providers = providers

    async def answer(self, payload: dict[str, Any], result: RouteResult) -> Answer:
        started = time.perf_counter()

        def elapsed() -> int:
            return round((time.perf_counter() - started) * 1000)

        primary = result.decision.model
        try:
            response = await self._call(primary, payload)
        except AnswerFailed as first:
            if primary.tier != "local":
                first.latency_ms = elapsed()
                raise
            target, blocked = self._fallback_target(result)
            if target is None:
                first.message = f"{first.message}; no cloud fallback: {blocked}"
                first.latency_ms = elapsed()
                raise
            logger.warning(
                "fallback",
                extra={"from_model": primary.id, "to_model": target.id, "cause": first.cause},
            )
            try:
                response = await self._call(target, payload)
            except AnswerFailed as second:
                second.message = (
                    f"local {primary.id} failed ({first.message}); "
                    f"premium fallback {target.id} also failed: {second.message}"
                )
                second.fallback_from = primary.id
                second.latency_ms = elapsed()
                raise
            return Answer(response, target, elapsed(), primary.id, first.cause)
        return Answer(response, primary, elapsed())

    def _fallback_target(self, result: RouteResult) -> tuple[ModelConfig | None, str]:
        """The premium model to fall back to, or None and the reason there is none."""
        routing = self._config.routing
        if not routing.fallback.local_to_premium:
            return None, "fallback.local_to_premium is off in routing.yaml"
        if routing.privacy_mode and result.pii_detected:
            return None, "PII was detected and privacy_mode keeps this prompt on this machine"
        if result.decision.reason == "explicit_model":
            return None, "a model asked for by ID is never swapped"
        if result.decision.reason == "caller_override":
            return None, "the caller forced X-Router-Tier: local"
        target = self._config.default_model("premium")
        if target.missing_settings:
            return (
                None,
                f"the premium model is not configured (set {', '.join(target.missing_settings)})",
            )
        if not result.required_capabilities <= target.capabilities:
            return None, f"{target.id} lacks a capability the request needs"
        if result.input_tokens + result.output_reserve > target.context_window:
            return None, f"the request does not fit {target.id}'s context window"
        return target, ""

    async def _call(self, model: ModelConfig, payload: dict[str, Any]) -> ChatCompletionResponse:
        try:
            provider = self._providers.for_model(model)
        except ProviderNotConfigured as exc:
            raise AnswerFailed(503, str(exc), model, None) from exc
        retry = self._config.routing.retry
        attempts = retry.premium_attempts if model.tier == "premium" else 1
        timeout = getattr(self._config.routing.timeouts_s, model.tier)
        for attempt in range(1, attempts + 1):
            try:
                return await self._call_once(provider, model, payload, timeout)
            except AnswerFailed as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = retry.backoff_s * attempt
                logger.warning(
                    "premium_retry",
                    extra={
                        "model_id": model.id,
                        "attempt": attempt,
                        "delay_s": delay,
                        "error": exc.message,
                    },
                )
                await sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    async def _call_once(
        self, provider: Provider, model: ModelConfig, payload: dict[str, Any], timeout: float
    ) -> ChatCompletionResponse:
        name = provider.name
        try:
            # for_model() only returns a provider once `model.model` is set.
            data = await provider.chat_completion(model.model or "", payload, timeout)
            answer = ChatCompletionResponse.model_validate(data)
        except ProviderTimeout as exc:
            raise AnswerFailed(504, str(exc), model, "timeout", retryable=True) from exc
        except ProviderUnavailable as exc:
            raise AnswerFailed(502, str(exc), model, "unreachable", retryable=True) from exc
        except ProviderError as exc:
            pulled_hint = model.tier == "local" and exc.status_code == 404
            hint = " (is the model pulled? run `make models`)" if pulled_hint else ""
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise AnswerFailed(502, f"{exc}{hint}", model, "error", retryable) from exc
        except ValidationError as exc:
            message = f"unexpected response shape from {name}: {exc}"
            raise AnswerFailed(502, message, model, "bad_response") from exc
        if is_empty_answer(answer):
            raise AnswerFailed(502, f"{name} returned an empty answer", model, "empty_answer")
        return answer
