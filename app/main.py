"""FastAPI app: the gateway. Auth, request IDs, routing, answers, feedback and stats.

Gateway, router (app/router), aggregator (app/aggregator), providers
(app/providers) and storage (app/storage) are packages inside this one service;
they are not separate deployments.
"""

import logging
import re
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, NoReturn, cast
from uuid import uuid4

import httpx
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.aggregator.costs import Costs, compute_costs, price
from app.aggregator.dispatch import Answer, AnswerFailed, Dispatcher
from app.aggregator.formatter import answered_metadata, request_record, router_headers
from app.aggregator.guardrails import input_chars
from app.config import TIERS, AppConfig, Settings, Tier, load_config
from app.log import configure_logging
from app.metrics import Metrics
from app.providers import ProviderError, ProviderRegistry, ProviderUnavailable
from app.router.engine import Router, RouteResult
from app.router.policy import RoutingError
from app.schemas import (
    CacheStatus,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CostEstimate,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStats,
    HealthResponse,
    LatencyP95,
    ModelCard,
    ModelList,
    OllamaStatus,
    PremiumStatus,
    RouteResponse,
    StatsResponse,
    StorageStatus,
)
from app.storage import MemoryStore, PostgresStore, Recorder, StorageError, Store
from app.storage.cache import (
    CacheBackend,
    CachedAnswer,
    CacheError,
    RedisBackend,
    ResponseCache,
    cache_key,
)

logger = logging.getLogger("app")

TierHeader = Annotated[
    str | None,
    Header(alias="X-Router-Tier", description="Force a tier: local or premium"),
]
ClientHeader = Annotated[
    str | None,
    Header(
        alias="X-Router-Client",
        description="Optional label stored with the request, e.g. playground or demo-loader",
    ),
]
CacheControlHeader = Annotated[
    str | None,
    Header(
        alias="Cache-Control",
        description="no-cache: skip the cache lookup (the answer is still stored); "
        "no-store: neither read nor write the cache",
    ),
]
_CLIENT_LABEL = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
DEFAULT_CLIENT = "api"


def create_app(
    config: AppConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    settings: Settings | None = None,
    store: Store | None = None,
    cache_backend: CacheBackend | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own config, settings, store, cache backend and a
    mock HTTP transport. Without a cache backend or REDIS_URL, the cache is off."""
    settings = settings or Settings()
    configure_logging(settings.log_level)
    if config is None:
        config = load_config(settings.config_dir, settings.ollama_context_length)
    router = Router(config)
    premium_model = config.default_model("premium")
    baseline_model = config.baseline_model
    guardrails = config.routing.guardrails
    if store is None:
        if settings.database_url is not None:
            store = PostgresStore(settings.database_url.get_secret_value())
        else:
            store = MemoryStore()
    metrics = Metrics()
    recorder = Recorder(store, on_error=lambda op: metrics.storage_errors.labels(op).inc())
    cache_config = config.routing.cache
    if cache_backend is None and settings.redis_url is not None:
        cache_backend = RedisBackend(settings.redis_url.get_secret_value(), cache_config.timeout_s)
    cache = ResponseCache(
        cache_backend,
        cache_config,
        on_event=lambda operation, result: metrics.cache.labels(operation, result).inc(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await store.start()
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.providers = ProviderRegistry(client)
            app.state.dispatcher = Dispatcher(config, app.state.providers)
            logger.info(
                "startup",
                extra={
                    "ollama_base_url": config.default_local_model.base_url,
                    "local_models": [m.id for m in config.models.local],
                    "premium_model": premium_model.id,
                    "premium_missing": premium_model.missing_settings,
                    "threshold": config.routing.threshold,
                    "auth_configured": settings.router_api_key is not None,
                    "storage": store.backend,
                    "baseline_model": baseline_model.id,
                    "baseline_priced": baseline_model.priced,
                    "cache": cache.status_name,
                    "cache_ttl_s": cache_config.ttl_s if cache.enabled else None,
                },
            )
            if settings.router_api_key is None:
                logger.warning("ROUTER_API_KEY is not set; every /v1 endpoint will answer 503")
            if store.backend == "memory":
                logger.warning("DATABASE_URL is not set; request rows are kept in memory only")
            if not baseline_model.priced:
                logger.warning(
                    "baseline model has no prices; cost savings will be recorded as unknown",
                    extra={"baseline_model": baseline_model.id},
                )
            try:
                yield
            finally:
                await recorder.drain()
                await store.close()
                await cache.close()

    app = FastAPI(title="Local SLM Router API", version="0.4.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = f"req_{uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    bearer = HTTPBearer(auto_error=False, description="The router's ROUTER_API_KEY")

    async def require_api_key(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        expected = settings.router_api_key
        if expected is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "the server has no ROUTER_API_KEY; set it in .env and restart the API",
            )
        given = credentials.credentials if credentials else ""
        if not secrets.compare_digest(given.encode(), expected.get_secret_value().encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing or invalid API key; send Authorization: Bearer <ROUTER_API_KEY>",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def fail(
        request: Request, code: int, message: str, headers: dict[str, str] | None = None
    ) -> NoReturn:
        logger.warning(
            "request_failed",
            extra={
                "request_id": request.state.request_id,
                "path": request.url.path,
                "status_code": code,
                "error": message,
            },
        )
        raise HTTPException(code, detail=message, headers=headers)

    def parse_tier(request: Request, tier: str | None) -> Tier | None:
        if tier is None:
            return None
        value = tier.strip().lower()
        if value not in TIERS:
            fail(request, 400, f"X-Router-Tier must be 'local' or 'premium', got {tier!r}")
        return cast(Tier, value)

    def run_router(
        request: Request, body: ChatCompletionRequest, forced: Tier | None
    ) -> RouteResult:
        try:
            result = router.route(body, forced)
        except RoutingError as exc:
            fail(request, exc.status_code, exc.message)
        meta = result.metadata()
        logger.info(
            "routed",
            extra={
                "request_id": request.state.request_id,
                "path": request.url.path,
                "tier": meta.tier,
                "model_id": meta.model_id,
                "reason": meta.reason,
                "task_type": meta.task_type,
                "score": meta.complexity_score,
                "signals": meta.signals,
                "pii_kinds": meta.pii_kinds,
            },
        )
        return result

    def check_input_size(request: Request, body: ChatCompletionRequest) -> None:
        size = input_chars([m.content or "" for m in body.messages])
        if size > guardrails.max_input_chars:
            fail(
                request,
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"input is {size} characters; the limit is {guardrails.max_input_chars} "
                "(routing.yaml guardrails.max_input_chars)",
            )

    def client_label(request: Request, value: str | None) -> str:
        if value is None:
            return DEFAULT_CLIENT
        if not _CLIENT_LABEL.match(value):
            fail(request, 400, "X-Router-Client must be 1-64 letters, digits, '.', '_' or '-'")
        return value

    async def storage_status() -> StorageStatus:
        try:
            await store.ping()
        except StorageError as exc:
            return StorageStatus(backend=store.backend, reachable=False, error=str(exc))
        return StorageStatus(backend=store.backend, reachable=True)

    async def cache_status() -> CacheStatus:
        if not cache.enabled:
            return CacheStatus(backend="disabled", reachable=None)
        try:
            await cache.ping()
        except CacheError as exc:
            return CacheStatus(
                backend=cache.status_name, reachable=False, ttl_s=cache_config.ttl_s, error=str(exc)
            )
        return CacheStatus(backend=cache.status_name, reachable=True, ttl_s=cache_config.ttl_s)

    @app.get("/health")
    async def health(request: Request) -> HealthResponse:
        """Liveness of the API, plus whether Ollama and the analytics store answer.

        No auth, and no secrets: the premium block names missing settings, not values.
        """
        local = config.default_local_model
        expected = [m.model for m in config.models.local if m.model]
        provider = request.app.state.providers.for_model(local)
        try:
            pulled = await provider.list_models(config.routing.timeouts_s.health)
        except (ProviderUnavailable, ProviderError) as exc:
            logger.warning("ollama_unhealthy", extra={"error": str(exc)})
            ollama_status = OllamaStatus(
                reachable=False,
                base_url=local.base_url,
                models_available=[],
                models_missing=expected,
                error=str(exc),
            )
        else:
            ollama_status = OllamaStatus(
                reachable=True,
                base_url=local.base_url,
                models_available=[m for m in expected if m in pulled],
                models_missing=[m for m in expected if m not in pulled],
            )
        storage = await storage_status()
        cache_health = await cache_status()
        healthy = (
            ollama_status.reachable
            and not ollama_status.models_missing
            and storage.reachable
            and cache_health.reachable is not False
        )
        return HealthResponse(
            status="ok" if healthy else "degraded",
            ollama=ollama_status,
            premium=PremiumStatus(
                model_id=premium_model.id,
                provider=premium_model.provider,
                configured=not premium_model.missing_settings,
                missing=premium_model.missing_settings,
            ),
            storage=storage,
            cache=cache_health,
            auth_configured=settings.router_api_key is not None,
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Prometheus exposition format. No auth, like /health; bind the port to localhost."""
        return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/v1/route", dependencies=[Depends(require_api_key)])
    async def route(
        body: ChatCompletionRequest, request: Request, response: Response, tier: TierHeader = None
    ) -> RouteResponse:
        """Dry run: the routing decision and its full breakdown. No model is called."""
        check_input_size(request, body)
        result = run_router(request, body, parse_tier(request, tier))
        meta = result.metadata()
        response.headers.update(router_headers(meta))
        return RouteResponse(
            id=request.state.request_id,
            router=meta,
            explanation=result.explanation(),
            cost_estimate=CostEstimate(
                input_tokens_estimate=result.input_tokens,
                input_cost_usd=price(result.decision.model, result.input_tokens),
                baseline_input_cost_usd=price(baseline_model, result.input_tokens),
                baseline_model_id=baseline_model.id,
            ),
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        response: Response,
        background: BackgroundTasks,
        tier: TierHeader = None,
        client: ClientHeader = None,
        cache_control: CacheControlHeader = None,
    ) -> ChatCompletionResponse:
        """OpenAI-compatible chat. `model: "auto"` routes; a configured model ID skips it.

        Identical requests are answered from the exact-match cache when it is on."""
        started = time.perf_counter()
        if body.stream:
            fail(request, status.HTTP_400_BAD_REQUEST, "streaming is not supported")
        label = client_label(request, client)
        check_input_size(request, body)
        forced = parse_tier(request, tier)
        result = run_router(request, body, forced)
        request_id = request.state.request_id
        record_args = {
            "request_id": request_id,
            "client": label,
            "body": body,
            "result": result,
            "scanner": router.pii,
            "preview_chars": guardrails.prompt_preview_chars,
        }

        # Exact-match cache: after routing (so the key names the model that would
        # answer) and before any model call.
        key: str | None = None
        if cache.applies_to(result):
            key = cache_key(
                body, result, forced, config.routing.privacy_mode, cache_config.key_prefix
            )
            directives = {d.strip().lower() for d in (cache_control or "").split(",")}
            if directives & {"no-cache", "no-store"}:
                cache.bypassed()
                response.headers["X-Router-Cache"] = "bypass"
                if "no-store" in directives:
                    key = None
            else:
                _, cached = await cache.lookup(key)
                if cached is not None:
                    served = serve_cached(request, response, result, cached, record_args, started)
                    if served is not None:
                        return served
                response.headers["X-Router-Cache"] = "miss"

        payload = body.model_dump(exclude_none=True, exclude={"model", "stream"})
        payload["stream"] = False
        try:
            answer = await request.app.state.dispatcher.answer(payload, result)
        except AnswerFailed as exc:
            metrics.requests.labels(
                exc.model.tier, exc.model.id, result.decision.reason, "error"
            ).inc()
            recorder.submit(request_record(**record_args, failure=exc))
            fail(request, exc.status_code, exc.message, router_headers(result.metadata()))

        usage = answer.response.usage
        if usage is not None:
            input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
        else:
            # Some providers omit usage; estimate rather than record zero.
            input_tokens = result.input_tokens
            text = "".join(c.message.content or "" for c in answer.response.choices)
            output_tokens = router.tokens.count(text)
        costs = compute_costs(answer.model, baseline_model, input_tokens, output_tokens)
        meta = answered_metadata(result, answer, costs)

        model_id = answer.model.id
        metrics.requests.labels(meta.tier, model_id, meta.reason, "ok").inc()
        metrics.latency.labels(meta.tier, model_id).observe(answer.latency_ms / 1000)
        metrics.tokens.labels(model_id, "input").inc(input_tokens)
        metrics.tokens.labels(model_id, "output").inc(output_tokens)
        if costs.cost_usd:
            metrics.cost.inc(costs.cost_usd)
        if costs.savings_usd and costs.savings_usd > 0:
            metrics.savings.inc(costs.savings_usd)
        if answer.fallback_from is not None:
            metrics.fallbacks.labels(answer.fallback_from, answer.fallback_cause).inc()
        recorder.submit(
            request_record(
                **record_args,
                meta=meta,
                answer=answer,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        logger.info(
            "chat_completed",
            extra={
                "request_id": request_id,
                "client": label,
                "model_id": model_id,
                "model": answer.model.model,
                "tier": meta.tier,
                "latency_ms": answer.latency_ms,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "cost_usd": costs.cost_usd,
                "savings_usd": costs.savings_usd,
                "fallback_from": answer.fallback_from,
            },
        )
        if key is not None:
            entry = CachedAnswer(
                answer.response, answer.model.id, input_tokens, output_tokens, time.time()
            )
            background.add_task(cache.store, key, entry)
        response.headers.update(router_headers(meta))
        return answer.response.model_copy(
            update={"id": request_id, "model": answer.model.model, "router": meta}
        )

    def serve_cached(
        request: Request,
        response: Response,
        result: RouteResult,
        cached: CachedAnswer,
        record_args: dict[str, object],
        started: float,
    ) -> ChatCompletionResponse | None:
        """Answer from the cache: no model call, cost 0, logged and stored as a cache hit."""
        model = config.models.get(cached.model_id)
        if model is None:
            return None  # the model was removed from models.yaml since; treat as a miss
        request_id = request.state.request_id
        latency_ms = round((time.perf_counter() - started) * 1000)
        baseline = price(baseline_model, cached.input_tokens, cached.output_tokens)
        costs = Costs(cost_usd=0.0, baseline_cost_usd=baseline, savings_usd=baseline)
        answer = Answer(cached.response, model, latency_ms)
        meta = answered_metadata(result, answer, costs, cache_hit=True)
        meta = meta.model_copy(
            update={
                "tier": model.tier,
                "model_id": model.id,
                "model": model.model,
                "summary": f"{meta.summary}; served from the exact-match cache "
                f"(answered earlier by {model.id})",
            }
        )
        if costs.savings_usd:
            metrics.savings.inc(costs.savings_usd)
        recorder.submit(
            request_record(
                **record_args,
                meta=meta,
                answer=answer,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
            )
        )
        logger.info(
            "cache_hit",
            extra={
                "request_id": request_id,
                "client": record_args["client"],
                "model_id": model.id,
                "tier": model.tier,
                "latency_ms": latency_ms,
                "savings_usd": costs.savings_usd,
                "cached_seconds_ago": round(time.time() - cached.stored_at),
            },
        )
        response.headers.update(router_headers(meta))
        response.headers["X-Router-Cache"] = "hit"
        return cached.response.model_copy(
            update={"id": request_id, "model": model.model, "router": meta}
        )

    @app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
    async def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
        """Thumbs up (+1) or down (-1) for an answer, by its request_id. One per answer."""
        # The answer's row is written in the background; it may still be in flight.
        await recorder.wait_for(body.request_id)
        comment = router.pii.mask(body.comment.strip()) if body.comment else None
        try:
            saved = await store.save_feedback(body.request_id, body.rating, comment or None)
        except StorageError as exc:
            metrics.storage_errors.labels("save_feedback").inc()
            fail(request, status.HTTP_503_SERVICE_UNAVAILABLE, f"feedback not saved: {exc}")
        if saved is None:
            fail(
                request,
                status.HTTP_404_NOT_FOUND,
                f"unknown request_id {body.request_id!r}; only answered chat completions "
                "can be rated (dry runs are not stored)",
            )
        metrics.feedback.labels("up" if body.rating == 1 else "down").inc()
        logger.info(
            "feedback",
            extra={"request_id": body.request_id, "rating": body.rating, "status": saved},
        )
        return FeedbackResponse(request_id=body.request_id, rating=body.rating, status=saved)

    @app.get("/v1/stats", dependencies=[Depends(require_api_key)])
    async def stats(
        request: Request,
        hours: Annotated[float | None, Query(gt=0, description="Only the last N hours")] = None,
    ) -> StatsResponse:
        """Totals for the playground sidebar. Money values are estimates."""
        await recorder.drain()
        since = datetime.now(UTC) - timedelta(hours=hours) if hours else None
        try:
            totals = await store.stats(since)
        except StorageError as exc:
            metrics.storage_errors.labels("stats").inc()
            fail(request, status.HTTP_503_SERVICE_UNAVAILABLE, f"stats unavailable: {exc}")

        def share(count: int) -> float | None:
            return round(count / totals.ok, 4) if totals.ok else None

        savings_pct = None
        if totals.savings_usd is not None and totals.baseline_cost_usd:
            savings_pct = round(100 * totals.savings_usd / totals.baseline_cost_usd, 2)
        votes = totals.feedback_up + totals.feedback_down
        return StatsResponse(
            window_hours=hours,
            storage=store.backend,
            requests=totals.requests,
            ok=totals.ok,
            errors=totals.requests - totals.ok,
            local_share=share(totals.local),
            premium_share=share(totals.premium),
            fallback_share=share(totals.fallbacks),
            cache_hit_share=share(totals.cache_hits),
            cost_usd=totals.cost_usd,
            baseline_cost_usd=totals.baseline_cost_usd,
            savings_usd=totals.savings_usd,
            savings_pct=savings_pct,
            p95_latency_ms=LatencyP95(
                all=totals.p95_ms, local=totals.p95_local_ms, premium=totals.p95_premium_ms
            ),
            feedback=FeedbackStats(
                up=totals.feedback_up,
                down=totals.feedback_down,
                up_rate=round(totals.feedback_up / votes, 4) if votes else None,
            ),
        )

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def models() -> ModelList:
        """Configured models and their tiers, in the OpenAI list shape, plus `auto`."""
        cards = [ModelCard(id="auto", owned_by="slm-router", tier=None)]
        cards += [
            ModelCard(
                id=m.id,
                owned_by=m.provider,
                tier=m.tier,
                model=m.model,
                context_window=m.context_window,
                capabilities=sorted(m.capabilities),
                configured=not m.missing_settings,
                usd_per_1m_input=m.usd_per_1m_input,
                usd_per_1m_output=m.usd_per_1m_output,
                priced_on=m.priced_on,
            )
            for m in config.models.models
        ]
        return ModelList(data=cards)

    return app
