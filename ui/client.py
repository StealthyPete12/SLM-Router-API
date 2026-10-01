"""API client and pure helpers for the Streamlit playground.

Nothing here imports Streamlit, so tests/test_playground_client.py covers it
without a browser. ui/playground.py only lays these pieces out.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

DEFAULT_API_URL = "http://localhost:8000"
CLIENT_LABEL = "playground"  # stored as `client` on every request the playground sends
RoutingMode = Literal["auto", "local", "premium"]
EntryKind = Literal["chat", "dry_run", "unanswered"]

REASON_LABELS = {
    "score_below_threshold": "Score below the threshold",
    "score_at_or_above_threshold": "Score at or above the threshold",
    "caller_override": "Tier forced by the caller (X-Router-Tier)",
    "explicit_model": "Model asked for by ID",
    "capability_required": "Only this tier supports a required capability",
    "context_window_exceeded": "Too long for the other tier's context window",
    "privacy_pii": "PII detected; privacy mode keeps it local",
}


class ApiError(Exception):
    """The API could not be reached, or answered with an error."""

    def __init__(self, status_code: int | None, detail: str, request_id: str | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code  # None when the API could not be reached
        self.detail = detail
        self.request_id = request_id

    def __str__(self) -> str:
        where = f"HTTP {self.status_code}" if self.status_code else "connection failed"
        suffix = f" (request {self.request_id})" if self.request_id else ""
        return f"{where}: {self.detail}{suffix}"


def normalise_base_url(url: str) -> str:
    """http://host:8000, without a trailing slash or /v1. Raises ValueError if not http(s)."""
    url = url.strip().rstrip("/")
    url = url.removesuffix("/v1")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"the API URL must start with http:// or https://, got {url!r}")
    return url


def tier_headers(mode: RoutingMode) -> dict[str, str]:
    """Forcing a tier uses the router's supported X-Router-Tier header."""
    if mode == "auto":
        return {}
    if mode not in ("local", "premium"):
        raise ValueError(f"unknown routing mode {mode!r}")
    return {"X-Router-Tier": mode}


def build_payload(messages: list[dict[str, str]], max_tokens: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": "auto", "messages": messages}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return payload


def format_usd(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "$0"
    return f"${value:.6f}" if abs(value) < 0.01 else f"${value:.4f}"


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def format_share(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


@dataclass(frozen=True)
class RoutingCard:
    """Everything the playground shows about one routing decision."""

    request_id: str
    dry_run: bool
    tier: str
    model_id: str
    model: str | None
    task_type: str
    score: int
    threshold: int
    reason: str
    summary: str
    signals: dict[str, int]
    pii_detected: bool = False
    pii_kinds: list[str] = field(default_factory=list)
    latency_ms: int | None = None
    cost_usd: float | None = None
    baseline_cost_usd: float | None = None
    savings_usd: float | None = None
    cache_hit: bool = False
    fallback_from: str | None = None
    fallback_cause: str | None = None
    input_tokens_estimate: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Dry runs only: the hard-rule trace and the price of the input alone.
    rules: list[dict[str, str]] = field(default_factory=list)
    estimated_input_cost_usd: float | None = None
    estimated_baseline_input_cost_usd: float | None = None

    @classmethod
    def _from_router(cls, request_id: str, router: dict[str, Any], **extra: Any) -> "RoutingCard":  # noqa: ANN401
        return cls(
            request_id=request_id,
            tier=router["tier"],
            model_id=router["model_id"],
            model=router.get("model"),
            task_type=router["task_type"],
            score=int(router["complexity_score"]),
            threshold=int(router["threshold"]),
            reason=router["reason"],
            summary=router.get("summary", ""),
            signals={k: int(v) for k, v in (router.get("signals") or {}).items()},
            pii_detected=bool(router.get("pii_detected")),
            pii_kinds=list(router.get("pii_kinds") or []),
            latency_ms=router.get("latency_ms"),
            cost_usd=router.get("cost_usd"),
            baseline_cost_usd=router.get("baseline_cost_usd"),
            savings_usd=router.get("savings_usd"),
            cache_hit=bool(router.get("cache_hit")),
            fallback_from=router.get("fallback_from"),
            fallback_cause=router.get("fallback_cause"),
            input_tokens_estimate=router.get("input_tokens_estimate"),
            **extra,
        )

    @classmethod
    def from_chat(cls, body: dict[str, Any]) -> "RoutingCard":
        """From a POST /v1/chat/completions response."""
        if not isinstance(body.get("router"), dict):
            raise ValueError("the response has no `router` block; is this the SLM router?")
        usage = body.get("usage") or {}
        return cls._from_router(
            body["id"],
            body["router"],
            dry_run=False,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    @classmethod
    def from_route(cls, body: dict[str, Any]) -> "RoutingCard":
        """From a POST /v1/route (dry run) response."""
        if not isinstance(body.get("router"), dict):
            raise ValueError("the response has no `router` block; is this the SLM router?")
        estimate = body.get("cost_estimate") or {}
        explanation = body.get("explanation") or {}
        return cls._from_router(
            body["id"],
            body["router"],
            dry_run=True,
            rules=list(explanation.get("rules") or []),
            estimated_input_cost_usd=estimate.get("input_cost_usd"),
            estimated_baseline_input_cost_usd=estimate.get("baseline_input_cost_usd"),
        )

    @property
    def score_fraction(self) -> float:
        return max(0.0, min(self.score / 100, 1.0))

    @property
    def threshold_fraction(self) -> float:
        return max(0.0, min(self.threshold / 100, 1.0))

    @property
    def reason_label(self) -> str:
        return REASON_LABELS.get(self.reason, self.reason)

    @property
    def signal_rows(self) -> list[tuple[str, int]]:
        """Every signal and its points, largest first; zeros last but still shown."""
        return sorted(self.signals.items(), key=lambda item: -item[1])

    @property
    def flags(self) -> list[str]:
        flags = []
        if self.dry_run:
            flags.append("dry run: no model called")
        if self.fallback_from:
            flags.append(f"fallback from {self.fallback_from} ({self.fallback_cause or 'failed'})")
        if self.cache_hit:
            flags.append("cache hit")
        if self.pii_detected:
            flags.append(f"PII detected: {', '.join(self.pii_kinds) or 'yes'}")
        if self.reason == "caller_override":
            flags.append(f"forced {self.tier}")
        return flags


@dataclass(frozen=True)
class ChatResult:
    content: str
    card: RoutingCard


@dataclass(frozen=True)
class Health:
    state: Literal["ok", "degraded", "unreachable"]
    detail: str
    premium_configured: bool = False
    premium_provider: str | None = None
    auth_configured: bool = False


class RouterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalise_base_url(base_url)
        self._api_key = api_key or None
        self._timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.Client:
        headers = {"X-Router-Client": CLIENT_LABEL}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self._timeout,
            transport=self._transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        try:
            with self._client() as client:
                response = client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(None, f"the API did not answer within {self._timeout:g}s") from exc
        except httpx.TransportError as exc:
            raise ApiError(None, f"cannot reach the API at {self.base_url}") from exc
        request_id = response.headers.get("X-Request-ID")
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(str(item.get("msg", item)) for item in detail)
            raise ApiError(response.status_code, str(detail)[:500], request_id)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                response.status_code, "the API answered with non-JSON", request_id
            ) from exc

    def health(self) -> Health:
        try:
            body = self._request("GET", "/health")
        except ApiError as exc:
            return Health("unreachable", str(exc))
        ollama = body.get("ollama", {})
        storage = body.get("storage", {})
        premium = body.get("premium", {})
        problems = []
        if not ollama.get("reachable"):
            problems.append("Ollama unreachable")
        elif ollama.get("models_missing"):
            problems.append(f"models not pulled: {', '.join(ollama['models_missing'])}")
        if storage and not storage.get("reachable"):
            problems.append("analytics store unreachable")
        if not premium.get("configured"):
            problems.append(f"premium not configured ({', '.join(premium.get('missing', []))})")
        state = "ok" if body.get("status") == "ok" else "degraded"
        return Health(
            state=state,
            detail="; ".join(problems) or "all services answer",
            premium_configured=bool(premium.get("configured")),
            premium_provider=premium.get("provider"),
            auth_configured=bool(body.get("auth_configured")),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        mode: RoutingMode = "auto",
        max_tokens: int | None = None,
    ) -> ChatResult:
        body = self._request(
            "POST",
            "/v1/chat/completions",
            json=build_payload(messages, max_tokens),
            headers=tier_headers(mode),
        )
        choices = body.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content") or ""
        return ChatResult(content=content, card=RoutingCard.from_chat(body))

    def dry_run(
        self,
        messages: list[dict[str, str]],
        mode: RoutingMode = "auto",
        max_tokens: int | None = None,
    ) -> RoutingCard:
        """POST /v1/route: the decision a chat request would get. No model is called."""
        body = self._request(
            "POST",
            "/v1/route",
            json=build_payload(messages, max_tokens),
            headers=tier_headers(mode),
        )
        return RoutingCard.from_route(body)

    def feedback(self, request_id: str, rating: int, comment: str | None = None) -> dict[str, Any]:
        if rating not in (1, -1):
            raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down)")
        payload: dict[str, Any] = {"request_id": request_id, "rating": rating}
        if comment and comment.strip():
            payload["comment"] = comment.strip()
        return self._request("POST", "/v1/feedback", json=payload)

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")


class FeedbackLedger:
    """Remembers which answers were rated, so a double click sends one vote, not two."""

    def __init__(self) -> None:
        self.ratings: dict[str, int] = {}

    def can_rate(self, request_id: str | None) -> bool:
        return bool(request_id) and request_id not in self.ratings

    def record(self, request_id: str, rating: int) -> None:
        self.ratings[request_id] = rating


def conversation_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The turns to send to the API: answered exchanges only.

    Dry runs and failed prompts stay on screen but are not part of the
    conversation the model sees, since they never got an answer.
    """
    return [
        {"role": entry["role"], "content": entry["content"]}
        for entry in history
        if entry.get("kind", "chat") == "chat"
    ]
