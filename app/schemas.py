"""OpenAI Chat Completions request and response shapes, plus the router's own blocks."""

from datetime import date
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from app.config import Capability, TaskType, Tier
from app.router.policy import Outcome, Reason, RuleName


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # None for an assistant turn that only carries tool calls.
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    # Unknown OpenAI fields (user, seed, ...) are accepted and ignored.
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: PositiveInt | None = None
    stream: bool = False
    # Forwarded as-is; they also decide which models are capable of the request.
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_tool_calls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # As in OpenAI responses: tool_calls appears only when the model made some.
        data = handler(self)
        if data.get("tool_calls") is None:
            data.pop("tool_calls", None)
        return data


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    total_tokens: NonNegativeInt


FallbackCause = Literal["timeout", "unreachable", "error", "empty_answer", "bad_response"]


class RouterMetadata(BaseModel):
    """Why the request went where it did. Attached to every routed response.

    On a fallback, tier and model describe the model that answered, `reason` is
    still the routing decision, and `fallback_from` names the local model that failed.
    """

    tier: Tier
    reason: Reason
    task_type: TaskType
    complexity_score: int
    threshold: int
    signals: dict[str, int]
    model_id: str
    model: str | None  # provider-facing name; None until a premium model is configured
    input_tokens_estimate: int
    summary: str
    pii_detected: bool = False
    pii_kinds: list[str] = []
    # The fields below are set on real responses, not on dry runs.
    latency_ms: int | None = None
    # Estimates in USD; None when a price is not configured in models.yaml.
    cost_usd: float | None = None
    baseline_cost_usd: float | None = None
    savings_usd: float | None = None
    # Served from the exact-match cache: no model was called, cost_usd is 0, and tier
    # and model name the model that wrote the cached answer.
    cache_hit: bool = False
    fallback_from: str | None = None
    fallback_cause: FallbackCause | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[Choice] = Field(min_length=1)
    usage: Usage | None = None
    router: RouterMetadata | None = None


class EvidenceItem(BaseModel):
    task: TaskType
    signal: str
    where: str
    points: int


class ClassificationExplanation(BaseModel):
    task_type: TaskType
    decided_by: str
    totals: dict[str, int]
    evidence: list[EvidenceItem]


class RuleResult(BaseModel):
    rule: RuleName
    outcome: Outcome
    detail: str


class RouteExplanation(BaseModel):
    classification: ClassificationExplanation
    signal_details: dict[str, dict[str, Any]]
    rules: list[RuleResult]
    required_capabilities: list[Capability]
    output_reserve_tokens: int
    model_ready: bool
    model_missing_settings: list[str]


class CostEstimate(BaseModel):
    """Dry-run price of the input alone; output cost depends on the answer not yet written."""

    input_tokens_estimate: int
    input_cost_usd: float | None
    baseline_input_cost_usd: float | None
    baseline_model_id: str
    note: str = "input tokens only, estimated; no model was called"


class RouteResponse(BaseModel):
    """POST /v1/route: the full decision, with no model called."""

    id: str
    object: Literal["router.decision"] = "router.decision"
    dry_run: Literal[True] = True
    router: RouterMetadata
    explanation: RouteExplanation
    cost_estimate: CostEstimate


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal[1, -1]  # thumbs up or thumbs down
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    object: Literal["router.feedback"] = "router.feedback"
    request_id: str
    rating: Literal[1, -1]
    # One rating per answer: a second submission replaces the first.
    status: Literal["recorded", "updated"]


class LatencyP95(BaseModel):
    all: float | None
    local: float | None
    premium: float | None


class FeedbackStats(BaseModel):
    up: int
    down: int
    up_rate: float | None  # share of thumbs up, 0 to 1; None with no votes


class StatsResponse(BaseModel):
    """GET /v1/stats: totals for the playground sidebar. Money values are estimates."""

    object: Literal["router.stats"] = "router.stats"
    window_hours: float | None
    storage: Literal["postgres", "memory"]
    requests: int
    ok: int
    errors: int
    # Shares of successful requests, 0 to 1; None before the first one.
    local_share: float | None
    premium_share: float | None
    fallback_share: float | None
    cache_hit_share: float | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    savings_usd: float | None
    savings_pct: float | None
    p95_latency_ms: LatencyP95
    feedback: FeedbackStats


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str
    tier: Tier | None  # None for "auto"
    model: str | None = None
    context_window: int | None = None
    capabilities: list[Capability] = []
    configured: bool = True
    usd_per_1m_input: float | None = None
    usd_per_1m_output: float | None = None
    priced_on: date | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OllamaStatus(BaseModel):
    reachable: bool
    base_url: str
    models_available: list[str]
    models_missing: list[str]
    error: str | None = None


class PremiumStatus(BaseModel):
    model_id: str
    provider: str
    configured: bool
    missing: list[str]  # names of settings to set, never their values


class StorageStatus(BaseModel):
    backend: Literal["postgres", "memory"]
    reachable: bool
    error: str | None = None


class CacheStatus(BaseModel):
    backend: Literal["redis", "memory", "disabled"]
    reachable: bool | None  # None when the cache is disabled
    ttl_s: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"] = "ok"
    ollama: OllamaStatus
    premium: PremiumStatus
    storage: StorageStatus
    cache: CacheStatus
    auth_configured: bool
