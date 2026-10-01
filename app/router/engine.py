"""Runs the whole routing pipeline for one request; shared by /v1/route and chat.

estimate tokens -> classify -> score -> PII scan -> policy -> metadata
No provider is called here, so a dry run is free.
"""

from dataclasses import dataclass

from app.aggregator.guardrails import PiiScanner
from app.config import AppConfig, Capability, Tier
from app.router.classifier import Classification, classify
from app.router.policy import Decision, PolicyInputs, decide
from app.router.privacy import PiiDetector
from app.router.scoring import Score, ScoreInputs, score_prompt
from app.router.tokens import TokenEstimator
from app.schemas import (
    ChatCompletionRequest,
    ClassificationExplanation,
    EvidenceItem,
    RouteExplanation,
    RouterMetadata,
    RuleResult,
)

_JSON_FORMATS = {"json_object", "json_schema"}


@dataclass(frozen=True)
class RouteResult:
    decision: Decision
    classification: Classification
    score: Score
    input_tokens: int
    output_reserve: int
    required_capabilities: frozenset[Capability]
    threshold: int
    pii_kinds: tuple[str, ...] = ()

    @property
    def pii_detected(self) -> bool:
        return bool(self.pii_kinds)

    def metadata(self, latency_ms: int | None = None) -> RouterMetadata:
        model = self.decision.model
        return RouterMetadata(
            tier=self.decision.tier,
            reason=self.decision.reason,
            task_type=self.classification.task_type,
            complexity_score=self.score.total,
            threshold=self.threshold,
            signals=self.score.signals,
            model_id=model.id,
            model=model.model,
            input_tokens_estimate=self.input_tokens,
            summary=self.summary(),
            latency_ms=latency_ms,
            pii_detected=self.pii_detected,
            pii_kinds=list(self.pii_kinds),
        )

    def explanation(self) -> RouteExplanation:
        c = self.classification
        return RouteExplanation(
            classification=ClassificationExplanation(
                task_type=c.task_type,
                decided_by=c.decided_by,
                totals=c.totals,
                evidence=[EvidenceItem(**vars(e)) for e in c.evidence],
            ),
            signal_details=self.score.details,
            rules=[RuleResult(**vars(r)) for r in self.decision.rules],
            required_capabilities=sorted(self.required_capabilities),
            output_reserve_tokens=self.output_reserve,
            model_ready=not self.decision.model.missing_settings,
            model_missing_settings=self.decision.model.missing_settings,
        )

    def summary(self) -> str:
        """One sentence a demo viewer can read: where it went, and why."""
        d, s = self.decision, self.score
        breakdown = " + ".join(f"{name} {pts}" for name, pts in s.signals.items() if pts)
        scored = f"score {s.total} ({breakdown or 'no points'}, {self.classification.task_type})"
        why = {
            "score_below_threshold": f"{scored} is below the threshold of {self.threshold}",
            "score_at_or_above_threshold": f"{scored} meets the threshold of {self.threshold}",
            "explicit_model": "the caller asked for this model by ID",
            "caller_override": f"X-Router-Tier forced {d.tier}; {scored}",
            "capability_required": (
                f"the request needs {', '.join(sorted(self.required_capabilities))}, "
                "which only this tier supports"
            ),
            "context_window_exceeded": (
                f"~{self.input_tokens + self.output_reserve} tokens do not fit "
                "the other tier's context window"
            ),
            "privacy_pii": "PII was detected and privacy_mode keeps it on this machine",
        }[d.reason]
        return f"{d.tier} -> {d.model.id}: {why}"


class Router:
    def __init__(self, config: AppConfig, pii_detector: PiiDetector | None = None) -> None:
        self._config = config
        self.tokens = TokenEstimator(config.routing.tokens)
        self.pii = PiiScanner(config.routing.pii)
        self._detect_pii = pii_detector or self.pii.detect

    def route(self, body: ChatCompletionRequest, forced_tier: Tier | None) -> RouteResult:
        """Raises policy.RoutingError when the request cannot be routed."""
        routing = self._config.routing
        contents = [m.content or "" for m in body.messages]
        last_user = next((m.content or "" for m in reversed(body.messages) if m.role == "user"), "")

        input_tokens = self.tokens.estimate_messages(contents)
        classification = classify(last_user, routing.classifier)
        score = score_prompt(
            ScoreInputs(
                task_type=classification.task_type,
                text=last_user,
                input_tokens=input_tokens,
                conversation_messages=sum(m.role != "system" for m in body.messages),
                max_tokens=body.max_tokens,
            ),
            routing,
        )
        required: set[Capability] = {"chat"}
        if body.response_format and body.response_format.get("type") in _JSON_FORMATS:
            required.add("json_mode")
        if body.tools:
            required.add("tools")
        output_reserve = body.max_tokens or routing.context_fit.default_output_reserve
        pii_kinds = tuple(self._detect_pii("\n".join(contents)))

        decision = decide(
            PolicyInputs(
                requested_model=body.model,
                forced_tier=forced_tier,
                task_type=classification.task_type,
                score=score.total,
                required_capabilities=frozenset(required),
                input_tokens=input_tokens,
                output_reserve=output_reserve,
                pii_detected=bool(pii_kinds),
            ),
            self._config,
        )
        return RouteResult(
            decision=decision,
            classification=classification,
            score=score,
            input_tokens=input_tokens,
            output_reserve=output_reserve,
            required_capabilities=frozenset(required),
            threshold=routing.threshold,
            pii_kinds=pii_kinds,
        )
