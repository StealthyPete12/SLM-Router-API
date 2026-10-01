"""Routing policy: hard rules first, then the score threshold.

The rules run in this order, each narrowing the set of tiers (and models) still allowed:

1. privacy         PII with privacy_mode on removes the premium tier. Nothing overrides it.
2. override        an explicit model ID, or X-Router-Tier, states the wanted tier.
3. capability      removes models lacking a feature the request needs (e.g. tools).
4. context_window  removes models whose context window cannot hold input plus reply.
5. threshold       if no override, score < threshold wants local, otherwise premium.

The wanted tier is used when still allowed. Otherwise the request goes to the
other tier and the reason names the rule that removed the wanted one, so a
forced route can never bypass privacy, capability or context safety. An
explicit model is never silently swapped: if a rule removes it, routing fails.
"""

from dataclasses import dataclass
from typing import Literal

from app.config import TIERS, AppConfig, Capability, ModelConfig, TaskType, Tier

Reason = Literal[
    "privacy_pii",
    "explicit_model",
    "caller_override",
    "capability_required",
    "context_window_exceeded",
    "score_below_threshold",
    "score_at_or_above_threshold",
]
RuleName = Literal["privacy", "override", "capability", "context_window", "threshold"]
Outcome = Literal["pass", "applied", "ignored"]


@dataclass(frozen=True)
class PolicyInputs:
    requested_model: str  # "auto" or a model ID from models.yaml
    forced_tier: Tier | None  # from X-Router-Tier
    task_type: TaskType
    score: int
    required_capabilities: frozenset[Capability]
    input_tokens: int
    output_reserve: int
    pii_detected: bool


@dataclass(frozen=True)
class RuleCheck:
    rule: RuleName
    outcome: Outcome
    detail: str


@dataclass(frozen=True)
class Decision:
    tier: Tier
    model: ModelConfig
    reason: Reason
    rules: list[RuleCheck]


class RoutingError(Exception):
    """The request cannot be routed; status_code is the HTTP status to answer with."""

    def __init__(self, status_code: int, message: str, rules: list[RuleCheck]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.rules = rules


def _other(tier: Tier) -> Tier:
    return "premium" if tier == "local" else "local"


def tier_candidates(tier: Tier, task: TaskType, config: AppConfig) -> list[ModelConfig]:
    """Models of a tier in preference order: the task's model, the default, the rest."""
    mapping = config.routing.models_for(tier)
    preferred = [mapping.get(task, mapping["default"]), mapping["default"]]
    ordered = [config.models.get(i) for i in preferred] + config.models.in_tier(tier)
    unique: dict[str, ModelConfig] = {}
    for model in ordered:
        if model is not None:
            unique.setdefault(model.id, model)
    return list(unique.values())


def decide(inputs: PolicyInputs, config: AppConfig) -> Decision:
    routing = config.routing
    rules: list[RuleCheck] = []
    explicit = inputs.requested_model != "auto"

    # Candidate models per tier.
    if explicit:
        model = config.models.get(inputs.requested_model)
        if model is None:
            known = ", ".join(["auto", *(m.id for m in config.models.models)])
            raise RoutingError(
                404, f"unknown model {inputs.requested_model!r}; use one of: {known}", rules
            )
        if inputs.forced_tier is not None and inputs.forced_tier != model.tier:
            raise RoutingError(
                400,
                f"X-Router-Tier: {inputs.forced_tier} conflicts with model {model.id!r}, "
                f"which is {model.tier}; send model 'auto' to force a tier",
                rules,
            )
        allowed: dict[Tier, list[ModelConfig]] = {model.tier: [model]}
    else:
        allowed = {tier: tier_candidates(tier, inputs.task_type, config) for tier in TIERS}
    removed_by: dict[Tier, Reason] = {}

    def remove(tier: Tier, reason: Reason) -> None:
        if tier in allowed:
            del allowed[tier]
            removed_by[tier] = reason

    # 1. Privacy.
    if routing.privacy_mode and inputs.pii_detected:
        remove("premium", "privacy_pii")
        rules.append(RuleCheck("privacy", "applied", "PII detected and privacy_mode is on"))
    elif routing.privacy_mode:
        rules.append(RuleCheck("privacy", "pass", "no PII detected"))
    else:
        rules.append(RuleCheck("privacy", "pass", "privacy_mode is off"))

    # 2. Caller override: the wanted tier, if the caller states one.
    wanted: Tier | None = None
    wanted_by: Reason | None = None
    if explicit:
        wanted, wanted_by = model.tier, "explicit_model"
        rules.append(RuleCheck("override", "applied", f"explicit model {model.id!r}"))
    elif inputs.forced_tier is not None and routing.overrides.allow_tier_header:
        wanted, wanted_by = inputs.forced_tier, "caller_override"
        rules.append(RuleCheck("override", "applied", f"X-Router-Tier: {wanted}"))
    elif inputs.forced_tier is not None:
        detail = "X-Router-Tier is disabled in routing.yaml (overrides.allow_tier_header)"
        rules.append(RuleCheck("override", "ignored", detail))
    else:
        rules.append(RuleCheck("override", "pass", "no override requested"))

    # 3. Capability.
    needed = inputs.required_capabilities
    removed: list[str] = []
    for tier in list(allowed):
        allowed[tier] = [m for m in allowed[tier] if needed <= m.capabilities]
        if not allowed[tier]:
            remove(tier, "capability_required")
            removed.append(tier)
    needs = ", ".join(sorted(needed))
    if removed:
        detail = f"needs {needs}; no {' or '.join(removed)} model supports it"
        rules.append(RuleCheck("capability", "applied", detail))
    else:
        rules.append(RuleCheck("capability", "pass", f"needs {needs}"))

    # 4. Context-window fit.
    tokens_needed = inputs.input_tokens + inputs.output_reserve
    removed = []
    for tier in list(allowed):
        allowed[tier] = [m for m in allowed[tier] if m.context_window >= tokens_needed]
        if not allowed[tier]:
            remove(tier, "context_window_exceeded")
            removed.append(tier)
    budget = (
        f"~{inputs.input_tokens} input + {inputs.output_reserve} reply = {tokens_needed} tokens"
    )
    if removed:
        detail = f"{budget}; too large for every {' and '.join(removed)} model"
        rules.append(RuleCheck("context_window", "applied", detail))
    else:
        rules.append(RuleCheck("context_window", "pass", budget))

    # 5. Score threshold.
    score_tier: Tier = "premium" if inputs.score >= routing.threshold else "local"
    comparison = "<" if score_tier == "local" else ">="
    score_text = f"score {inputs.score} {comparison} threshold {routing.threshold}"

    if not allowed:
        rules.append(RuleCheck("threshold", "ignored", f"{score_text}, but no tier is left"))
        causes = ", ".join(f"{tier} removed by {why}" for tier, why in removed_by.items())
        target = f"model {inputs.requested_model!r}" if explicit else "any configured model"
        status = 413 if "context_window_exceeded" in removed_by.values() else 422
        raise RoutingError(status, f"cannot route to {target}: {causes}", rules)

    if wanted is not None:
        rules.append(RuleCheck("threshold", "ignored", f"{score_text}; override decides"))
        target = wanted
    else:
        rules.append(RuleCheck("threshold", "applied", f"{score_text} -> {score_tier}"))
        target = score_tier

    if target in allowed:
        tier = target
        reason: Reason = wanted_by or (
            "score_below_threshold" if tier == "local" else "score_at_or_above_threshold"
        )
    else:
        tier = _other(target)
        reason = removed_by[target]
    return Decision(tier=tier, model=allowed[tier][0], reason=reason, rules=rules)
