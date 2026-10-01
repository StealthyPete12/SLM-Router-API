"""Table-driven tests for the routing policy: hard-rule order, threshold, model choice."""

from dataclasses import replace

import pytest

from app.config import AppConfig
from app.router.policy import PolicyInputs, RoutingError, decide

BASE = PolicyInputs(
    requested_model="auto",
    forced_tier=None,
    task_type="qa",
    score=10,
    required_capabilities=frozenset({"chat"}),
    input_tokens=50,
    output_reserve=1024,
    pii_detected=False,
)
TOOLS = frozenset({"chat", "tools"})
TOO_BIG_FOR_LOCAL = 8000  # + 1024 reply > 8192, well under the premium window


def with_routing(config: AppConfig, **changes) -> AppConfig:
    return config.model_copy(update={"routing": config.routing.model_copy(update=changes)})


@pytest.mark.parametrize(
    ("changes", "tier", "model_id", "reason"),
    [
        # Threshold boundary (30).
        ({"score": 29}, "local", "phi3-mini", "score_below_threshold"),
        ({"score": 30}, "premium", "premium-default", "score_at_or_above_threshold"),
        ({"score": 0, "task_type": "chitchat"}, "local", "phi3-mini", "score_below_threshold"),
        # Model choice inside the local tier.
        ({"task_type": "summarise", "score": 20}, "local", "mistral-7b", "score_below_threshold"),
        ({"task_type": "translate", "score": 15}, "local", "mistral-7b", "score_below_threshold"),
        ({"task_type": "rewrite", "score": 10}, "local", "mistral-7b", "score_below_threshold"),
        ({"task_type": "creative", "score": 20}, "local", "llama3-8b", "score_below_threshold"),
        ({"task_type": "extract", "score": 15}, "local", "llama3-8b", "score_below_threshold"),
        # Caller override.
        ({"score": 45, "forced_tier": "local"}, "local", "phi3-mini", "caller_override"),
        ({"score": 0, "forced_tier": "premium"}, "premium", "premium-default", "caller_override"),
        # Explicit model skips selection, whatever the score.
        ({"score": 90, "requested_model": "mistral-7b"}, "local", "mistral-7b", "explicit_model"),
        # Context fit beats the score and the override.
        (
            {"input_tokens": TOO_BIG_FOR_LOCAL},
            "premium",
            "premium-default",
            "context_window_exceeded",
        ),
        (
            {"input_tokens": TOO_BIG_FOR_LOCAL, "forced_tier": "local"},
            "premium",
            "premium-default",
            "context_window_exceeded",
        ),
        (
            {"input_tokens": 7000, "output_reserve": 2000},
            "premium",
            "premium-default",
            "context_window_exceeded",
        ),
        # Capability beats the score and the override.
        ({"required_capabilities": TOOLS}, "premium", "premium-default", "capability_required"),
        (
            {"required_capabilities": TOOLS, "forced_tier": "local"},
            "premium",
            "premium-default",
            "capability_required",
        ),
        # Without privacy_mode, detected PII changes nothing.
        (
            {"pii_detected": True, "score": 45},
            "premium",
            "premium-default",
            "score_at_or_above_threshold",
        ),
    ],
)
def test_decisions(config: AppConfig, changes, tier, model_id, reason):
    decision = decide(replace(BASE, **changes), config)

    assert (decision.tier, decision.model.id, decision.reason) == (tier, model_id, reason)


def test_rules_run_in_documented_order(config: AppConfig):
    decision = decide(BASE, config)

    assert [r.rule for r in decision.rules] == [
        "privacy",
        "override",
        "capability",
        "context_window",
        "threshold",
    ]


@pytest.mark.parametrize("forced", [None, "premium"])
def test_privacy_beats_score_and_forced_premium(config: AppConfig, forced):
    private = with_routing(config, privacy_mode=True)

    decision = decide(replace(BASE, score=80, forced_tier=forced, pii_detected=True), private)

    assert (decision.tier, decision.reason) == ("local", "privacy_pii")
    assert decision.rules[0].outcome == "applied"


def test_privacy_never_falls_back_to_cloud_when_local_cannot_fit(config: AppConfig):
    private = with_routing(config, privacy_mode=True)
    inputs = replace(BASE, pii_detected=True, input_tokens=TOO_BIG_FOR_LOCAL)

    with pytest.raises(RoutingError) as error:
        decide(inputs, private)

    assert error.value.status_code == 413
    assert "privacy_pii" in error.value.message


def test_privacy_blocks_explicit_premium_model(config: AppConfig):
    private = with_routing(config, privacy_mode=True)
    inputs = replace(BASE, requested_model="premium-default", pii_detected=True)

    with pytest.raises(RoutingError, match="privacy_pii") as error:
        decide(inputs, private)
    assert error.value.status_code == 422


def test_override_header_can_be_disabled(config: AppConfig):
    locked = with_routing(
        config, overrides=config.routing.overrides.model_copy(update={"allow_tier_header": False})
    )

    decision = decide(replace(BASE, score=45, forced_tier="local"), locked)

    assert (decision.tier, decision.reason) == ("premium", "score_at_or_above_threshold")
    assert decision.rules[1].outcome == "ignored"


@pytest.mark.parametrize(
    ("changes", "status", "message"),
    [
        ({"requested_model": "gpt-4o"}, 404, "unknown model 'gpt-4o'"),
        ({"requested_model": "phi3-mini", "forced_tier": "premium"}, 400, "conflicts"),
        # An explicit model is never swapped silently.
        ({"requested_model": "phi3-mini", "input_tokens": TOO_BIG_FOR_LOCAL}, 413, "phi3-mini"),
        ({"requested_model": "phi3-mini", "required_capabilities": TOOLS}, 422, "capability"),
        ({"input_tokens": 200_000}, 413, "context_window_exceeded"),
    ],
)
def test_unroutable_requests_fail_clearly(config: AppConfig, changes, status, message):
    with pytest.raises(RoutingError) as error:
        decide(replace(BASE, **changes), config)

    assert error.value.status_code == status
    assert message in error.value.message


def test_falls_back_to_another_model_in_the_tier_that_fits(config: AppConfig):
    # Give only llama3-8b a larger window: summarise prefers mistral-7b, which no longer fits.
    models = [
        m.model_copy(update={"context_window": 32768}) if m.id == "llama3-8b" else m
        for m in config.models.models
    ]
    bigger = config.model_copy(
        update={"models": config.models.model_copy(update={"models": models})}
    )

    decision = decide(replace(BASE, task_type="summarise", input_tokens=TOO_BIG_FOR_LOCAL), bigger)

    assert (decision.tier, decision.model.id) == ("local", "llama3-8b")
