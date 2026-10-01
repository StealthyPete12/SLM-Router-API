"""Playground helpers (ui/client.py) against the real API, in-process: no browser, no network."""

import json

import httpx
import pytest

from tests.conftest import API_KEY
from tests.fakes import Upstreams, connect_error
from ui.client import (
    ApiError,
    FeedbackLedger,
    RouterClient,
    RoutingCard,
    build_payload,
    conversation_messages,
    format_ms,
    format_usd,
    normalise_base_url,
    tier_headers,
)

COMPLEX = "Compare Postgres and MongoDB for event sourcing, step by step."


class AppTransport(httpx.BaseTransport):
    """Sends the playground's requests to a FastAPI TestClient and records them."""

    def __init__(self, test_client):
        self.test_client = test_client
        self.sent: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        response = self.test_client.request(
            request.method,
            request.url.path,
            content=request.content,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            params=request.url.params,
        )
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )


@pytest.fixture
def transport(priced_env, make_client):
    return AppTransport(make_client(Upstreams(), api_key=None))


@pytest.fixture
def router_client(transport):
    return RouterClient("http://router.test:8000/v1/", API_KEY, transport=transport)


def user(text):
    return [{"role": "user", "content": text}]


def test_base_url_is_normalised():
    assert normalise_base_url(" http://api:8000/v1/ ") == "http://api:8000"
    with pytest.raises(ValueError, match="http"):
        normalise_base_url("api:8000")


def test_payload_defaults_to_auto():
    assert build_payload(user("hi")) == {"model": "auto", "messages": user("hi")}
    assert build_payload(user("hi"), 128)["max_tokens"] == 128


def test_forced_tiers_use_the_router_header():
    assert tier_headers("auto") == {}
    assert tier_headers("local") == {"X-Router-Tier": "local"}
    assert tier_headers("premium") == {"X-Router-Tier": "premium"}
    with pytest.raises(ValueError):
        tier_headers("cheap")


def test_chat_returns_answer_and_full_routing_card(router_client, transport):
    result = router_client.chat(user("Hi! What can you do?"), max_tokens=64)

    card = result.card
    assert result.content == "Hello! How can I help?"
    assert (card.tier, card.model_id, card.task_type, card.score, card.threshold) == (
        "local",
        "phi3-mini",
        "chitchat",
        0,
        30,
    )
    assert card.request_id.startswith("req_")
    assert card.dry_run is False
    assert card.latency_ms is not None
    assert card.cost_usd == 0
    assert card.savings_usd == card.baseline_cost_usd > 0
    assert card.fallback_from is None and card.cache_hit is False and card.pii_detected is False
    assert dict(card.signal_rows)["task_base"] == 0
    assert (card.input_tokens, card.output_tokens) == (12, 7)
    sent = transport.sent[-1]
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    assert sent.headers["x-router-client"] == "playground"
    assert json.loads(sent.content)["model"] == "auto"


def test_forced_premium_is_honoured(router_client, transport):
    card = router_client.chat(user("Hi!"), mode="premium").card

    assert card.tier == "premium"
    assert card.reason == "caller_override"
    assert "forced premium" in card.flags
    assert transport.sent[-1].headers["x-router-tier"] == "premium"


def test_dry_run_uses_route_and_calls_no_model(make_client):
    upstreams = Upstreams()
    transport = AppTransport(make_client(upstreams, api_key=None))
    client = RouterClient("http://router.test", API_KEY, transport=transport)

    card = client.dry_run(user(COMPLEX), mode="local")

    assert transport.sent[-1].url.path == "/v1/route"
    assert upstreams.calls == []  # no model was called
    assert card.dry_run is True
    assert (card.tier, card.score, card.reason) == ("local", 45, "caller_override")
    assert card.latency_ms is None and card.cost_usd is None
    assert [r["rule"] for r in card.rules][-1] == "threshold"
    assert "dry run: no model called" in card.flags
    assert card.input_tokens_estimate > 0


def test_feedback_uses_the_answer_request_id(router_client, transport):
    request_id = router_client.chat(user("Hi!")).card.request_id

    first = router_client.feedback(request_id, 1, "  great  ")
    second = router_client.feedback(request_id, -1)

    assert json.loads(transport.sent[-2].content) == {
        "request_id": request_id,
        "rating": 1,
        "comment": "great",
    }
    assert (first["status"], second["status"]) == ("recorded", "updated")
    with pytest.raises(ValueError):
        router_client.feedback(request_id, 0)


def test_feedback_for_a_dry_run_is_a_clear_error(router_client):
    card = router_client.dry_run(user("Hi!"))

    with pytest.raises(ApiError) as error:
        router_client.feedback(card.request_id, 1)

    assert error.value.status_code == 404
    assert "dry runs are not stored" in error.value.detail


def test_api_errors_carry_status_detail_and_request_id(make_client, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")
    transport = AppTransport(make_client(Upstreams(local=[connect_error()]), api_key=None))
    client = RouterClient("http://router.test", API_KEY, transport=transport)

    with pytest.raises(ApiError) as error:
        client.chat(user("Hi!"))

    assert error.value.status_code == 502
    assert "no cloud fallback" in error.value.detail
    assert error.value.request_id.startswith("req_")
    assert "HTTP 502" in str(error.value)


def test_wrong_key_is_401(transport):
    client = RouterClient("http://router.test", "wrong", transport=transport)

    with pytest.raises(ApiError) as error:
        client.stats()

    assert error.value.status_code == 401


def test_unreachable_api_is_reported_not_raised_by_health():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = RouterClient("http://nowhere:8000", API_KEY, transport=httpx.MockTransport(refuse))

    health = client.health()
    assert health.state == "unreachable"
    assert "cannot reach the API" in health.detail
    with pytest.raises(ApiError) as error:
        client.chat(user("Hi!"))
    assert error.value.status_code is None


def test_health_summarises_the_stack(router_client):
    health = router_client.health()

    assert health.state == "ok"
    assert health.premium_configured is True
    assert health.auth_configured is True


def test_card_parsing_rejects_non_router_responses():
    with pytest.raises(ValueError, match="router"):
        RoutingCard.from_chat({"id": "x", "choices": []})


def test_card_flags_fallback_and_pii():
    card = RoutingCard.from_chat(
        {
            "id": "req_1",
            "choices": [],
            "router": {
                "tier": "premium",
                "model_id": "premium-default",
                "task_type": "rewrite",
                "complexity_score": 10,
                "threshold": 30,
                "reason": "score_below_threshold",
                "signals": {"task_base": 10, "input_length": 0},
                "pii_detected": True,
                "pii_kinds": ["email"],
                "fallback_from": "mistral-7b",
                "fallback_cause": "timeout",
            },
        }
    )

    assert card.flags == ["fallback from mistral-7b (timeout)", "PII detected: email"]
    assert card.score_fraction == 0.1
    assert card.reason_label == "Score below the threshold"


def test_ledger_prevents_duplicate_votes():
    ledger = FeedbackLedger()

    assert ledger.can_rate("req_1")
    ledger.record("req_1", 1)
    assert not ledger.can_rate("req_1")
    assert not ledger.can_rate(None)


def test_dry_runs_and_failures_stay_out_of_the_conversation():
    history = [
        {"role": "user", "content": "Hi", "kind": "chat"},
        {"role": "assistant", "content": "Hello", "kind": "chat"},
        {"role": "user", "content": "Plan a trip", "kind": "dry_run"},
        {"role": "assistant", "content": "", "kind": "dry_run"},
        {"role": "user", "content": "Oops", "kind": "unanswered"},
        {"role": "assistant", "error": "HTTP 502", "kind": "unanswered"},
    ]

    assert conversation_messages(history) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_formatting():
    assert format_usd(None) == "unknown"
    assert format_usd(0) == "$0"
    assert format_usd(0.000123) == "$0.000123"
    assert format_usd(1.5) == "$1.5000"
    assert format_ms(None) == "n/a"
    assert format_ms(250) == "250 ms"
    assert format_ms(1840) == "1.84 s"
