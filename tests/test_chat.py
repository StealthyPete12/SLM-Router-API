import json

import httpx
import pytest

from tests.conftest import OLLAMA, PREMIUM, PREMIUM_KEY, PREMIUM_MODEL, no_network

OLLAMA_REPLY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1790000000,
    "model": "phi3:mini",
    "system_fingerprint": "fp_ollama",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
}

HELLO = [{"role": "user", "content": "Hi! What can you do?"}]


class FakeOllama:
    """Records what the API sent and answers with a canned response."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.response = response or httpx.Response(200, json=OLLAMA_REPLY)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    @property
    def sent(self) -> dict:
        return json.loads(self.requests[-1].content)


def test_forwards_to_ollama_and_returns_openai_shape(make_client):
    fake = FakeOllama()
    client = make_client(fake)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "phi3-mini", "messages": HELLO, "temperature": 0.2, "user": "ignored"},
    )

    assert response.status_code == 200
    assert str(fake.requests[0].url) == f"{OLLAMA}/chat/completions"
    assert fake.sent == {
        "model": "phi3:mini",
        "messages": HELLO,
        "temperature": 0.2,
        "stream": False,
    }
    body = response.json()
    assert body["id"].startswith("req_")
    assert body["id"] == response.headers["X-Request-ID"]
    assert body["object"] == "chat.completion"
    assert body["model"] == "phi3:mini"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello! How can I help?",
    }
    assert body["usage"]["total_tokens"] == 19


def test_auto_routes_chitchat_to_phi3_with_metadata(make_client):
    # Phase 0 sent every "auto" request to llama3-8b; Phase 1 routes it.
    fake = FakeOllama()

    response = make_client(fake).post(
        "/v1/chat/completions", json={"model": "auto", "messages": HELLO}
    )

    assert response.status_code == 200
    assert fake.sent["model"] == "phi3:mini"
    body = response.json()
    assert body["model"] == "phi3:mini"
    assert body["system_fingerprint"] == "fp_ollama"
    router = body["router"]
    assert router["tier"] == "local"
    assert router["reason"] == "score_below_threshold"
    assert router["task_type"] == "chitchat"
    assert router["complexity_score"] == 0
    assert router["threshold"] == 30
    assert router["model_id"] == "phi3-mini"
    assert router["signals"]["task_base"] == 0
    assert router["latency_ms"] >= 0
    assert response.headers["X-Router-Tier"] == "local"
    assert response.headers["X-Router-Model"] == "phi3-mini"
    assert response.headers["X-Router-Score"] == "0"


def test_unknown_model_is_404_without_calling_ollama(make_client):
    fake = FakeOllama()

    response = make_client(fake).post(
        "/v1/chat/completions", json={"model": "gpt-4o", "messages": HELLO}
    )

    assert response.status_code == 404
    assert "phi3-mini" in response.json()["detail"]
    assert fake.requests == []


def test_ollama_gets_no_authorization_header(make_client):
    fake = FakeOllama()

    make_client(fake).post("/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO})

    assert "authorization" not in fake.requests[0].headers


def test_streaming_is_rejected(make_client):
    response = make_client(FakeOllama()).post(
        "/v1/chat/completions", json={"model": "auto", "messages": HELLO, "stream": True}
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "auto", "messages": []},
        {"model": "auto"},
        {"model": "auto", "messages": [{"role": "robot", "content": "hi"}]},
    ],
)
def test_invalid_request_is_422(make_client, payload):
    assert make_client(FakeOllama()).post("/v1/chat/completions", json=payload).status_code == 422


def test_model_not_pulled_is_502_with_hint(make_client):
    fake = FakeOllama(httpx.Response(404, json={"error": {"message": "model not found"}}))

    response = make_client(fake).post(
        "/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO}
    )

    assert response.status_code == 502
    assert "make models" in response.json()["detail"]
    assert response.headers["X-Request-ID"].startswith("req_")


def test_ollama_unreachable_is_502(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    response = make_client(handler).post(
        "/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO}
    )

    assert response.status_code == 502
    assert "cannot reach" in response.json()["detail"]


def test_ollama_timeout_is_504(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    response = make_client(handler).post(
        "/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO}
    )

    assert response.status_code == 504


def test_malformed_ollama_response_is_502(make_client):
    fake = FakeOllama(httpx.Response(200, json={"choices": []}))

    response = make_client(fake).post(
        "/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO}
    )

    assert response.status_code == 502
    assert "unexpected response shape" in response.json()["detail"]


COMPLEX = [
    {"role": "user", "content": "Compare Postgres and MongoDB for event sourcing, step by step."}
]
PREMIUM_REPLY = OLLAMA_REPLY | {"model": PREMIUM_MODEL, "system_fingerprint": "fp_cloud"}


def test_complex_prompt_goes_to_premium_provider(make_client):
    fake = FakeOllama(httpx.Response(200, json=PREMIUM_REPLY))

    response = make_client(fake).post(
        "/v1/chat/completions", json={"model": "auto", "messages": COMPLEX, "max_tokens": 200}
    )

    assert response.status_code == 200
    sent = fake.requests[0]
    assert str(sent.url) == f"{PREMIUM}/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {PREMIUM_KEY}"
    assert fake.sent == {
        "model": PREMIUM_MODEL,
        "messages": COMPLEX,
        "max_tokens": 200,
        "stream": False,
    }
    body = response.json()
    assert body["model"] == PREMIUM_MODEL
    assert body["router"]["tier"] == "premium"
    assert body["router"]["task_type"] == "analysis"
    assert body["router"]["complexity_score"] == 45
    assert body["router"]["signals"]["reasoning_cues"] == 5
    assert response.headers["X-Router-Tier"] == "premium"
    assert response.headers["X-Router-Model"] == "premium-default"
    assert response.headers["X-Router-Score"] == "45"


def test_forced_local_tier_header(make_client):
    fake = FakeOllama()

    response = make_client(fake).post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": COMPLEX},
        headers={"X-Router-Tier": "local"},
    )

    assert response.status_code == 200
    assert str(fake.requests[0].url) == f"{OLLAMA}/chat/completions"
    assert response.json()["router"]["reason"] == "caller_override"


def test_invalid_tier_header_is_400(make_client):
    response = make_client(no_network).post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": HELLO},
        headers={"X-Router-Tier": "cheap"},
    )

    assert response.status_code == 400


def test_tools_need_a_capable_model(make_client):
    fake = FakeOllama(httpx.Response(200, json=PREMIUM_REPLY))
    tools = [{"type": "function", "function": {"name": "now", "parameters": {}}}]

    response = make_client(fake).post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": HELLO, "tools": tools},
        headers={"X-Router-Tier": "local"},
    )

    assert response.status_code == 200
    assert fake.sent["tools"] == tools
    assert response.json()["router"]["reason"] == "capability_required"
    assert response.headers["X-Router-Tier"] == "premium"


def test_premium_not_configured_is_503_without_calling_out(make_client, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")

    response = make_client(no_network).post(
        "/v1/chat/completions", json={"model": "auto", "messages": COMPLEX}
    )

    assert response.status_code == 503
    assert "PREMIUM_API_KEY" in response.json()["detail"]
    assert response.headers["X-Router-Tier"] == "premium"


@pytest.mark.parametrize(
    ("upstream", "status", "detail"),
    [
        (httpx.Response(500, text="overloaded"), 502, "HTTP 500: overloaded"),
        (httpx.Response(429, text="rate limited"), 502, "rate limited"),
        # Auth error bodies can quote part of the key, so they are not echoed.
        (httpx.Response(401, text="Incorrect API key sk-tes***key"), 502, "authentication"),
    ],
)
def test_premium_provider_errors_are_502(make_client, upstream, status, detail):
    response = make_client(FakeOllama(upstream)).post(
        "/v1/chat/completions", json={"model": "auto", "messages": COMPLEX}
    )

    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert "sk-tes" not in response.text
    assert "make models" not in response.json()["detail"]


def test_premium_timeout_is_504(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    response = make_client(handler).post(
        "/v1/chat/completions", json={"model": "premium-default", "messages": HELLO}
    )

    assert response.status_code == 504
