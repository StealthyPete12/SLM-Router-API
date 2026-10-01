"""POST /v1/route (the free dry run) and bearer-key authentication."""

import pytest

from tests.conftest import API_KEY, no_network

COMPLEX = "Compare Postgres and MongoDB for event sourcing, step by step."


def dry_run(client, content, **extra):
    headers = extra.pop("headers", {})
    return client.post(
        "/v1/route",
        json={"model": "auto", "messages": [{"role": "user", "content": content}], **extra},
        headers=headers,
    )


def test_dry_run_explains_without_calling_a_model(make_client):
    response = dry_run(make_client(no_network), COMPLEX)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "router.decision"
    assert body["dry_run"] is True
    assert body["id"] == response.headers["X-Request-ID"]
    router = body["router"]
    assert (router["tier"], router["task_type"], router["complexity_score"]) == (
        "premium",
        "analysis",
        45,
    )
    assert router["signals"] == {
        "task_base": 40,
        "input_length": 0,
        "reasoning_cues": 5,
        "multi_part": 0,
        "output_demand": 0,
        "conversation_depth": 0,
        "code_in_input": 0,
    }
    assert router["latency_ms"] is None
    assert "45" in router["summary"]
    explanation = body["explanation"]
    assert [r["rule"] for r in explanation["rules"]][-1] == "threshold"
    assert {"task": "analysis", "signal": "compare", "where": "instruction", "points": 3} in (
        explanation["classification"]["evidence"]
    )
    assert explanation["model_ready"] is True
    assert response.headers["X-Router-Score"] == "45"


def test_dry_run_reports_unconfigured_premium_model(make_client, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")

    body = dry_run(make_client(no_network), COMPLEX).json()

    assert body["router"]["tier"] == "premium"
    assert body["explanation"]["model_ready"] is False
    assert body["explanation"]["model_missing_settings"] == ["PREMIUM_API_KEY"]


def test_dry_run_counts_conversation_depth_and_max_tokens(make_client):
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(13)
    ]
    body = (
        make_client(no_network)
        .post(
            "/v1/route",
            json={
                "model": "auto",
                "messages": [{"role": "system", "content": "Be brief."}, *turns],
                "max_tokens": 1500,
            },
        )
        .json()
    )

    assert body["router"]["signals"]["conversation_depth"] == 10
    assert body["router"]["signals"]["output_demand"] == 10
    assert body["explanation"]["output_reserve_tokens"] == 1500


@pytest.mark.parametrize(
    ("model", "status"),
    [("gpt-4o", 404), ("mistral-7b", 200)],
)
def test_dry_run_explicit_models(make_client, model, status):
    response = make_client(no_network).post(
        "/v1/route", json={"model": model, "messages": [{"role": "user", "content": COMPLEX}]}
    )

    assert response.status_code == status
    if status == 200:
        assert response.json()["router"]["reason"] == "explicit_model"


@pytest.mark.parametrize("path", ["/v1/route", "/v1/chat/completions"])
@pytest.mark.parametrize("api_key", [None, "wrong-key", ""])
def test_missing_or_wrong_key_is_401(make_client, path, api_key):
    client = make_client(no_network, api_key=None)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}

    response = client.post(
        path,
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["X-Request-ID"].startswith("req_")


@pytest.mark.parametrize("server_key", [None, "", "   "])
def test_server_without_key_fails_closed(make_client, server_key):
    # Compose passes an unset ROUTER_API_KEY as "": that must not mean "no key needed".
    client = make_client(no_network, api_key=None, server_key=server_key)

    response = dry_run(client, "hi", headers={"Authorization": "Bearer "})

    assert response.status_code == 503
    assert "ROUTER_API_KEY" in response.json()["detail"]


def test_health_needs_no_key_and_exposes_no_secrets(make_client, monkeypatch):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    monkeypatch.delenv("PREMIUM_API_KEY")
    response = make_client(handler, api_key=None).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["auth_configured"] is True
    assert body["premium"] == {
        "model_id": "premium-default",
        "provider": "openai",
        "configured": False,
        "missing": ["PREMIUM_API_KEY"],
    }
    assert API_KEY not in response.text
