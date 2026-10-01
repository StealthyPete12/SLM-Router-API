"""Local-to-premium fallback, the premium retry, and privacy routing with real PII detection."""

import httpx
import pytest

from tests.conftest import PREMIUM_MODEL
from tests.fakes import Upstreams, connect_error, reply, timeout_error

HELLO = [{"role": "user", "content": "Hi! What can you do?"}]
COMPLEX = [
    {"role": "user", "content": "Compare Postgres and MongoDB for event sourcing, step by step."}
]
PII_REWRITE = [
    {
        "role": "user",
        "content": "Rewrite this note to jane@example.com so it sounds friendlier: send it now",
    }
]


def chat(client, messages=HELLO, **headers):
    return client.post(
        "/v1/chat/completions", json={"model": "auto", "messages": messages}, headers=headers
    )


def with_privacy(config):
    return config.model_copy(
        update={"routing": config.routing.model_copy(update={"privacy_mode": True})}
    )


@pytest.mark.parametrize(
    ("failure", "cause"),
    [
        (connect_error(), "unreachable"),
        (timeout_error(), "timeout"),
        (httpx.Response(500, text="boom"), "error"),
        (reply(""), "empty_answer"),
    ],
)
def test_local_failure_falls_back_to_premium(make_client, failure, cause):
    upstreams = Upstreams(local=[failure])

    response = chat(make_client(upstreams))

    assert response.status_code == 200
    assert upstreams.tiers_called == ["local", "premium"]
    body = response.json()
    assert body["model"] == PREMIUM_MODEL
    router = body["router"]
    assert router["tier"] == "premium"
    assert router["model_id"] == "premium-default"
    assert router["reason"] == "score_below_threshold"  # the decision is unchanged
    assert router["fallback_from"] == "phi3-mini"
    assert router["fallback_cause"] == cause
    assert "phi3-mini failed" in router["summary"]
    assert response.headers["X-Router-Tier"] == "premium"
    assert response.headers["X-Router-Fallback-From"] == "phi3-mini"


def test_no_fallback_when_local_is_forced(make_client):
    upstreams = Upstreams(local=[connect_error()])

    response = chat(make_client(upstreams), COMPLEX, **{"X-Router-Tier": "local"})

    assert response.status_code == 502
    assert upstreams.tiers_called == ["local"]
    assert "forced X-Router-Tier: local" in response.json()["detail"]


def test_no_fallback_for_an_explicit_model(make_client):
    upstreams = Upstreams(local=[timeout_error()])

    response = make_client(upstreams).post(
        "/v1/chat/completions", json={"model": "phi3-mini", "messages": HELLO}
    )

    assert response.status_code == 504
    assert upstreams.tiers_called == ["local"]
    assert "never swapped" in response.json()["detail"]


def test_privacy_kept_prompt_never_reaches_the_cloud(make_client, config):
    upstreams = Upstreams(local=[connect_error()])
    client = make_client(upstreams, app_config=with_privacy(config))

    response = chat(client, PII_REWRITE)

    assert response.status_code == 502
    assert upstreams.tiers_called == ["local"]
    assert "privacy_mode" in response.json()["detail"]


def test_no_fallback_when_premium_is_not_configured(make_client, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")
    upstreams = Upstreams(local=[connect_error()])

    response = chat(make_client(upstreams))

    assert response.status_code == 502
    assert upstreams.tiers_called == ["local"]
    assert "PREMIUM_API_KEY" in response.json()["detail"]


def test_both_tiers_failing_reports_both(make_client):
    upstreams = Upstreams(local=[connect_error()], premium=[httpx.Response(503, text="down")])

    response = chat(make_client(upstreams))

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "local phi3-mini failed" in detail
    assert "premium fallback premium-default also failed" in detail
    # One fallback call plus one premium retry.
    assert upstreams.tiers_called == ["local", "premium", "premium"]


def test_premium_is_retried_once_after_a_server_error(make_client, _no_retry_wait):
    upstreams = Upstreams(
        premium=[httpx.Response(500, text="overloaded"), reply("ok", PREMIUM_MODEL)]
    )

    response = chat(make_client(upstreams), COMPLEX)

    assert response.status_code == 200
    assert upstreams.tiers_called == ["premium", "premium"]
    assert _no_retry_wait == [1.0]
    assert response.json()["router"]["fallback_from"] is None


def test_premium_auth_error_is_not_retried(make_client, _no_retry_wait):
    upstreams = Upstreams(premium=[httpx.Response(401, text="bad key")])

    response = chat(make_client(upstreams), COMPLEX)

    assert response.status_code == 502
    assert upstreams.tiers_called == ["premium"]
    assert _no_retry_wait == []


def test_pii_is_reported_in_the_dry_run(make_client):
    response = make_client(Upstreams()).post(
        "/v1/route", json={"model": "auto", "messages": PII_REWRITE}
    )

    router = response.json()["router"]
    assert router["pii_detected"] is True
    assert router["pii_kinds"] == ["email"]
    assert router["tier"] == "local"  # rewrite scores 10; privacy_mode is off


def test_privacy_mode_keeps_pii_local_even_when_premium_is_forced(make_client, config):
    upstreams = Upstreams()
    client = make_client(upstreams, app_config=with_privacy(config))

    response = chat(client, PII_REWRITE, **{"X-Router-Tier": "premium"})

    assert response.status_code == 200
    router = response.json()["router"]
    assert (router["tier"], router["model_id"], router["reason"]) == (
        "local",
        "mistral-7b",
        "privacy_pii",
    )
    assert upstreams.tiers_called == ["local"]
