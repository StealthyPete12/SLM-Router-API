"""Request rows, /v1/feedback, /v1/stats, /v1/models, /metrics and storage health."""

import pytest

from app.storage import MemoryStore
from tests.conftest import PREMIUM_MODEL
from tests.fakes import Upstreams, connect_error, reply

HELLO = [{"role": "user", "content": "Hi! What can you do?"}]
COMPLEX = [
    {"role": "user", "content": "Compare Postgres and MongoDB for event sourcing, step by step."}
]


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def client(priced_env, make_client, store):
    return make_client(Upstreams(), store=store)


def chat(client, messages=HELLO, **headers):
    response = client.post(
        "/v1/chat/completions", json={"model": "auto", "messages": messages}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


def stats(client, **params):
    response = client.get("/v1/stats", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_every_answer_leaves_a_row(client, store):
    body = chat(client, **{"X-Router-Client": "playground"})
    stats(client)  # waits for background writes

    row = store.requests[body["id"]]
    assert row.client == "playground"
    assert (row.tier, row.model_id, row.provider, row.status) == (
        "local",
        "phi3-mini",
        "ollama",
        "ok",
    )
    assert (row.input_tokens, row.output_tokens) == (12, 7)
    assert row.latency_ms is not None
    assert row.cost_usd == 0
    # Baseline: (12 * 2.0 + 7 * 8.0) / 1e6
    assert row.baseline_cost_usd == row.savings_usd == 0.00008
    assert row.prompt_preview == "Hi! What can you do?"
    assert len(row.prompt_hash) == 64
    assert row.signals["task_base"] == 0


def test_router_block_carries_costs(client):
    router = chat(client)["router"]

    assert router["cost_usd"] == 0
    assert router["baseline_cost_usd"] == 0.00008
    assert router["savings_usd"] == 0.00008
    assert router["cache_hit"] is False
    assert router["fallback_from"] is None
    assert router["latency_ms"] >= 0


def test_stored_preview_is_masked_and_short(client, store):
    long_note = "Rewrite this for jane@example.com: " + "word " * 100
    body = chat(client, [{"role": "user", "content": long_note}])
    stats(client)

    row = store.requests[body["id"]]
    assert "jane@example.com" not in row.prompt_preview
    assert row.prompt_preview.startswith("Rewrite this for [EMAIL]:")
    assert len(row.prompt_preview) == 200
    assert row.pii_detected is True


def test_failed_request_is_stored_as_an_error(make_client, store, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")
    client = make_client(Upstreams(local=[connect_error()]), store=store)

    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": HELLO})

    assert response.status_code == 502
    stats(client)
    row = store.requests[response.headers["X-Request-ID"]]
    assert (row.status, row.status_code, row.tier) == ("error", 502, "local")
    assert "cannot reach" in row.error


def test_invalid_client_label_is_400(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": HELLO},
        headers={"X-Router-Client": "bad label!"},
    )
    assert response.status_code == 400


def test_oversized_input_is_413_before_routing(make_client, config):
    small = config.model_copy(
        update={
            "routing": config.routing.model_copy(
                update={
                    "guardrails": config.routing.guardrails.model_copy(
                        update={"max_input_chars": 10}
                    )
                }
            )
        }
    )
    client = make_client(Upstreams(), app_config=small)

    for path in ("/v1/chat/completions", "/v1/route"):
        response = client.post(path, json={"model": "auto", "messages": HELLO})
        assert response.status_code == 413
        assert "max_input_chars" in response.json()["detail"]


def test_feedback_is_one_rating_per_answer(client, store):
    request_id = chat(client)["id"]

    first = client.post("/v1/feedback", json={"request_id": request_id, "rating": 1})
    second = client.post(
        "/v1/feedback",
        json={"request_id": request_id, "rating": -1, "comment": "Wrong, mail me at a@example.com"},
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == "recorded"
    assert second.json()["status"] == "updated"
    assert len(store.feedback) == 1
    saved = store.feedback[request_id]
    assert saved.rating == -1
    assert saved.comment == "Wrong, mail me at [EMAIL]"
    assert stats(client)["feedback"] == {"up": 0, "down": 1, "up_rate": 0.0}


def test_feedback_for_unknown_or_dry_run_ids_is_404(client):
    dry_run_id = client.post("/v1/route", json={"model": "auto", "messages": HELLO}).json()["id"]

    for request_id in ("req_does_not_exist", dry_run_id):
        response = client.post("/v1/feedback", json={"request_id": request_id, "rating": 1})
        assert response.status_code == 404
        assert "dry runs are not stored" in response.json()["detail"]


@pytest.mark.parametrize("rating", [0, 2, "up"])
def test_feedback_rating_must_be_plus_or_minus_one(client, rating):
    response = client.post("/v1/feedback", json={"request_id": "req_x", "rating": rating})
    assert response.status_code == 422


def test_stats_before_any_traffic(client):
    body = stats(client)

    assert body["requests"] == 0
    assert body["local_share"] is None
    assert body["savings_pct"] is None
    assert body["p95_latency_ms"] == {"all": None, "local": None, "premium": None}
    assert body["feedback"]["up_rate"] is None
    assert body["storage"] == "memory"


def test_stats_shares_savings_and_feedback(priced_env, make_client, store):
    upstreams = Upstreams(
        local=[reply(), connect_error(), reply()],
        premium=[reply("Premium answer.", PREMIUM_MODEL, (40, 120))],
    )
    client = make_client(upstreams, store=store)

    local_id = chat(client)["id"]  # local
    fallback = chat(client)  # local fails, premium answers
    chat(client, COMPLEX)  # premium
    chat(client)  # local
    client.post("/v1/feedback", json={"request_id": local_id, "rating": 1})
    client.post("/v1/feedback", json={"request_id": fallback["id"], "rating": -1})

    body = stats(client, hours=1)
    assert fallback["router"]["fallback_from"] == "phi3-mini"
    assert (body["requests"], body["ok"], body["errors"]) == (4, 4, 0)
    assert body["local_share"] == 0.5
    assert body["premium_share"] == 0.25
    assert body["fallback_share"] == 0.25
    assert body["cache_hit_share"] == 0
    # Local rows save their baseline; premium rows (incl. the fallback) save nothing.
    local_baseline = (12 * 2.0 + 7 * 8.0) / 1e6
    premium_cost = (40 * 2.0 + 120 * 8.0) / 1e6
    assert body["savings_usd"] == pytest.approx(2 * local_baseline)
    assert body["cost_usd"] == pytest.approx(2 * premium_cost)
    expected_pct = 100 * 2 * local_baseline / (2 * local_baseline + 2 * premium_cost)
    assert body["savings_pct"] == pytest.approx(expected_pct, abs=0.01)
    assert body["p95_latency_ms"]["local"] is not None
    assert body["feedback"] == {"up": 1, "down": 1, "up_rate": 0.5}


def test_stats_and_feedback_need_the_api_key(make_client):
    client = make_client(Upstreams(), api_key=None)

    assert client.get("/v1/stats").status_code == 401
    assert client.post("/v1/feedback", json={"request_id": "x", "rating": 1}).status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_models_lists_auto_and_the_catalog(client):
    body = client.get("/v1/models").json()

    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["auto", "phi3-mini", "llama3-8b", "mistral-7b", "premium-default"]
    premium = body["data"][-1]
    assert premium["tier"] == "premium"
    assert premium["configured"] is True
    assert premium["usd_per_1m_output"] == 8.0
    assert "sk-" not in str(body)


def test_metrics_expose_the_designed_series(client):
    chat(client)
    chat(client, COMPLEX)

    text = client.get("/metrics").text

    assert (
        'router_requests_total{model="phi3-mini",reason="score_below_threshold",'
        'status="ok",tier="local"} 1.0' in text
    )
    assert 'router_latency_seconds_bucket{le="0.05",model="premium-default",tier="premium"}' in text
    assert 'router_tokens_total{direction="input",model="phi3-mini"} 12.0' in text
    assert 'router_tokens_total{direction="output",model="premium-default"} 120.0' in text
    assert "router_savings_usd_total 8e-05" in text


def test_dry_run_estimates_input_cost_only(client):
    body = client.post("/v1/route", json={"model": "auto", "messages": COMPLEX}).json()

    estimate = body["cost_estimate"]
    tokens = estimate["input_tokens_estimate"]
    assert tokens == body["router"]["input_tokens_estimate"]
    assert estimate["input_cost_usd"] == pytest.approx(tokens * 2.0 / 1e6)
    assert estimate["baseline_model_id"] == "premium-default"
    assert body["router"]["cost_usd"] is None  # nothing was spent


def test_health_reports_storage(client):
    body = client.get("/health").json()

    assert body["storage"] == {"backend": "memory", "reachable": True, "error": None}
    assert body["status"] == "ok"
