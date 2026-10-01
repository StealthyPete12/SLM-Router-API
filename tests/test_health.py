import httpx

from tests.conftest import OLLAMA


def models_response(*tags: str) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": t} for t in tags]})


def test_ok_when_ollama_has_all_models(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{OLLAMA}/models"
        return models_response("phi3:mini", "llama3.1:8b", "mistral:7b", "other:latest")

    response = make_client(handler).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api"] == "ok"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["models_missing"] == []


def test_degraded_when_models_not_pulled(make_client):
    response = make_client(lambda request: models_response("phi3:mini")).get("/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["models_available"] == ["phi3:mini"]
    assert body["ollama"]["models_missing"] == ["llama3.1:8b", "mistral:7b"]


def test_api_stays_up_when_ollama_is_unreachable(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    response = make_client(handler).get("/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["api"] == "ok"
    assert body["ollama"]["reachable"] is False
    assert "connection refused" in body["ollama"]["error"]


def test_fresh_ollama_with_no_models(make_client):
    # A freshly started Ollama answers {"data": null} rather than an empty list.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": None})

    body = make_client(handler).get("/health").json()

    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["models_missing"] == ["phi3:mini", "llama3.1:8b", "mistral:7b"]
