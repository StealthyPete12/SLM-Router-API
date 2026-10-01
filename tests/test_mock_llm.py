"""The development mock model server: deterministic, OpenAI-shaped, clearly labelled."""

from fastapi.testclient import TestClient

from app.devtools.mock_llm import create_app


def client(monkeypatch, tier="local"):
    monkeypatch.setenv("MOCK_TIER", tier)
    monkeypatch.setenv("MOCK_MODELS", "phi3:mini,mistral:7b")
    monkeypatch.setenv("MOCK_DELAY_MS", "0")
    monkeypatch.setenv("MOCK_MS_PER_TOKEN", "0")
    return TestClient(create_app())


def test_answers_are_labelled_deterministic_and_have_usage(monkeypatch):
    mock = client(monkeypatch)
    body = {"model": "phi3:mini", "messages": [{"role": "user", "content": "Hi!"}]}

    first = mock.post("/v1/chat/completions", json=body).json()
    second = mock.post("/v1/chat/completions", json=body).json()

    content = first["choices"][0]["message"]["content"]
    assert content.startswith("[Simulated local answer from phi3:mini; mock mode")
    assert content == second["choices"][0]["message"]["content"]
    assert first["usage"]["completion_tokens"] > 0


def test_unknown_model_is_404_and_models_are_listed(monkeypatch):
    mock = client(monkeypatch)

    assert mock.post("/v1/chat/completions", json={"model": "x", "messages": []}).status_code == 404
    ids = [m["id"] for m in mock.get("/v1/models").json()["data"]]
    assert ids == ["phi3:mini", "mistral:7b"]


def test_max_tokens_caps_the_answer(monkeypatch):
    mock = client(monkeypatch, tier="premium")
    body = {"model": "mistral:7b", "messages": [{"role": "user", "content": "x"}], "max_tokens": 4}

    words = mock.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]

    assert len(words.split("] ", 1)[1].split()) <= 3
