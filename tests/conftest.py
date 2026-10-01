from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.aggregator import dispatch
from app.config import DEFAULT_CONFIG_DIR, AppConfig, Settings, load_config
from app.main import create_app
from app.storage import MemoryStore, Store

OLLAMA = "http://ollama:11434/v1"
PREMIUM = "https://api.openai.com/v1"  # never reached: every test uses a mock transport
PREMIUM_MODEL = "test-premium-model"
PREMIUM_KEY = "sk-test-not-a-real-key"
API_KEY = "test-router-key"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell (and any real keys) out of the tests."""
    for name in (
        "OLLAMA_BASE_URL",
        "LOCAL_PROVIDER",
        "OLLAMA_CONTEXT_LENGTH",
        "CONFIG_DIR",
        "LOG_LEVEL",
        "ROUTER_API_KEY",
        "PREMIUM_PROVIDER",
        "PREMIUM_BASE_URL",
        "PREMIUM_MODEL",
        "PREMIUM_API_KEY",
        "PREMIUM_USD_PER_1M_INPUT",
        "PREMIUM_USD_PER_1M_OUTPUT",
        "PREMIUM_PRICED_ON",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Premium retries back off with asyncio.sleep; record the delays instead of waiting."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(dispatch, "sleep", fake_sleep)
    return delays


@pytest.fixture
def priced_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test prices for the premium model; not any provider's real price list."""
    monkeypatch.setenv("PREMIUM_USD_PER_1M_INPUT", "2.0")
    monkeypatch.setenv("PREMIUM_USD_PER_1M_OUTPUT", "8.0")
    monkeypatch.setenv("PREMIUM_PRICED_ON", "2026-10-01")


@pytest.fixture
def premium_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured premium model with a fake key. Calls still only reach the mock."""
    monkeypatch.setenv("PREMIUM_MODEL", PREMIUM_MODEL)
    monkeypatch.setenv("PREMIUM_API_KEY", PREMIUM_KEY)


@pytest.fixture
def config(premium_env: None) -> AppConfig:
    return load_config(DEFAULT_CONFIG_DIR)


@pytest.fixture
def make_client(config: AppConfig) -> Iterator[Callable[..., TestClient]]:
    """Build a TestClient whose outgoing HTTP calls are answered by `handler`.

    The client sends the router API key unless `api_key` says otherwise. Request
    rows go to `store` (a fresh MemoryStore by default).
    """
    clients: list[TestClient] = []

    def factory(
        handler: Handler,
        api_key: str | None = API_KEY,
        server_key: str | None = API_KEY,
        app_config: AppConfig | None = None,
        store: Store | None = None,
    ) -> TestClient:
        app = create_app(
            config=app_config or config,
            transport=httpx.MockTransport(handler),
            settings=Settings(router_api_key=server_key),
            store=store or MemoryStore(),
        )
        client = TestClient(app)
        if api_key is not None:
            client.headers["Authorization"] = f"Bearer {api_key}"
        client.__enter__()  # runs the lifespan, which creates the HTTP client
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected outgoing call to {request.url}")
