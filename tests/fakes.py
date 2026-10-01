"""Fake upstreams for API tests: Ollama and the premium provider behind one MockTransport."""

import json
from collections.abc import Callable

import httpx

from tests.conftest import OLLAMA, PREMIUM, PREMIUM_MODEL


def reply(content: str = "Hello! How can I help?", model: str = "phi3:mini", usage=(12, 7)):
    body = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1790000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return httpx.Response(200, json=body)


Behaviour = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class Upstreams:
    """Answers Ollama and premium calls with queued behaviours; the last one repeats."""

    def __init__(
        self, local: list[Behaviour] | None = None, premium: list[Behaviour] | None = None
    ):
        self.local = local or [reply()]
        self.premium = premium or [reply("Premium answer.", PREMIUM_MODEL, (40, 120))]
        self.calls: list[tuple[str, dict]] = []

    @property
    def tiers_called(self) -> list[str]:
        return [tier for tier, _ in self.calls]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(OLLAMA):
            tier, queue = "local", self.local
        elif url.startswith(PREMIUM):
            tier, queue = "premium", self.premium
        else:
            raise AssertionError(f"unexpected call to {url}")
        if url.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "phi3:mini"}, {"id": "llama3.1:8b"}, {"id": "mistral:7b"}]},
            )
        self.calls.append((tier, json.loads(request.content)))
        behaviour = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(behaviour, Exception):
            raise behaviour
        if callable(behaviour) and not isinstance(behaviour, httpx.Response):
            return behaviour(request)
        return behaviour


def connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("connection refused")


def timeout_error() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("too slow")
