"""OpenAICompatibleProvider: one client for Ollama and any OpenAI-format cloud API.

Ollama, OpenAI, Gemini and Anthropic's compatibility layer all accept the same
`POST {base_url}/chat/completions` call, so adding a model is a models.yaml entry.
"""

from typing import Any

import httpx

from app.config import ModelConfig
from app.providers.base import (
    Provider,
    ProviderError,
    ProviderNotConfigured,
    ProviderTimeout,
    ProviderUnavailable,
)

# Upstream bodies are echoed in errors to help debugging, but not for auth
# failures, whose bodies can quote part of the key.
_AUTH_STATUSES = {401, 403}
_MAX_DETAIL_CHARS = 500


class OpenAICompatibleProvider(Provider):
    def __init__(
        self, name: str, base_url: str, client: httpx.AsyncClient, api_key: str | None = None
    ) -> None:
        self.name = name
        self.base_url = base_url
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def _request(
        self, method: str, path: str, timeout: float, json: dict[str, Any] | None = None
    ) -> Any:  # noqa: ANN401 - returns parsed JSON
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(
                method, url, timeout=timeout, json=json, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.name} timed out after {timeout:g}s calling {url}"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"cannot reach {self.name} at {url}: {exc!r}") from exc
        if response.is_error:
            if response.status_code in _AUTH_STATUSES:
                detail = "authentication failed; check the API key"
            else:
                detail = response.text[:_MAX_DETAIL_CHARS]
            raise ProviderError(self.name, response.status_code, detail)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(self.name, response.status_code, "body is not JSON") from exc

    async def chat_completion(
        self, model: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        data = await self._request(
            "POST", "/chat/completions", timeout, json=payload | {"model": model}
        )
        if not isinstance(data, dict):
            raise ProviderError(self.name, 200, "chat response is not a JSON object")
        return data

    async def list_models(self, timeout: float) -> set[str]:
        data = await self._request("GET", "/models", timeout)
        try:
            # Ollama sends "data": null when no model has been pulled yet.
            return {item["id"] for item in data["data"] or []}
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.name, 200, "unexpected /models response shape") from exc


class ProviderRegistry:
    """Builds the provider for a configured model, sharing one HTTP client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    def for_model(self, model: ModelConfig) -> OpenAICompatibleProvider:
        """Raise ProviderNotConfigured when the model ID or API key is missing."""
        if model.missing_settings:
            raise ProviderNotConfigured(model.id, model.missing_settings)
        # The key is read from the environment on every call, so it is never cached
        # beyond the request and rotating it needs no restart.
        return OpenAICompatibleProvider(model.provider, model.base_url, self._client, model.api_key)
