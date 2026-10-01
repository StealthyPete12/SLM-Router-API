"""The Provider interface and the errors every provider raises."""

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(Exception):
    """The provider answered, but with an error status or an unusable body."""

    def __init__(self, provider: str, status_code: int, detail: str) -> None:
        super().__init__(f"{provider} returned HTTP {status_code}: {detail}")
        self.provider = provider
        self.status_code = status_code
        self.detail = detail


class ProviderUnavailable(Exception):
    """The provider could not be reached."""


class ProviderTimeout(ProviderUnavailable):
    """The provider did not answer in time."""


class ProviderNotConfigured(Exception):
    """The model cannot be called until the named settings are provided."""

    def __init__(self, model_id: str, missing: list[str]) -> None:
        super().__init__(
            f"model {model_id!r} is not configured: set {', '.join(missing)} in .env "
            "and restart the API"
        )
        self.missing = missing


class Provider(ABC):
    """Calls one model endpoint. Payloads and responses use the OpenAI chat format."""

    name: str

    @abstractmethod
    async def chat_completion(
        self, model: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Send a non-streaming chat request for `model` and return the JSON response."""

    @abstractmethod
    async def list_models(self, timeout: float) -> set[str]:
        """Return the provider-facing names of the models the endpoint serves."""
