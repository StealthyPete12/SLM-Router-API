"""Typed loading of config/models.yaml, config/routing.yaml and environment settings.

Any malformed or inconsistent configuration raises ConfigError with a message that
names the file and the offending field, so the API fails at startup, not mid-request.
"""

import os
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# ${VAR} or ${VAR:-default}, as in docker compose.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Configuration is missing, malformed or inconsistent."""


TaskType = Literal[
    "chitchat",
    "qa",
    "rewrite",
    "summarise",
    "translate",
    "extract",
    "creative",
    "code",
    "math",
    "analysis",
]
TASK_TYPES: tuple[TaskType, ...] = get_args(TaskType)

Tier = Literal["local", "premium"]
TIERS: tuple[Tier, ...] = get_args(Tier)

Capability = Literal["chat", "json_mode", "tools"]


class Settings(BaseSettings):
    """Process settings from environment variables. Secrets never live in YAML."""

    config_dir: Path = DEFAULT_CONFIG_DIR
    log_level: str = "INFO"
    # Mirrors the Ollama server setting; when set, local context windows must match it.
    ollama_context_length: PositiveInt | None = None
    # Bearer key clients must send. Unset or empty means every /v1 endpoint answers 503.
    router_api_key: SecretStr | None = None
    # postgresql://user:password@host:5432/db. Unset keeps request rows in memory only
    # (tests and quick local runs); it is a secret because it carries the password.
    database_url: SecretStr | None = None
    # redis://host:6379/0 for the exact-match cache. Unset means no cache (tests and
    # quick runs); a secret because it may carry a password.
    redis_url: SecretStr | None = None

    @field_validator("router_api_key", "database_url", "redis_url", mode="before")
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        # docker compose passes an unset variable as "", which must not become a valid key.
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    tier: Tier
    base_url: str
    # Provider-facing name. None (an empty value) means "not configured yet": the
    # router can still pick the model in a dry run, but calling it fails clearly.
    model: str | None
    context_window: PositiveInt
    # None means "not priced yet": cost (or the baseline, for the baseline model) is
    # then recorded as unknown rather than guessed.
    usd_per_1m_input: NonNegativeFloat | None
    usd_per_1m_output: NonNegativeFloat | None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    priced_on: date | None = None
    capabilities: frozenset[Capability] = frozenset({"chat"})

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"must start with http:// or https://, got {value!r}")
        return value.rstrip("/")

    @field_validator("model", "usd_per_1m_input", "usd_per_1m_output", "priced_on", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        # `${VAR:-}` expands to "" when the variable is unset.
        return None if value == "" else value

    @model_validator(mode="after")
    def _check_tier_rules(self) -> Self:
        if self.tier == "local" and self.model is None:
            raise ValueError(f"local model {self.id!r} needs a `model` (the Ollama tag)")
        priced = self.usd_per_1m_input is not None or self.usd_per_1m_output is not None
        if self.tier == "premium" and priced and self.priced_on is None:
            raise ValueError(f"premium model {self.id!r} has prices but no `priced_on` date")
        return self

    @property
    def api_key(self) -> str | None:
        """The key, read from the environment at call time. Never logged."""
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env) or None

    @property
    def priced(self) -> bool:
        return self.usd_per_1m_input is not None and self.usd_per_1m_output is not None

    @property
    def missing_settings(self) -> list[str]:
        """Environment variables that must be set before this model can be called."""
        missing = []
        if self.model is None:
            missing.append("model (PREMIUM_MODEL)" if self.tier == "premium" else "model")
        if self.api_key_env and self.api_key is None:
            missing.append(self.api_key_env)
        return missing


class ModelsConfig(_Strict):
    baseline: str | None = None
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        ids = [m.id for m in self.models]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate model ids: {duplicates}")
        if self.baseline is not None:
            baseline = self.get(self.baseline)
            if baseline is None:
                raise ValueError(f"baseline {self.baseline!r} is not a configured model id")
            if baseline.tier != "premium":
                raise ValueError(f"baseline {self.baseline!r} must be a premium model")
        local_urls = {m.base_url for m in self.models if m.tier == "local"}
        if len(local_urls) > 1:
            raise ValueError(f"local models must share one Ollama base_url, got {local_urls}")
        return self

    def get(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def local(self) -> list[ModelConfig]:
        return [m for m in self.models if m.tier == "local"]

    def in_tier(self, tier: Tier) -> list[ModelConfig]:
        return [m for m in self.models if m.tier == tier]


class Timeouts(_Strict):
    local: PositiveFloat
    premium: PositiveFloat
    health: PositiveFloat


class TokensConfig(_Strict):
    encoding: str = Field(min_length=1)
    per_message_overhead: NonNegativeInt
    reply_overhead: NonNegativeInt


def _check_regex(value: str) -> str:
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {value!r}: {exc}") from exc
    return value


Regex = Annotated[str, AfterValidator(_check_regex)]


class TaskRules(_Strict):
    keywords: list[str] = []
    patterns: list[Regex] = []


class ClassifierWeights(_Strict):
    instruction: PositiveInt
    body: PositiveInt
    pattern: PositiveInt


class ClassifierConfig(_Strict):
    instruction_max_chars: PositiveInt
    weights: ClassifierWeights
    chitchat_max_words: NonNegativeInt
    fallback_task: TaskType
    priority: list[TaskType]
    code_block_pattern: Regex
    tasks: dict[TaskType, TaskRules]

    @model_validator(mode="after")
    def _check_tasks(self) -> Self:
        if sorted(self.priority) != sorted(TASK_TYPES):
            raise ValueError(f"priority must list every task type exactly once: {TASK_TYPES}")
        missing = sorted(set(TASK_TYPES) - set(self.tasks))
        if missing:
            raise ValueError(f"tasks is missing rules for: {missing}")
        return self


class InputLengthModifier(_Strict):
    bands: list[PositiveInt] = Field(min_length=1)
    points: list[NonNegativeInt]

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.bands != sorted(set(self.bands)):
            raise ValueError("bands must be strictly increasing")
        if len(self.points) != len(self.bands) + 1:
            raise ValueError("points needs exactly one more entry than bands")
        return self


class CappedKeywordModifier(_Strict):
    each: NonNegativeInt
    max: NonNegativeInt
    keywords: list[str] = Field(min_length=1)


class MultiPartModifier(_Strict):
    each: NonNegativeInt
    max: NonNegativeInt
    question_pattern: Regex
    numbered_item_pattern: Regex


class OutputDemandModifier(_Strict):
    points: NonNegativeInt
    keywords: list[str]
    words_requested_pattern: Regex
    min_words_requested: PositiveInt
    min_max_tokens: PositiveInt


class ConversationDepthModifier(_Strict):
    more_than: list[PositiveInt] = Field(min_length=1)
    points: list[NonNegativeInt]

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.more_than != sorted(set(self.more_than)):
            raise ValueError("more_than must be strictly increasing")
        if len(self.points) != len(self.more_than):
            raise ValueError("points needs one entry per more_than threshold")
        return self


class FlatModifier(_Strict):
    points: NonNegativeInt


class Modifiers(_Strict):
    input_length: InputLengthModifier
    reasoning_cues: CappedKeywordModifier
    multi_part: MultiPartModifier
    output_demand: OutputDemandModifier
    conversation_depth: ConversationDepthModifier
    code_in_input: FlatModifier


class Guardrails(_Strict):
    max_input_chars: PositiveInt
    prompt_preview_chars: PositiveInt


class PiiConfig(_Strict):
    # Kind -> regex. Matches are masked as [KIND], in this order.
    patterns: dict[str, Regex] = Field(min_length=1)
    # Kinds whose matches must also pass the Luhn checksum (card numbers).
    luhn_checked: list[str] = []

    @model_validator(mode="after")
    def _check_kinds(self) -> Self:
        unknown = sorted(set(self.luhn_checked) - set(self.patterns))
        if unknown:
            raise ValueError(f"luhn_checked names kinds with no pattern: {unknown}")
        return self


class Fallback(_Strict):
    local_to_premium: bool


class Retry(_Strict):
    premium_attempts: PositiveInt
    backoff_s: NonNegativeFloat


class Overrides(_Strict):
    allow_tier_header: bool


class ContextFit(_Strict):
    default_output_reserve: NonNegativeInt


class CacheConfig(_Strict):
    enabled: bool
    ttl_s: PositiveInt
    key_prefix: str = Field(min_length=1)
    store_pii: bool
    timeout_s: PositiveFloat


class RoutingConfig(_Strict):
    threshold: Annotated[int, Field(ge=0, le=100)]
    max_score: PositiveInt
    tokens: TokensConfig
    task_base: dict[TaskType, NonNegativeInt]
    classifier: ClassifierConfig
    modifiers: Modifiers
    overrides: Overrides
    context_fit: ContextFit
    guardrails: Guardrails
    pii: PiiConfig
    privacy_mode: bool
    local_models: dict[TaskType | Literal["default"], str]
    premium_models: dict[TaskType | Literal["default"], str]
    timeouts_s: Timeouts
    fallback: Fallback
    retry: Retry
    cache: CacheConfig

    @field_validator("local_models", "premium_models")
    @classmethod
    def _require_default(cls, value: dict[str, str]) -> dict[str, str]:
        if "default" not in value:
            raise ValueError("must define a 'default' model id")
        return value

    @field_validator("task_base")
    @classmethod
    def _require_every_task(cls, value: dict[str, int]) -> dict[str, int]:
        missing = sorted(set(TASK_TYPES) - set(value))
        if missing:
            raise ValueError(f"missing base points for: {missing}")
        return value

    @model_validator(mode="after")
    def _check_threshold(self) -> Self:
        if self.threshold > self.max_score:
            raise ValueError(f"threshold {self.threshold} is above max_score {self.max_score}")
        return self

    def models_for(self, tier: Tier) -> dict[str, str]:
        return self.local_models if tier == "local" else self.premium_models


class AppConfig(_Strict):
    models: ModelsConfig
    routing: RoutingConfig

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        for tier in TIERS:
            for task, model_id in self.routing.models_for(tier).items():
                field = f"routing.{tier}_models.{task}"
                model = self.models.get(model_id)
                if model is None:
                    raise ValueError(f"{field}: unknown model id {model_id!r}")
                if model.tier != tier:
                    raise ValueError(f"{field}: {model_id!r} is not a {tier} model")
        return self

    @property
    def default_local_model(self) -> ModelConfig:
        return self.default_model("local")

    def default_model(self, tier: Tier) -> ModelConfig:
        model = self.models.get(self.routing.models_for(tier)["default"])
        assert model is not None  # guaranteed by _check_references
        return model

    @property
    def baseline_model(self) -> ModelConfig:
        """Savings are measured against this model: models.yaml `baseline`, else the
        premium default (never the most expensive flagship, which would inflate them)."""
        if self.models.baseline is not None:
            model = self.models.get(self.models.baseline)
            assert model is not None  # guaranteed by ModelsConfig._check_consistency
            return model
        return self.default_model("premium")


def _expand_env(value: Any) -> Any:  # noqa: ANN401 - walks arbitrary parsed YAML
    """Expand ${VAR} and ${VAR:-default} in every string of parsed YAML."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name) or default
        if resolved is None:
            raise ConfigError(f"environment variable {name} is not set and has no default")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping at the top level")
    try:
        return _expand_env(data)
    except ConfigError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc


def _format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"invalid configuration in {source}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def load_config(
    config_dir: Path = DEFAULT_CONFIG_DIR, ollama_context_length: int | None = None
) -> AppConfig:
    """Load and validate both YAML files; raise ConfigError on any problem."""
    raw_models = _read_yaml(config_dir / "models.yaml")
    raw_routing = _read_yaml(config_dir / "routing.yaml")

    try:
        models = ModelsConfig.model_validate(raw_models)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error("models.yaml", exc)) from exc
    try:
        routing = RoutingConfig.model_validate(raw_routing)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error("routing.yaml", exc)) from exc
    try:
        config = AppConfig(models=models, routing=routing)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error("routing.yaml vs models.yaml", exc)) from exc

    if ollama_context_length is not None:
        mismatched = [
            f"{m.id} ({m.context_window})"
            for m in config.models.local
            if m.context_window != ollama_context_length
        ]
        if mismatched:
            raise ConfigError(
                f"OLLAMA_CONTEXT_LENGTH is {ollama_context_length} but models.yaml sets "
                f"context_window for {', '.join(mismatched)}; make them match"
            )
    return config


if __name__ == "__main__":
    # `python -m app.config` validates the config and prints the Ollama tag of every
    # local model, one per line. `make models` uses it to know what to pull.
    settings = Settings()
    app_config = load_config(settings.config_dir, settings.ollama_context_length)
    for local_model in app_config.models.local:
        print(local_model.model)
