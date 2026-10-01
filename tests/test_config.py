import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from app.config import DEFAULT_CONFIG_DIR, ConfigError, load_config

REPO_ROOT = DEFAULT_CONFIG_DIR.parent


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the real config directory."""
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def edit_yaml(path: Path, edit: Callable[[dict], object]) -> None:
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data))


def test_repo_config_loads():
    config = load_config(DEFAULT_CONFIG_DIR)

    assert {m.model for m in config.models.local} == {"phi3:mini", "llama3.1:8b", "mistral:7b"}
    assert config.default_local_model.id == "llama3-8b"
    assert config.default_local_model.base_url == "http://ollama:11434/v1"
    assert all(m.context_window == 8192 for m in config.models.local)


def test_base_url_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    config = load_config(DEFAULT_CONFIG_DIR)

    assert config.default_local_model.base_url == "http://host.docker.internal:11434/v1"


def test_context_window_must_match_ollama_context_length():
    load_config(DEFAULT_CONFIG_DIR, ollama_context_length=8192)
    with pytest.raises(ConfigError, match="OLLAMA_CONTEXT_LENGTH is 4096"):
        load_config(DEFAULT_CONFIG_DIR, ollama_context_length=4096)


def test_compose_context_length_matches_models_yaml():
    """docker-compose.yml's default OLLAMA_CONTEXT_LENGTH must equal models.yaml."""
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    defaults = set(re.findall(r"\$\{OLLAMA_CONTEXT_LENGTH:-(\d+)\}", compose))
    config = load_config(DEFAULT_CONFIG_DIR)

    assert defaults == {str(m.context_window) for m in config.models.local}


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path)


def test_invalid_yaml(config_dir):
    (config_dir / "routing.yaml").write_text("local_models: [unclosed")

    with pytest.raises(ConfigError, match=r"routing\.yaml is not valid YAML"):
        load_config(config_dir)


def test_wrong_field_type_names_the_field(config_dir):
    edit_yaml(
        config_dir / "models.yaml",
        lambda d: d["models"][0].update(context_window="big"),
    )

    with pytest.raises(ConfigError, match=r"models\.0\.context_window"):
        load_config(config_dir)


def test_unknown_field_is_rejected(config_dir):
    edit_yaml(config_dir / "routing.yaml", lambda d: d.update(treshold=30))

    with pytest.raises(ConfigError, match="treshold"):
        load_config(config_dir)


def test_duplicate_model_ids(config_dir):
    edit_yaml(config_dir / "models.yaml", lambda d: d["models"].append(d["models"][0]))

    with pytest.raises(ConfigError, match="duplicate model ids"):
        load_config(config_dir)


def test_routing_must_reference_known_models(config_dir):
    edit_yaml(
        config_dir / "routing.yaml",
        lambda d: d["local_models"].update(default="gpt-nothing"),
    )

    with pytest.raises(ConfigError, match="unknown model id 'gpt-nothing'"):
        load_config(config_dir)


def test_routing_requires_default_model(config_dir):
    edit_yaml(config_dir / "routing.yaml", lambda d: d["local_models"].pop("default"))

    with pytest.raises(ConfigError, match="'default'"):
        load_config(config_dir)


def test_unset_env_var_without_default(config_dir):
    edit_yaml(
        config_dir / "models.yaml",
        lambda d: d["models"][0].update(base_url="${NOT_SET_ANYWHERE}/v1"),
    )

    with pytest.raises(ConfigError, match="NOT_SET_ANYWHERE is not set"):
        load_config(config_dir)


def test_premium_model_comes_from_environment(monkeypatch):
    unset = load_config(DEFAULT_CONFIG_DIR).default_model("premium")
    assert unset.model is None
    assert unset.missing_settings == ["model (PREMIUM_MODEL)", "PREMIUM_API_KEY"]

    monkeypatch.setenv("PREMIUM_MODEL", "some-model")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-test")
    premium = load_config(DEFAULT_CONFIG_DIR).default_model("premium")

    assert premium.model == "some-model"
    assert premium.missing_settings == []
    assert premium.api_key_env == "PREMIUM_API_KEY"  # the name lives in YAML, not the key


def test_repo_routing_defaults():
    routing = load_config(DEFAULT_CONFIG_DIR).routing

    assert routing.threshold == 30
    assert routing.max_score == 100
    assert routing.tokens.encoding == "o200k_base"
    assert routing.premium_models["default"] == "premium-default"


@pytest.mark.parametrize(
    ("file", "edit", "message"),
    [
        ("models.yaml", lambda d: d["models"][0].update(tier="cloud"), r"models\.0\.tier"),
        ("models.yaml", lambda d: d.update(baseline="phi3-mini"), "must be a premium model"),
        ("models.yaml", lambda d: d.update(baseline="nope"), "baseline 'nope'"),
        (
            "models.yaml",
            lambda d: d["models"][3].update(usd_per_1m_input=2.5),
            "has prices but no `priced_on`",
        ),
        ("models.yaml", lambda d: d["models"][0].update(model=""), "needs a `model`"),
        ("models.yaml", lambda d: d["models"][3].update(capabilities=["fly"]), "capabilities"),
        ("routing.yaml", lambda d: d["premium_models"].pop("default"), "'default'"),
        (
            "routing.yaml",
            lambda d: d["premium_models"].update(default="phi3-mini"),
            "is not a premium model",
        ),
        (
            "routing.yaml",
            lambda d: d["local_models"].update(code="premium-default"),
            "is not a local model",
        ),
        ("routing.yaml", lambda d: d["task_base"].pop("math"), "missing base points"),
        ("routing.yaml", lambda d: d["task_base"].update(poetry=5), "task_base.poetry"),
        ("routing.yaml", lambda d: d["classifier"]["priority"].pop(), "priority must list"),
        (
            "routing.yaml",
            lambda d: d["classifier"]["tasks"]["math"].update(patterns=["(unclosed"]),
            "invalid regular expression",
        ),
        (
            "routing.yaml",
            lambda d: d["modifiers"]["input_length"].update(points=[0, 5]),
            "one more entry than bands",
        ),
        ("routing.yaml", lambda d: d.update(threshold=101), "threshold"),
    ],
)
def test_invalid_config_is_rejected(config_dir, file, edit, message):
    edit_yaml(config_dir / file, edit)

    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)
