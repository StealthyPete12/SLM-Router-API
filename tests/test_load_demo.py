"""The demo loader: prompt-file parsing, fixture hygiene, and a full run in-process."""

import io
import json
import re

import pytest

from app.config import TASK_TYPES
from app.router.engine import Router
from app.schemas import ChatCompletionRequest
from app.storage import MemoryStore
from eval.load_demo import (
    CLIENT_LABEL,
    PROMPTS,
    DemoDataError,
    PreflightError,
    load_prompts,
    main,
    preflight,
    run,
)
from tests.fakes import Upstreams, connect_error


def routed(config, prompt):
    request = ChatCompletionRequest(
        model="auto", messages=[{"role": "user", "content": prompt.prompt}], max_tokens=128
    )
    return Router(config).route(request, prompt.tier).metadata()


def test_prompt_file_covers_every_task_type_and_both_tiers(config):
    prompts = load_prompts()
    routes = [routed(config, p) for p in prompts]

    assert {r.task_type for r in routes} == set(TASK_TYPES)
    assert {r.tier for r in routes} == {"local", "premium"}
    assert any(r.reason == "caller_override" and r.tier == "local" for r in routes)
    assert any(r.reason == "caller_override" and r.tier == "premium" for r in routes)
    assert any(r.pii_detected for r in routes)
    assert {p.feedback for p in prompts} == {1, -1, None}


def test_fallback_set_is_local_only(config):
    fallback = load_prompts(sets=("fallback",))

    assert fallback
    assert all(routed(config, p).tier == "local" for p in fallback)


def test_fixtures_hold_no_real_personal_data():
    text = PROMPTS.read_text(encoding="utf-8")

    for email in re.findall(r"[\w.+-]+@([\w.-]+)", text):
        assert email == "example.com"
    for phone in re.findall(r"\+?\d[\d ().-]{8,}\d", text):
        assert "555" in phone, phone


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"id": "a", "set": "main"}', "`prompt` must be"),
        ('{"id": "a", "set": "other", "prompt": "x"}', "`set` must be"),
        ('{"id": "a", "set": "main", "prompt": "x", "tier": "cheap"}', "`tier` must be"),
        ('{"id": "a", "set": "main", "prompt": "x", "feedback": 5}', "`feedback` must be"),
        ('{"id": "a", "set": "main", "prompt": "x", "extra": 1}', "unknown fields"),
    ],
)
def test_malformed_prompt_files_are_rejected(tmp_path, line, message):
    path = tmp_path / "prompts.jsonl"
    path.write_text(line + "\n")

    with pytest.raises(DemoDataError, match=message):
        load_prompts(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = json.dumps({"id": "a", "set": "main", "prompt": "x"})
    path.write_text(f"{row}\n{row}\n")

    with pytest.raises(DemoDataError, match="duplicate id"):
        load_prompts(path)


def demo_client(make_client, upstreams, store):
    client = make_client(upstreams, store=store)
    client.headers["X-Router-Client"] = CLIENT_LABEL
    return client


def test_a_run_fills_every_table_the_dashboard_reads(priced_env, make_client):
    store = MemoryStore()
    client = demo_client(make_client, Upstreams(), store)
    prompts = load_prompts(sets=("main",))
    out = io.StringIO()

    warnings = preflight(client, prompts, fallback_run=False)
    assert warnings == ["the API has no DATABASE_URL: rows stay in its memory, not in Postgres"]
    summary = run(client, prompts, max_tokens=64, out=out)

    assert summary.failures == []
    assert summary.ok == len(prompts)
    assert set(summary.tiers) == {"local", "premium"}
    assert summary.savings_usd > 0
    client.get("/v1/stats")  # waits for background writes
    assert len(store.requests) == len(prompts)
    assert {r.client for r in store.requests.values()} == {CLIENT_LABEL}
    rated = [p for p in prompts if p.feedback is not None]
    assert len(store.feedback) == len(rated)
    assert all("synthetic demo rating" in f.comment for f in store.feedback.values())
    assert "[  1/" in out.getvalue()


def test_fallback_run_records_fallbacks(priced_env, make_client):
    store = MemoryStore()
    client = demo_client(make_client, Upstreams(local=[connect_error()]), store)
    prompts = load_prompts(sets=("fallback",))

    summary = run(client, prompts, max_tokens=64, out=io.StringIO())

    assert summary.fallbacks == summary.ok == len(prompts)
    client.get("/v1/stats")
    assert all(r.fallback_from for r in store.requests.values())


def test_preflight_blocks_without_a_premium_model(make_client, monkeypatch):
    monkeypatch.delenv("PREMIUM_API_KEY")
    client = make_client(Upstreams())

    with pytest.raises(PreflightError, match="STACK_MODE=mock"):
        preflight(client, load_prompts(), fallback_run=False)


def test_preflight_rejects_a_wrong_key(make_client):
    client = make_client(Upstreams(), api_key="wrong")

    with pytest.raises(PreflightError, match="rejected"):
        preflight(client, load_prompts(), fallback_run=False)


def test_main_needs_a_key_and_never_prints_it(monkeypatch, capsys):
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)

    assert main(["--url", "http://127.0.0.1:9"]) == 2
    assert "ROUTER_API_KEY" in capsys.readouterr().err

    monkeypatch.setenv("ROUTER_API_KEY", "super-secret-key-value")
    assert main(["--url", "http://127.0.0.1:9", "--timeout", "1"]) == 2
    captured = capsys.readouterr()
    assert "cannot reach the router" in captured.err
    assert "super-secret-key-value" not in captured.out + captured.err
