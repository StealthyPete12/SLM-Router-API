"""Provisioning files: Grafana dashboard and data sources, Prometheus, Compose, SQL vs schema.

Static checks only; no containers needed. With TEST_DATABASE_URL set (a Postgres
initialised from infra/postgres/init.sql, e.g. the Compose one), every dashboard
query is also run against the real schema.
"""

import json
import os
import re
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from app.storage import RequestRecord
from app.storage.db import _COLUMNS
from scripts.build_dashboard import OUTPUT, render

ROOT = Path(__file__).resolve().parent.parent
INFRA = ROOT / "infra"
DASHBOARD = json.loads(OUTPUT.read_text(encoding="utf-8"))
PANELS = [p for p in DASHBOARD["panels"] if p["type"] != "row"]
DATASOURCE_UIDS = {"slm-postgres", "slm-prometheus"}


def schema_columns() -> dict[str, set[str]]:
    """Columns per table, read from the CREATE TABLE statements in init.sql."""
    sql = (INFRA / "postgres/init.sql").read_text()
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", sql, re.S):
        tables[name] = {
            line.split()[0]
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("--")
        }
    return tables


def panel_sql() -> list[tuple[str, str]]:
    return [
        (p["title"], t["rawSql"])
        for p in PANELS
        for t in p.get("targets", [])
        if t["datasource"]["uid"] == "slm-postgres"
    ]


# Dashboard ------------------------------------------------------------------------------


def test_committed_dashboard_matches_the_builder():
    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        "infra/grafana/dashboards/slm-router.json is stale: run "
        "`uv run python -m scripts.build_dashboard`"
    )


def test_dashboard_identity_is_stable():
    assert DASHBOARD["uid"] == "slm-router"
    assert DASHBOARD["editable"] is False
    ids = [p["id"] for p in DASHBOARD["panels"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("area", "titles"),
    [
        (
            "A. Cost savings",
            ["Estimated savings", "Savings vs all-premium baseline", "Estimated savings per day"],
        ),
        (
            "B. Routing decisions",
            ["Routing mix (% of successful requests)", "Requests by route over time"],
        ),
        ("C. Latency", ["p95 latency by tier (Prometheus)"]),
        ("D. Token usage", ["Tokens by model (input vs output)"]),
        ("E. Quality feedback", ["Thumbs-up rate", "Thumbs-up rate by tier"]),
        (
            "F. Model performance",
            ["Router accuracy", "Under-routed", "Answer quality", "Evaluation runs"],
        ),
    ],
)
def test_required_panels_are_present(area, titles):
    present = {p["title"] for p in PANELS}
    assert set(titles) <= present, f"{area} is missing {set(titles) - present}"


def test_every_panel_uses_a_provisioned_datasource_and_explains_itself():
    for panel in PANELS:
        if panel["type"] == "text":
            continue
        assert panel["datasource"]["uid"] in DATASOURCE_UIDS, panel["title"]
        assert panel.get("description"), f"{panel['title']} has no description"
        for target in panel["targets"]:
            assert target["datasource"]["uid"] in DATASOURCE_UIDS


def test_panels_fit_the_grid():
    for panel in DASHBOARD["panels"]:
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24, panel["title"]


def test_latency_uses_the_prometheus_histogram():
    panel = next(p for p in PANELS if p["title"] == "p95 latency by tier (Prometheus)")
    expr = panel["targets"][0]["expr"]
    assert expr.startswith("histogram_quantile(0.95,")
    assert "router_latency_seconds_bucket" in expr
    assert "by (le, tier)" in expr


def test_money_is_labelled_as_estimated():
    money = [
        p
        for p in PANELS
        if p.get("fieldConfig", {}).get("defaults", {}).get("unit") == "currencyUSD"
    ]
    assert money
    for panel in money:
        assert "stimated" in panel["title"] or "stimated" in panel["description"]


def test_ratios_guard_against_division_by_zero():
    for title, sql in panel_sql():
        for divisor in re.findall(r"/\s*(\w+\(.*?\))", sql):
            assert divisor.startswith("NULLIF(") or "count(*) >= 5" in sql, (title, divisor)


def test_eval_panels_have_a_clear_no_data_state():
    for panel in PANELS:
        sql = " ".join(t.get("rawSql", "") for t in panel.get("targets", []))
        if "eval_runs" in sql:
            # The empty state says which command fills it.
            assert "make eval" in panel["fieldConfig"]["defaults"]["noValue"], panel["title"]


def test_answer_quality_shows_the_latest_measured_score():
    # Routing-only runs store quality_score NULL; they must not blank the panel.
    panel = next(p for p in PANELS if p["title"] == "Answer quality")
    assert "quality_score IS NOT NULL" in panel["targets"][0]["rawSql"]


def test_cache_hits_stay_out_of_per_model_latency_and_tokens():
    for title in ("p95 latency by tier (whole time range)", "Tokens by model (input vs output)"):
        panel = next(p for p in PANELS if p["title"] == title)
        assert "NOT cache_hit" in panel["targets"][0]["rawSql"], title


def test_time_filters_are_applied():
    for title, sql in panel_sql():
        if "eval_runs" in sql and "LIMIT 1" in sql:
            continue  # "latest run", documented in the description
        assert "$__timeFilter(" in sql, title


# SQL vs schema --------------------------------------------------------------------------

SQL_WORDS = {
    "select",
    "from",
    "where",
    "and",
    "or",
    "not",
    "is",
    "null",
    "as",
    "in",
    "on",
    "join",
    "group",
    "by",
    "order",
    "desc",
    "asc",
    "limit",
    "filter",
    "case",
    "when",
    "then",
    "end",
    "within",
    "count",
    "sum",
    "nullif",
    "percentile_cont",
    "date_trunc",
    "to_char",
    "distinct",
    "__timefilter",
    "__timegroupalias",
    "__interval",
    "client",  # $client variable
    "r",
    "f",  # table aliases
}


def test_dashboard_sql_only_uses_known_tables_and_columns():
    tables = schema_columns()
    known = set(tables) | set().union(*tables.values())
    for title, sql in panel_sql():
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql)  # literals and aliases
        words = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped)}
        unknown = words - known - SQL_WORDS
        assert not unknown, f"{title}: unknown identifiers {sorted(unknown)}"


def test_store_writes_every_requests_column():
    requests = schema_columns()["requests"]
    assert set(_COLUMNS) == requests
    assert {f.name for f in fields(RequestRecord)} == requests


def test_schema_has_the_designed_tables():
    tables = schema_columns()
    assert {"requests", "feedback", "eval_runs"} <= set(tables)
    assert {"request_id", "rating", "comment", "created_at"} <= tables["feedback"]
    assert {
        "threshold",
        "dataset_size",
        "router_accuracy",
        "under_routed_pct",
        "local_share",
        "savings_pct",
        "quality_score",
    } <= tables["eval_runs"]


def grafana_sql(sql: str) -> str:
    """Expand the Grafana macros this dashboard uses into plain Postgres."""
    sql = re.sub(r"\$__timeFilter\(([\w.]+)\)", r"\1 > now() - interval '7 days'", sql)
    sql = re.sub(
        r"\$__timeGroupAlias\(([\w.]+), [^)]+\)",
        r"date_trunc('hour', \1) AS time",
        sql,
    )
    return sql.replace("$client", "'__all__'")


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_dashboard_sql_runs_against_the_real_schema():
    import asyncio

    import asyncpg

    async def check() -> None:
        connection = await asyncpg.connect(os.environ["TEST_DATABASE_URL"])
        try:
            for title, sql in panel_sql():
                await connection.fetch(grafana_sql(sql))  # raises on any schema mismatch
                del title
        finally:
            await connection.close()

    asyncio.run(check())


# Grafana provisioning -------------------------------------------------------------------


def test_both_datasources_are_provisioned_with_stable_uids_and_no_secrets():
    text = (INFRA / "grafana/provisioning/datasources/datasources.yml").read_text()
    config = yaml.safe_load(text)
    sources = {d["uid"]: d for d in config["datasources"]}

    assert set(sources) == DATASOURCE_UIDS
    postgres, prometheus = sources["slm-postgres"], sources["slm-prometheus"]
    assert postgres["type"] == "grafana-postgresql-datasource"
    assert postgres["url"] == "postgres:5432"
    assert postgres["user"] == "grafana_reader"
    assert postgres["secureJsonData"]["password"] == "$GRAFANA_DB_PASSWORD"
    assert prometheus["type"] == "prometheus"
    assert prometheus["url"] == "http://prometheus:9090"


def test_dashboard_provider_points_at_the_mounted_folder():
    config = yaml.safe_load((INFRA / "grafana/provisioning/dashboards/dashboards.yml").read_text())
    provider = config["providers"][0]
    assert provider["type"] == "file"
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"
    assert provider["allowUiUpdates"] is False


# Prometheus -----------------------------------------------------------------------------


def test_prometheus_scrapes_the_api_metrics_endpoint():
    config = yaml.safe_load((INFRA / "prometheus/prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    api = jobs["slm-router-api"]
    assert api.get("metrics_path", "/metrics") == "/metrics"
    assert api["static_configs"][0]["targets"] == ["api:8000"]
    assert config["global"]["scrape_interval"].endswith("s")


# Compose --------------------------------------------------------------------------------

COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_compose_runs_the_whole_stack():
    services = COMPOSE["services"]
    assert {"api", "ollama", "postgres", "redis", "prometheus", "grafana", "playground"} <= set(
        services
    )
    assert services["playground"]["environment"]["ROUTER_API_URL"] == "http://api:8000"
    assert services["api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert services["api"]["environment"]["REDIS_URL"] == "${REDIS_URL:-redis://redis:6379/0}"
    for name in ("postgres", "redis", "prometheus", "grafana", "playground", "api", "ollama"):
        assert "healthcheck" in services[name], name


def test_compose_ports_stay_on_localhost():
    for name, service in COMPOSE["services"].items():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), f"{name} exposes {port}"


def test_compose_mounts_exist():
    for service in COMPOSE["services"].values():
        for volume in service.get("volumes", []):
            source = volume.split(":")[0]
            if source.startswith("./"):
                assert (ROOT / source).exists(), source


def test_no_secret_is_hardcoded_in_compose():
    secret = re.compile(r"(KEY|PASSWORD|SECRET|TOKEN)", re.I)
    for path in ("docker-compose.yml", "docker-compose.offline.yml"):
        for service in yaml.safe_load((ROOT / path).read_text())["services"].values():
            for name, value in (service.get("environment") or {}).items():
                if secret.search(name):
                    assert str(value).startswith("${"), f"{path}: {name} is hardcoded"


def test_offline_overlay_labels_local_answers_as_mock():
    offline = yaml.safe_load((ROOT / "docker-compose.offline.yml").read_text())["services"]
    assert offline["api"]["environment"]["LOCAL_PROVIDER"] == "mock"


def test_mock_overlay_is_labelled():
    mock = yaml.safe_load((ROOT / "docker-compose.mock.yml").read_text())["services"]
    assert mock["api"]["environment"]["PREMIUM_PROVIDER"] == "mock"
    assert mock["api"]["environment"]["PREMIUM_API_KEY"] == "mock-key-not-a-secret"
