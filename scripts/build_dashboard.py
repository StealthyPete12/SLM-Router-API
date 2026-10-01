"""Builds infra/grafana/dashboards/slm-router.json, the six-area router dashboard.

The JSON is generated so the SQL stays readable and reviewable here; Grafana
loads the committed JSON through file provisioning. After editing this file:

    uv run python -m scripts.build_dashboard

tests/test_grafana.py fails if the committed JSON is out of date.

Sources: Postgres (`requests`, `feedback`, `eval_runs`) for everything except
live latency, which comes from the Prometheus histogram. Money values are
estimates. Colours are fixed per entity (validated for CVD separation on both
Grafana themes), so a filter never repaints a series.
"""

import json
from pathlib import Path
from typing import Any

OUTPUT = Path(__file__).resolve().parent.parent / "infra/grafana/dashboards/slm-router.json"

POSTGRES = {"type": "grafana-postgresql-datasource", "uid": "slm-postgres"}
PROMETHEUS = {"type": "prometheus", "uid": "slm-prometheus"}

# One colour per entity, in fixed order: local, premium, fallback, cache.
COLOURS = {
    "Local": "#3987e5",
    "Premium": "#d95926",
    "Fallback": "#199e70",
    "Cache": "#c98500",
    "Input tokens": "#9085e9",
    "Output tokens": "#d55181",
}
TIER_COLOURS = {"local": COLOURS["Local"], "premium": COLOURS["Premium"]}

# The `client` variable filters by X-Router-Client. Its "All" value is the quoted
# literal '__all__', so the clause also works before any client exists.
CLIENT = "('__all__' IN ($client) OR client IN ($client))"
R_CLIENT = "('__all__' IN ($client) OR r.client IN ($client))"
OK_ROWS = f"$__timeFilter(created_at) AND status = 'ok' AND {CLIENT}"

NO_EVAL = "No evaluation runs stored yet: run `make eval eval-store`"
NO_EVAL_SHORT = "No eval runs yet (make eval-store)"
NO_QUALITY = "Not measured yet (paid check: make eval-quality-paid)"
ESTIMATE = (
    "Estimated: costs use provider-reported tokens priced from config/models.yaml; "
    "the baseline prices the same tokens at the premium default, although a cloud "
    "model would have written a different number of output tokens."
)


class Layout:
    """Places panels left to right on Grafana's 24-column grid."""

    def __init__(self) -> None:
        self.y = 0
        self.x = 0
        self.row_height = 0
        self.next_id = 1
        self.panels: list[dict[str, Any]] = []

    def row(self, title: str) -> None:
        self._newline()
        self.panels.append(
            {
                "id": self._id(),
                "type": "row",
                "title": title,
                "collapsed": False,
                "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
                "panels": [],
            }
        )
        self.y += 1

    def add(self, panel: dict[str, Any], w: int, h: int) -> None:
        if self.x + w > 24:
            self._newline()
        panel = {"id": self._id(), **panel, "gridPos": {"x": self.x, "y": self.y, "w": w, "h": h}}
        self.panels.append(panel)
        self.x += w
        self.row_height = max(self.row_height, h)

    def _newline(self) -> None:
        self.y += self.row_height
        self.x = 0
        self.row_height = 0

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id - 1


def sql(raw: str, fmt: str = "table", ref: str = "A") -> dict[str, Any]:
    return {
        "refId": ref,
        "datasource": POSTGRES,
        "editorMode": "code",
        "format": fmt,
        "rawQuery": True,
        "rawSql": " ".join(raw.split()),
    }


def promql(expr: str, legend: str, ref: str = "A") -> dict[str, Any]:
    return {
        "refId": ref,
        "datasource": PROMETHEUS,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "range": True,
    }


def colour_overrides(names: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colour}}],
        }
        for name, colour in names.items()
    ]


def stat(
    title: str,
    description: str,
    query: dict[str, Any],
    unit: str,
    decimals: int | None = None,
    no_value: str = "No data in this time range",
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "unit": unit,
        "noValue": no_value,
        "color": {"mode": "fixed", "fixedColor": "text"},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": query["datasource"],
        "targets": [query],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "none",
            "graphMode": "none",
            "justifyMode": "center",
            "textMode": "value",
            "orientation": "auto",
        },
    }


def build() -> dict[str, Any]:
    layout = Layout()

    layout.add(
        {
            "type": "text",
            "title": "",
            "transparent": True,
            "options": {
                "mode": "markdown",
                "content": (
                    "**Local SLM Router.** Every money value is an *estimate* against an "
                    "all-premium baseline. Filter by `client` to separate demo-loader traffic "
                    "from playground or API traffic; rows whose provider is `mock` come from "
                    "the development mock models (simulated answers and latency, "
                    "illustrative prices). Load demo data with `make demo`."
                ),
            },
        },
        w=24,
        h=2,
    )

    # A. Cost savings -------------------------------------------------------------
    layout.row("Cost savings (estimated USD)")
    layout.add(
        stat(
            "Estimated savings",
            f"Sum of baseline cost minus actual cost over successful requests. {ESTIMATE} "
            "Unknown (not zero) when the premium model has no prices in models.yaml.",
            sql(f'SELECT sum(savings_usd) AS "Estimated savings" FROM requests WHERE {OK_ROWS}'),
            "currencyUSD",
            decimals=4,
            no_value="Unknown: no priced requests",
        ),
        w=6,
        h=5,
    )
    layout.add(
        stat(
            "Savings vs all-premium baseline",
            "Estimated savings as a share of what the same requests would have cost on the "
            "baseline model. Only requests whose savings are known count, on both sides "
            "of the ratio.",
            sql(
                f"""SELECT 100.0 * sum(savings_usd)
                / NULLIF(sum(baseline_cost_usd) FILTER (WHERE savings_usd IS NOT NULL), 0)
                AS "Savings vs baseline" FROM requests WHERE {OK_ROWS}"""
            ),
            "percent",
            decimals=1,
            no_value="Unknown: no priced requests",
        ),
        w=6,
        h=5,
    )
    layout.add(
        stat(
            "Estimated spend",
            f"What the requests actually cost. Local models cost 0. {ESTIMATE}",
            sql(f'SELECT sum(cost_usd) AS "Estimated spend" FROM requests WHERE {OK_ROWS}'),
            "currencyUSD",
            decimals=4,
            no_value="Unknown: no priced requests",
        ),
        w=6,
        h=5,
    )
    layout.add(
        stat(
            "Successful requests",
            "Chat completions answered in the time range (dry runs are not stored).",
            sql(f'SELECT count(*) AS "Requests" FROM requests WHERE {OK_ROWS}'),
            "none",
            no_value="0",
        ),
        w=6,
        h=5,
    )
    layout.add(
        {
            "type": "barchart",
            "title": "Estimated savings per day",
            "description": (
                "Daily sum of estimated savings, by request date (UTC), for the days inside "
                f"the time range. {ESTIMATE}"
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS "Day",
                    sum(savings_usd) AS "Estimated savings"
                    FROM requests WHERE {OK_ROWS} AND savings_usd IS NOT NULL
                    GROUP BY 1 ORDER BY 1"""
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "decimals": 4,
                    "noValue": "No priced requests in this time range",
                    "color": {"mode": "fixed", "fixedColor": COLOURS["Local"]},
                    "custom": {"fillOpacity": 85, "lineWidth": 1, "axisSoftMin": 0},
                },
                "overrides": [],
            },
            "options": {
                "orientation": "vertical",
                "xField": "Day",
                "barWidth": 0.3,
                "showValue": "auto",
                "legend": {"showLegend": False, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "single", "sort": "none"},
            },
        },
        w=24,
        h=7,
    )

    # B. Routing decisions ---------------------------------------------------------
    layout.row("Routing decisions")
    shares = {
        "Local": "NOT cache_hit AND fallback_from IS NULL AND tier = 'local'",
        "Premium": "NOT cache_hit AND fallback_from IS NULL AND tier = 'premium'",
        "Fallback": "NOT cache_hit AND fallback_from IS NOT NULL",
        "Cache": "cache_hit",
    }
    share_columns = ", ".join(
        f'100.0 * count(*) FILTER (WHERE {condition}) / NULLIF(count(*), 0) AS "{name}"'
        for name, condition in shares.items()
    )
    layout.add(
        {
            "type": "bargauge",
            "title": "Routing mix (% of successful requests)",
            "description": (
                "Local: answered by a local model. Premium: routed to and answered by the cloud "
                "model. Fallback: a local model failed (timeout, error or empty answer) and the "
                "premium model answered. Cache: answered from the exact-match Redis cache with "
                "no model call. The four add up to 100%."
            ),
            "datasource": POSTGRES,
            "targets": [sql(f"SELECT {share_columns} FROM requests WHERE {OK_ROWS}")],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "decimals": 1,
                    "min": 0,
                    "max": 100,
                    "noValue": "No requests in this time range",
                    "color": {"mode": "fixed", "fixedColor": "text"},
                },
                "overrides": colour_overrides(
                    {k: COLOURS[k] for k in ("Local", "Premium", "Fallback", "Cache")}
                ),
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "basic",
                "valueMode": "text",
                "showUnfilled": True,
                "namePlacement": "left",
                "text": {"titleSize": 14, "valueSize": 16},
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            },
        },
        w=7,
        h=8,
    )
    count_columns = ", ".join(
        f'count(*) FILTER (WHERE {condition}) AS "{name}"' for name, condition in shares.items()
    )
    layout.add(
        {
            "type": "timeseries",
            "title": "Requests by route over time",
            "description": "Successful requests per interval, stacked by route.",
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT $__timeGroupAlias(created_at, $__interval), {count_columns}
                    FROM requests WHERE {OK_ROWS} GROUP BY 1 ORDER BY 1""",
                    fmt="time_series",
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "decimals": 0,
                    "noValue": "No requests in this time range",
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 85,
                        "lineWidth": 1,
                        "stacking": {"mode": "normal", "group": "A"},
                        "axisSoftMin": 0,
                    },
                },
                "overrides": colour_overrides(
                    {k: COLOURS[k] for k in ("Local", "Premium", "Fallback", "Cache")}
                ),
            },
            "options": {
                "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "none"},
            },
        },
        w=9,
        h=8,
    )
    layout.add(
        {
            "type": "table",
            "title": "Why requests went where they did",
            "description": (
                "Routing reason per answering tier. Reasons come from the policy: the score "
                "threshold, a caller override, a capability or context-window rule, or privacy."
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT reason AS "Reason", tier AS "Tier", count(*) AS "Requests"
                    FROM requests WHERE {OK_ROWS}
                    GROUP BY reason, tier ORDER BY count(*) DESC"""
                )
            ],
            "fieldConfig": {
                "defaults": {"noValue": "No requests in this time range"},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": name},
                        "properties": [{"id": "custom.width", "value": width}],
                    }
                    for name, width in (("Tier", 80), ("Requests", 80))
                ],
            },
            "options": {"showHeader": True, "cellHeight": "sm"},
        },
        w=8,
        h=8,
    )

    # C. Latency -------------------------------------------------------------------
    layout.row("Latency (p95)")
    layout.add(
        {
            "type": "timeseries",
            "title": "p95 latency by tier (Prometheus)",
            "description": (
                "95th percentile of end-to-end latency (fallback included), from the "
                "router_latency_seconds histogram over a sliding window, by the tier that "
                "answered. Percentiles are interpolated inside histogram buckets. Gaps mean "
                "no traffic in the window; the API's counters restart with the container."
            ),
            "datasource": PROMETHEUS,
            "targets": [
                promql(
                    "histogram_quantile(0.95, sum by (le, tier) "
                    "(rate(router_latency_seconds_bucket[$__rate_interval])))",
                    "{{tier}}",
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "decimals": 2,
                    "noValue": "No traffic in this time range",
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 0,
                        "pointSize": 8,
                        "showPoints": "auto",
                        "spanNulls": False,
                        "axisSoftMin": 0,
                    },
                },
                "overrides": colour_overrides(TIER_COLOURS),
            },
            "options": {
                "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        },
        w=16,
        h=8,
    )
    layout.add(
        {
            "type": "bargauge",
            "title": "p95 latency by tier (whole time range)",
            "description": (
                "Exact 95th percentile of stored latencies (Postgres percentile_cont) over the "
                "whole selected range, by the tier that answered. Use it when traffic came in "
                "a burst, such as one demo load. Cache hits are left out: no model ran."
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT tier AS "Tier",
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS "p95"
                    FROM requests WHERE {OK_ROWS} AND latency_ms IS NOT NULL AND NOT cache_hit
                    GROUP BY tier ORDER BY tier"""
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "ms",
                    "decimals": 0,
                    "min": 0,
                    "noValue": "No requests in this time range",
                    "color": {"mode": "fixed", "fixedColor": "text"},
                },
                "overrides": colour_overrides(
                    {"local": COLOURS["Local"], "premium": COLOURS["Premium"]}
                ),
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "basic",
                "valueMode": "text",
                "showUnfilled": True,
                "namePlacement": "left",
                "text": {"titleSize": 14, "valueSize": 16},
                "reduceOptions": {"calcs": [], "fields": "/^p95$/", "values": True},
            },
        },
        w=8,
        h=8,
    )

    # D. Token usage and E. Quality feedback ----------------------------------------
    layout.row("Token usage and quality feedback")
    layout.add(
        {
            "type": "barchart",
            "title": "Tokens by model (input vs output)",
            "description": (
                "Tokens as reported by each provider (estimated with tiktoken when a provider "
                "reports none), summed per models.yaml model id over successful requests. "
                "Cache hits are left out: no model processed them."
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT model_id AS "Model",
                    sum(input_tokens) AS "Input tokens", sum(output_tokens) AS "Output tokens"
                    FROM requests WHERE {OK_ROWS} AND NOT cache_hit
                    GROUP BY model_id ORDER BY model_id"""
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "noValue": "No requests in this time range",
                    "custom": {"fillOpacity": 85, "lineWidth": 1, "axisSoftMin": 0},
                },
                "overrides": colour_overrides(
                    {k: COLOURS[k] for k in ("Input tokens", "Output tokens")}
                ),
            },
            "options": {
                "orientation": "horizontal",
                "xField": "Model",
                "groupWidth": 0.7,
                "barWidth": 0.9,
                "showValue": "auto",
                "stacking": "none",
                "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "none"},
            },
        },
        w=12,
        h=9,
    )
    feedback_where = f"$__timeFilter(f.created_at) AND {R_CLIENT}"
    layout.add(
        stat(
            "Thumbs-up rate",
            "Share of rated answers given a thumbs up (one rating per answer; a changed vote "
            "replaces the first). Demo-loader ratings are synthetic fixtures, not user "
            "judgements: filter client to exclude them.",
            sql(
                f"""SELECT 100.0 * count(*) FILTER (WHERE f.rating = 1) / NULLIF(count(*), 0)
                AS "Thumbs-up rate" FROM feedback f JOIN requests r ON r.id = f.request_id
                WHERE {feedback_where}"""
            ),
            "percent",
            decimals=0,
            no_value="No feedback in this time range",
        ),
        w=4,
        h=9,
    )
    layout.add(
        {
            "type": "table",
            "title": "Thumbs-up rate by tier",
            "description": (
                "Ratings grouped by the tier that answered (fallbacks count as premium). The "
                "rate is shown once a tier has at least 5 votes; below that it is too noisy "
                "to read, so only the vote count is shown."
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    f"""SELECT r.tier AS "Tier", count(*) AS "Votes",
                    count(*) FILTER (WHERE f.rating = 1) AS "Thumbs up",
                    CASE WHEN count(*) >= 5
                        THEN 100.0 * count(*) FILTER (WHERE f.rating = 1) / count(*) END
                        AS "Thumbs-up rate"
                    FROM feedback f JOIN requests r ON r.id = f.request_id
                    WHERE {feedback_where} GROUP BY r.tier ORDER BY r.tier"""
                )
            ],
            "fieldConfig": {
                "defaults": {"noValue": "No feedback in this time range"},
                "overrides": [
                    *(
                        {
                            "matcher": {"id": "byName", "options": name},
                            "properties": [{"id": "custom.width", "value": width}],
                        }
                        for name, width in (("Tier", 90), ("Votes", 70), ("Thumbs up", 100))
                    ),
                    {
                        "matcher": {"id": "byName", "options": "Thumbs-up rate"},
                        "properties": [
                            {"id": "unit", "value": "percent"},
                            {"id": "decimals", "value": 0},
                            {"id": "noValue", "value": "Fewer than 5 votes"},
                        ],
                    },
                ],
            },
            "options": {"showHeader": True, "cellHeight": "sm"},
        },
        w=8,
        h=9,
    )

    # F. Model performance -----------------------------------------------------------
    layout.row("Model performance (evaluation runs)")
    latest = (
        'SELECT {column} AS "{name}" FROM eval_runs WHERE {column} IS NOT NULL '
        "ORDER BY created_at DESC LIMIT 1"
    )
    eval_note = (
        "From the latest evaluation run that measured it, whatever the time range. Written "
        "by `make eval-store` (eval/run_eval.py); see docs/results.md for the method."
    )
    for column, name, help_text, no_value in [
        (
            "router_accuracy",
            "Router accuracy",
            "Share of labelled prompts routed to their expected tier (dry run, no model called).",
            NO_EVAL_SHORT,
        ),
        (
            "under_routed_pct",
            "Under-routed",
            "Prompts labelled premium that were routed local, as % of all labelled prompts. "
            "This hurts answer quality, so the threshold is tuned to keep it low.",
            NO_EVAL_SHORT,
        ),
        (
            "quality_score",
            "Answer quality",
            "Graded local answers judged the same as or better than the premium answer, by "
            "the premium model (model-based grading, spot-checked by hand; not ground truth).",
            NO_QUALITY,
        ),
    ]:
        layout.add(
            stat(
                name,
                f"{help_text} {eval_note}",
                sql(latest.format(column=column, name=name)),
                "percent",
                decimals=1,
                no_value=no_value,
            ),
            w=4,
            h=6,
        )
    layout.add(
        {
            "type": "table",
            "title": "Evaluation runs",
            "description": (
                "Every evaluation run in the time range: threshold, dataset size, accuracy, "
                "under- and over-routing, local share, estimated savings and answer quality "
                f"(all %). {eval_note}"
            ),
            "datasource": POSTGRES,
            "targets": [
                sql(
                    """SELECT created_at AS "Run", threshold AS "Threshold",
                    dataset_size AS "Prompts", router_accuracy AS "Accuracy %",
                    under_routed_pct AS "Under-routed %", over_routed_pct AS "Over-routed %",
                    local_share AS "Local share %", savings_pct AS "Savings % (est.)",
                    quality_score AS "Quality %"
                    FROM eval_runs WHERE $__timeFilter(created_at)
                    ORDER BY created_at DESC LIMIT 50"""
                )
            ],
            "fieldConfig": {"defaults": {"noValue": NO_EVAL}, "overrides": []},
            "options": {"showHeader": True, "cellHeight": "sm"},
        },
        w=12,
        h=6,
    )

    return {
        "uid": "slm-router",
        "title": "SLM Router",
        "description": "Cost savings, routing, latency, tokens, feedback and evaluation results.",
        "tags": ["slm-router"],
        "editable": False,
        "graphTooltip": 1,
        "schemaVersion": 41,
        "version": 1,
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "refresh": "30s",
        "timezone": "browser",
        "fiscalYearStartMonth": 0,
        "liveNow": False,
        "links": [],
        "annotations": {"list": []},
        "templating": {
            "list": [
                {
                    "name": "client",
                    "label": "Client",
                    "type": "query",
                    "datasource": POSTGRES,
                    "query": "SELECT DISTINCT client FROM requests ORDER BY 1",
                    "definition": "SELECT DISTINCT client FROM requests ORDER BY 1",
                    "refresh": 2,
                    "multi": True,
                    "includeAll": True,
                    "allValue": "'__all__'",
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "options": [],
                    "sort": 1,
                    "description": "X-Router-Client label: api, playground, demo-loader, ...",
                }
            ]
        },
        "panels": layout.panels,
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(render(), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}"
    )
