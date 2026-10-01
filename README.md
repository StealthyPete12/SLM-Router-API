# SLM Router API

An OpenAI-compatible API that answers each prompt with the cheapest model that can handle it: simple prompts go to a small language model running locally in Ollama, complex ones to a premium cloud model. Every response says which model answered and why.

> **Status: Phase 3 (dashboard and playground).** Routing, PII masking and privacy routing, local-to-cloud fallback, cost and savings estimates, Postgres logging, feedback, Prometheus metrics, a provisioned Grafana dashboard, a Streamlit playground and a demo-data loader all work. Not built yet (Phase 4): the labelled evaluation set, router-accuracy results, threshold tuning and the Redis cache. No accuracy or savings result is claimed until that evaluation runs.

## Prerequisites

- Docker with Docker Compose v2
- GNU Make
- [uv](https://docs.astral.sh/uv/) (only for running lint and tests on the host; it installs Python 3.12 if needed)
- About 12 GB of free disk for the three models, and 8 GB of RAM or more (Llama 3.1 8B on CPU). No room for that? Use the offline mode below.
- Optional: an NVIDIA GPU with the NVIDIA Container Toolkit. CPU works, just more slowly.
- Optional: an API key for one OpenAI-compatible cloud provider (OpenAI, Gemini or Anthropic). Without one, local routing and dry runs still work.

## Run it

```bash
cp .env.example .env   # set ROUTER_API_KEY, the PREMIUM_* settings and prices
make up                # build and start the whole stack, wait until every service is healthy
make models            # once: pull phi3:mini, llama3.1:8b and mistral:7b (about 11 GB)
make health            # API status, Ollama models, Postgres, and whether premium is configured
make demo              # fill the dashboard with demo traffic (see "Demo data")
```

`make up` runs `docker compose up -d --build --wait` with the Compose files for your `STACK_MODE`, so plain `docker compose up` works too for the real mode. Models, the database, metrics and Grafana state live in volumes and survive `make down`. With an NVIDIA GPU, use `make up-gpu` in place of `make up`.

| Service | URL (localhost only) | What it is |
| --- | --- | --- |
| Router API | http://localhost:8000 (docs at `/docs`) | The OpenAI-compatible gateway |
| Playground | http://localhost:8501 | Streamlit chat with a routing card per answer |
| Grafana | http://localhost:3000 | The six-area dashboard; anonymous view, admin login from `.env` |
| Prometheus | http://localhost:9090 | Scrapes the API's `/metrics` every 10 s |
| Postgres | `localhost:5432` | `requests`, `feedback` and `eval_runs` |
| Ollama | http://localhost:11434 | Local models |

Every port is bound to 127.0.0.1. `make logs` follows every service; `make logs-api`, `logs-ollama`, `logs-postgres`, `logs-prometheus`, `logs-grafana` and `logs-playground` follow one.

### Modes: real, mock and offline

Set `STACK_MODE` in `.env` (the Makefile reads it) or pass it once, e.g. `make up STACK_MODE=mock`:

| `STACK_MODE` | Local answers | Premium answers | Needs |
| --- | --- | --- | --- |
| `real` (default) | Ollama | your cloud provider | pulled models, a provider key and prices |
| `mock` | Ollama | mock model (`docker-compose.mock.yml`) | pulled models |
| `offline` | mock models (`docker-compose.offline.yml`) | mock model | nothing: no key, no downloads |

The mock models (`app/devtools/mock_llm.py`) are for development only. Every mock answer starts with `[Simulated ... answer ...; mock mode, no model was called.]`, latency is simulated, the premium prices are illustrative placeholders, and each stored request has `provider = mock`. Numbers from these modes are demo data: never quote them as results.

See where a prompt would go, without calling any model:

```bash
curl -s http://localhost:8000/v1/route \
  -H "Authorization: Bearer $ROUTER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Compare Postgres and MongoDB for event sourcing, step by step."}]}'
```

```json
{
  "router": {
    "tier": "premium",
    "reason": "score_at_or_above_threshold",
    "task_type": "analysis",
    "complexity_score": 45,
    "threshold": 30,
    "signals": {"task_base": 40, "input_length": 0, "reasoning_cues": 5, "multi_part": 0,
                "output_demand": 0, "conversation_depth": 0, "code_in_input": 0},
    "model_id": "premium-default",
    "summary": "premium -> premium-default: score 45 (task_base 40 + reasoning_cues 5, analysis) meets the threshold of 30"
  },
  "explanation": {"classification": {"evidence": ["..."]}, "rules": ["..."], "signal_details": {"...": "..."}}
}
```

Ask for a real answer with the same body at `/v1/chat/completions`. The response is a normal OpenAI chat completion plus the `router` block, which on real answers also carries `latency_ms`, `cost_usd`, `baseline_cost_usd`, `savings_usd`, `cache_hit`, `fallback_from` and `pii_detected`. The decision also goes out as `X-Router-Tier`, `X-Router-Model` and `X-Router-Score` headers (and `X-Router-Fallback-From` after a fallback). `make dry-run` routes the 13 acceptance prompts through the running API.

On CPU, the first request to a model is slow while Ollama loads it into memory. Interactive API docs are at http://localhost:8000/docs.

## How routing works

Each prompt gets a complexity score from 0 to 100. Under 30 it stays local; 30 or more goes premium. All weights, keywords and thresholds live in `config/routing.yaml`.

1. **Tokens**: the request is measured with `tiktoken` (`o200k_base`). That's an estimate for routing; final costs (Phase 2) will use provider-reported usage.
2. **Task type**: keyword rules label the last user message as one of chitchat, qa, rewrite, summarise, translate, extract, creative, code, math or analysis. Keywords in the first line count more than ones in pasted text, and ties go to the more demanding task.
3. **Score**: task base points (0 to 40) plus modifiers for input length, reasoning cues ("step by step"), multi-part requests, output demand ("detailed", 500+ words, `max_tokens` ≥ 1000), conversation depth, and a code block in a non-code prompt.
4. **Hard rules**, before the threshold and in this order: privacy (regex PII detection; with `privacy_mode: true` a prompt with PII stays local even when premium is forced), caller override, model capability (e.g. `tools`), context-window fit. A forced tier never bypasses capability, context or privacy.
5. **Model inside the tier**: chitchat and Q&A use Phi-3 Mini; summarise, translate and rewrite use Mistral 7B; other local work uses Llama 3.1 8B. Premium uses one configurable default.
6. **Fallback**: a local timeout, error or empty answer is retried once on the premium default and recorded as `fallback_from`. Never for privacy-kept prompts, an explicit model ID or a forced `X-Router-Tier: local`: those fail with an error that says why. Premium errors (unreachable, timeout, 429, 5xx) get one retry after a backoff.

**Cost and savings** use provider-reported tokens priced from `models.yaml`: cost = (tokens in × input price + tokens out × output price) / 1,000,000, and savings = baseline cost − cost, where the baseline prices the same tokens at the premium default. They are estimates (a cloud model would have written a different number of tokens), and they are stored as unknown, not zero, until you set the premium prices.

## API

All `/v1` endpoints need `Authorization: Bearer <ROUTER_API_KEY>`. If the server has no key set, they answer 503. `/health` needs no key.

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Always 200 while the API is up. `status` is `ok` only when Ollama answers, every local model is pulled and Postgres answers. Also reports whether premium is configured (names of missing settings, never values). |
| `POST /v1/route` | Dry run: the decision, score breakdown, classifier evidence and hard-rule trace. Calls no model. |
| `POST /v1/chat/completions` | Non-streaming chat. `model: "auto"` routes; a model ID from `config/models.yaml` skips the router. Optional `X-Router-Tier: local\|premium` forces a tier; optional `X-Router-Client: <label>` is stored with the request (the playground sends `playground`, the loader `demo-loader`). |
| `POST /v1/feedback` | `{"request_id": "req_...", "rating": 1 or -1, "comment": "optional"}`. One rating per answer: a second one replaces the first (`status: updated`). Comments are PII-masked. Dry-run IDs are not stored, so they answer 404. |
| `GET /v1/stats` | Totals for the playground sidebar: requests, local/premium/fallback/cache shares, estimated cost and savings, p95 latency, feedback. `?hours=24` limits the window. |
| `GET /v1/models` | `auto` plus every configured model with tier, context window, capabilities, prices and whether it is configured. |
| `GET /metrics` | Prometheus metrics, no key (like `/health`): `router_requests_total{tier,model,reason,status}`, `router_latency_seconds{tier,model}`, `router_tokens_total{model,direction}`, `router_savings_usd_total`, `router_cost_usd_total`, `router_fallbacks_total`, `router_feedback_total`, `router_storage_errors_total`. |

Errors: missing or wrong key 401, invalid `X-Router-Tier` or `X-Router-Client`, or a header that contradicts an explicit model 400, input over `guardrails.max_input_chars` 413, unknown model 404, request too large for any allowed model 413, no capable model 422, invalid body 422, `stream: true` 400, provider unreachable or erroring 502, premium model not configured 503, provider timeout 504. Every response carries an `X-Request-ID` header.

Any OpenAI SDK works as a client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<ROUTER_API_KEY>")
reply = client.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "Hi! What can you do?"}]
)
print(reply.model)  # phi3:mini
```

## Dashboard

Grafana is provisioned entirely from `infra/grafana/`: both data sources (Postgres `slm-postgres`, through a read-only role, and Prometheus `slm-prometheus`) and the dashboard open at http://localhost:3000 with no clicking. The dashboard JSON is generated by `scripts/build_dashboard.py` (`make dashboard`); edit that, not the UI. The `Client` filter separates demo-loader, playground and API traffic.

| Area | Panels | Source |
| --- | --- | --- |
| Cost savings | Estimated savings, savings % vs the all-premium baseline, estimated spend, savings per day | `requests` |
| Routing decisions | Share of local, premium, fallback and cache answers; requests by route over time; routing reasons | `requests` |
| Latency | p95 by tier over time; p95 by tier for the whole range | Prometheus histogram; `requests` |
| Token usage | Input vs output tokens by model | `requests` |
| Quality feedback | Thumbs-up rate; by tier, once a tier has 5 or more votes | `feedback` joined to `requests` |
| Model performance | Router accuracy, under-routing, answer quality, evaluation runs | `eval_runs` |

**Expected empty states before Phase 4:** the four Model performance panels read "No eval runs (Phase 4)" because `eval/run_eval.py` does not exist yet, and the Cache share is 0% because the Redis cache arrives in Phase 4. Money panels read "Unknown" until premium prices are set. The Prometheus latency line has gaps where there was no traffic; the "whole range" panel covers bursts such as one demo load.

Check that Prometheus scrapes the API: http://localhost:9090/targets should show `slm-router-api` as **UP**, or `curl -s localhost:9090/api/v1/query?query=router_requests_total`.

## Playground

http://localhost:8501 (also `make playground` to run it on the host). It sends `model: "auto"` by default and shows, under every answer, a routing card: tier, model, task type, complexity score against the threshold on a bar, the reason, the signal breakdown, latency, estimated cost, baseline and savings, and fallback, cache and PII flags.

- **Routing**: Automatic, Force local or Force premium (sends `X-Router-Tier`; privacy, capability and context rules still apply).
- **Dry run**: uses `POST /v1/route`, calls no model, and shows the same card with the input-only cost estimate and the hard-rule trace.
- **Feedback**: thumbs up or down plus an optional comment under each answer, sent with that answer's `request_id`. The buttons disappear once you vote, so a double click sends one vote.
- **Clear conversation** resets the chat. Dry runs and failed prompts stay on screen but are not sent as conversation history.

The API key comes from the page or from `ROUTER_API_KEY` in the environment; the page never displays it. If the API is down, the sidebar says so and the page keeps working once it is back.

## Demo data

```bash
make demo   # 36 mixed prompts, then 4 more with the local backend stopped to show a real fallback
uv run python -m eval.load_demo --set main --url http://localhost:8000   # just the main set
```

`eval/load_demo.py` replays `eval/demo_prompts.jsonl`: every task type, both tiers, forced tiers, two prompts with fictional PII (example.com, 555 numbers) and a fallback set. It prints each decision and a summary. The loader's thumbs up and down are **synthetic** (the comment says so), every request is tagged `client = demo-loader`, and the prompts are fixed fixtures, so this fills the dashboard but measures nothing. Re-running adds the same traffic again; `make reset-data` (destructive, asks first) empties the database, metrics and Grafana state.

The loader stops with a clear message if the API is unreachable, the key is wrong, or no premium model is configured (use `STACK_MODE=mock` without a cloud key). It reads `ROUTER_API_KEY` from the environment and never prints it.

## Configuration

| File | Holds |
| --- | --- |
| `config/models.yaml` | Model catalog: ID, provider, tier, endpoint, provider-facing model name, API-key variable name, context window, prices and `priced_on`, capabilities |
| `config/routing.yaml` | Threshold, task base points, classifier keywords, modifiers, override and context-fit settings, model per task, timeouts |
| `.env` | `ROUTER_API_KEY`, `STACK_MODE`, the premium provider settings and prices, Ollama overrides, Postgres and Grafana credentials (local-only defaults). Copy `.env.example`. Gitignored. |
| `infra/` | `postgres/init.sql` (schema, applied once to an empty volume), `prometheus/prometheus.yml`, `grafana/` (provisioning and dashboard JSON) |

The premium model ID and prices are deliberately not hardcoded: set `PREMIUM_MODEL` from your provider's docs, and `PREMIUM_USD_PER_1M_INPUT`, `PREMIUM_USD_PER_1M_OUTPUT` and `PREMIUM_PRICED_ON` from its pricing page. Keys are only read from the environment variable named by `api_key_env`.

Prompts are never stored in full: each row keeps a PII-masked 200-character preview and a SHA-256 hash.

The API validates both YAML files at startup and refuses to start with an error that names the file and field. `config/` is mounted into the container, so after editing it run `docker compose restart api`.

Ollama's context window is set by `OLLAMA_CONTEXT_LENGTH` (default 8192). It must equal `context_window` in `models.yaml`. The API checks this at startup.

**Mac:** Docker Desktop has no GPU passthrough. Run Ollama natively, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`, then start everything except the Ollama container with `docker compose up -d --build --no-deps api postgres prometheus grafana playground` and pull models with `ollama pull <tag>`.

## Development

```bash
make install   # uv sync: creates .venv with the runtime and dev dependencies
make lint      # ruff check + ruff format --check
make format    # apply ruff fixes and formatting
make test      # pytest; every provider is mocked, so no Docker, models or API keys needed
make dry-run   # acceptance prompts through a running API's /v1/route
make demo      # demo traffic for the dashboard
make logs      # follow every service (the API logs one JSON object per event)
make dashboard # regenerate the Grafana dashboard JSON
make down      # stop the stack
```

`TEST_DATABASE_URL=postgresql://grafana_reader:<GRAFANA_DB_PASSWORD>@127.0.0.1:5432/router uv run pytest tests/test_infra.py` also runs every dashboard query against the live schema.
```

`uv run python -m scripts.acceptance` runs the acceptance prompts in-process, with no server. Run `make` on its own to list every command.

## Layout

```text
app/
  main.py                 gateway: auth, request IDs, every endpoint
  config.py               typed loading and validation of config/*.yaml
  schemas.py              OpenAI-style request and response models, router metadata
  router/                 tokens.py, classifier.py, scoring.py, policy.py, privacy.py, engine.py
  aggregator/             guardrails.py (PII, size, empty answers), dispatch.py (retry, fallback),
                          costs.py, formatter.py (router block, stored row, headers)
  providers/              base.py (Provider interface), openai_compat.py (Ollama and cloud)
  storage/                Postgres store, in-memory store (no DATABASE_URL), background recorder
  metrics.py              Prometheus counters and histograms
  devtools/mock_llm.py    development-only mock model server for the mock and offline modes
  log.py                  JSON log formatting
config/                   models.yaml, routing.yaml
infra/                    postgres/init.sql, prometheus/prometheus.yml, grafana/ provisioning
ui/                       playground.py (Streamlit), client.py (API client and card parsing)
eval/                     load_demo.py and demo_prompts.jsonl (demo traffic, not an evaluation)
scripts/                  acceptance.py (routing prompts), build_dashboard.py (dashboard JSON)
tests/                    unit, API, playground-client, loader and provisioning tests
docker-compose.yml        the stack; .gpu.yml adds a GPU, .mock.yml / .offline.yml mock models
```
