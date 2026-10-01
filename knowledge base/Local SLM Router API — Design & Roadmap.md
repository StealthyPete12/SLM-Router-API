# Local SLM Router API — Design & Roadmap

Oct 1, 2026 · @Timothy

## Overview

The router answers each prompt with the cheapest model that can handle it: prompts scoring under 30 go to a small language model running locally in Docker, the rest go to a premium cloud LLM. Every response says which model answered and why, and a dashboard tracks what that saved.

This is a showcase. It runs on one machine with `docker compose up` and exists to be demoed and read on GitHub, not installed by other people. So every choice below favours simple, explainable parts over scalable ones, and spends the effort where the capability becomes visible: the routing explanation, the dashboard and the evaluation results.

**What the finished project proves**

- A simple prompt is answered by a local SLM and a complex one by a cloud model, with the reason shown in the response.
- The dashboard shows the diagram's six metrics with real traffic behind them.
- An evaluation report gives router accuracy and cost savings against an all-premium baseline.
- The README tells the whole story in under two minutes.

**Out of scope for v1**

- Packaging or install support for other people
- Multi-tenant auth, billing and per-customer rate limits
- Kubernetes, autoscaling and high availability (Docker Compose only; Kubernetes is a stretch goal)
- Streaming responses
- Fine-tuning or training models

## Architecture

The whole system is seven containers on one machine. The diagram's middle boxes (gateway, classifier and router, aggregator) are modules inside a single FastAPI service, not separate services.

&#91;embedded content: Docker Compose stack · 7 containers plus cloud APIs\]

Requests enter at the gateway; the highlighted router sends each one to Ollama or a cloud model, and the aggregator logs every answer for the dashboard.

One service keeps the demo simple: the boxes stay separate in the code as packages but run as one process. Splitting them into microservices would add network hops and deployment work without showing anything new.

**How each box in the diagram is built**

| Diagram box | Built as | Notes |
| --- | --- | --- |
| Client Applications | Streamlit playground, plus any OpenAI SDK or curl | Clients point at the router's base URL and send `model: "auto"` |
| AI Router API Gateway | FastAPI app (`api` container) | OpenAI-compatible endpoint, bearer-key auth, request IDs |
| Prompt Classifier & Router | `app/router/` package | Token estimate, task type, complexity score, routing policy; rules-based in v1 |
| Local SLMs | Ollama container serving Llama 3.1 8B, Phi-3 Mini and Mistral 7B | [OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility) on port 11434; CPU works, GPU optional |
| Premium Cloud LLMs | OpenAI, Anthropic or Google over HTTPS | Start with one provider; adding another is a config entry |
| Response Aggregator | `app/aggregator/` package | One response shape, guardrails, cost and savings, logging |
| Monitoring & Analytics | Postgres, Prometheus and Grafana | Dashboard provisioned from files in the repo |
| Infrastructure | Docker Compose | Kubernetes is a stretch goal |

**Request lifecycle**

1. The client sends `POST /v1/chat/completions` with `model: "auto"`.
2. The gateway checks the API key, validates the body and assigns a `request_id`.
3. Input guardrails reject oversized input and scan for PII.
4. Redis is checked for an identical earlier request; a hit returns at once and is logged as a cache hit.
5. The classifier estimates tokens, labels the task and computes the complexity score.
6. The router policy picks the tier and the model (see Routing logic).
7. The provider is called with a timeout; a failed or empty local answer falls back to the cloud.
8. The aggregator normalises the answer, attaches the routing metadata and computes cost and savings.
9. The response goes back; the Postgres row, Prometheus metrics and cache write run as a background task.
10. The client can later send a thumbs up or down, linked by `request_id`.

**Hardware notes**

- Linux or Windows (WSL2) with an NVIDIA card: give the `ollama` container the GPU through the NVIDIA Container Toolkit.
- Mac: Docker Desktop has no GPU passthrough, so run Ollama natively and point the API at `host.docker.internal:11434` ([Ollama FAQ](https://docs.ollama.com/faq)).
- Ollama's default context window is 4,096 tokens. Set `OLLAMA_CONTEXT_LENGTH=8192` and mirror the value in `models.yaml`, so the router's context-fit rule matches what the model can actually take.

## Routing logic

Each prompt gets a complexity score from 0 to 100: under 30 it stays local, 30 or more goes to the cloud. Four hard rules are checked before the score, and every weight lives in `routing.yaml` so Phase 4 can tune it from data.

&#91;embedded content: Routing policy · 4 hard rules, then the threshold\]

Privacy is checked first, so a prompt with personal data never reaches the cloud, even when the caller tries to force a tier.

**Score = task base points + modifiers, capped at 100**

| Task type | Detected by (v1 keyword rules) | Base points |
| --- | --- | --- |
| Chit-chat | Greetings, very short messages with no task keyword | 0 |
| Simple Q&A | One short question: "what is", "who", "when" | 10 |
| Rewrite / format | "rewrite", "rephrase", "fix grammar", "convert to" | 10 |
| Summarise | "summarise" / "summarize", "tl;dr", "key points" | 15 |
| Translate | "translate", "in French" and similar | 15 |
| Extract / classify | "extract", "list all", "classify", "label" | 15 |
| Creative | "write a poem", "story", "slogan" | 20 |
| Code | Code blocks, "function", "bug", language names | 35 |
| Math / logic | Equations, "solve", "prove", "calculate" | 40 |
| Analysis / planning | "compare", "design", "strategy", "trade-offs", "plan" | 40 |

| Modifier | Rule | Points |
| --- | --- | --- |
| Input length | Under 200 tokens: 0 · 200 to 1,000: +5 · 1,000 to 3,000: +10 · over 3,000: +20 | 0 to +20 |
| Reasoning cues | "step by step", "explain why", "pros and cons", "justify", "edge cases" | +5 each, max +15 |
| Multi-part request | Each extra question or numbered requirement after the first | +5 each, max +10 |
| Output demand | "detailed", "comprehensive", "report", 500+ words asked for, or `max_tokens` of 1,000 or more | +10 |
| Conversation depth | More than 6 messages: +5 · more than 12: +10 | 0 to +10 |
| Code in input | A code block when the task is not already Code | +10 |

The response returns the breakdown (`task_base`, `input_length` and so on), so every decision can be explained and debugged.

**Picking the model inside a tier**

- Local: chit-chat and simple Q&A go to Phi-3 Mini (smallest, fastest); summarise, translate and rewrite go to Mistral 7B; everything else goes to Llama 3.1 8B.
- Premium: one default model until the evaluation shows a reason to split by task.

**Fallback**

- Local timeout (start at 60 s), error or empty answer: retry once on the premium default and record `fallback_from`.
- Premium error: one retry with backoff, then a 502 that carries the `request_id`.
- Privacy-routed requests never fall back to the cloud; they fail with a clear error instead.

**Worked examples**

| Prompt | Task | Score | Route |
| --- | --- | --- | --- |
| "Hi! What can you do?" | Chit-chat | 0 | Local · Phi-3 Mini |
| "Summarise this 600-word article in 3 bullets: …" | Summarise | 15 base + 5 length = 20 | Local · Mistral 7B |
| "Fix the bug in this function:" plus a pasted code block | Code | 35 base = 35 | Premium |
| "Compare Postgres and MongoDB for event sourcing, step by step." | Analysis | 40 base + 5 cue = 45 | Premium |
| "Rewrite this note to jane@example.com so it sounds friendlier: …" (privacy mode on) | Rewrite | Not used: PII rule | Local · Mistral 7B |

## API

The router speaks the OpenAI Chat Completions format, so any OpenAI SDK becomes a client by changing its base URL and sending `model: "auto"`. That drop-in property is one of the strongest things to show in the demo.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat. `model: "auto"` routes; a configured model ID skips the router |
| `POST /v1/route` | Dry run: returns the decision and score breakdown without calling a model |
| `POST /v1/feedback` | Thumbs up (+1) or down (-1) and an optional comment for a `request_id` |
| `GET /v1/stats` | Totals for the playground sidebar: requests, local share, savings, p95 latency |
| `GET /v1/models` | Configured models and their tiers |
| `GET /health` | Liveness, plus whether Ollama, Postgres and Redis answer |
| `GET /metrics` | Prometheus metrics |

Auth is one API key in `Authorization: Bearer ...`, the header OpenAI clients already send. An optional `X-Router-Tier: local|premium` header forces a tier.

**Drop-in client**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<ROUTER_API_KEY>")
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hi! What can you do?"}],
)
print(reply.model)  # e.g. phi3:mini
```

**Example response** (abridged; values illustrative)

```json
{
  "id": "req_8f3c...",
  "object": "chat.completion",
  "model": "mistral:7b",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 812, "completion_tokens": 64, "total_tokens": 876},
  "router": {
    "tier": "local",
    "reason": "score_below_threshold",
    "task_type": "summarise",
    "complexity_score": 20,
    "threshold": 30,
    "signals": {"task_base": 15, "input_length": 5},
    "latency_ms": 1840,
    "cost_usd": 0.0,
    "baseline_cost_usd": 0.0031,
    "savings_usd": 0.0031,
    "cache_hit": false,
    "fallback_from": null
  }
}
```

The same decision also goes out as `X-Router-Tier`, `X-Router-Model` and `X-Router-Score` response headers, for clients that drop unknown JSON fields.

## Data, cost and dashboards

Postgres holds one row per request and feeds five of the six dashboard panels; Prometheus supplies live latency. Because the dashboard and the evaluation report read the same rows, their numbers always agree.

**Tables** (created by `infra/postgres/init.sql`, no migration tool)

| Table | Holds | Key columns |
| --- | --- | --- |
| `requests` | One row per request | `id`, `created_at`, `client`, `task_type`, `complexity_score`, `signals` (jsonb), `tier`, `reason`, `model`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `baseline_cost_usd`, `savings_usd`, `cache_hit`, `fallback_from`, `pii_detected`, `status`, `prompt_preview` |
| `feedback` | Thumbs up or down per answer | `request_id`, `rating` (+1 / -1), `comment`, `created_at` |
| `eval_runs` | One row per evaluation run | `created_at`, `threshold`, `dataset_size`, `router_accuracy`, `under_routed_pct`, `local_share`, `savings_pct`, `quality_score` |

Prompts are stored only as a PII-masked 200-character preview plus a hash. That keeps personal data out of the database and out of every screenshot you publish.

**Cost and savings**

```latex
\text{cost} = \frac{t_{in}\,p_{in} + t_{out}\,p_{out}}{10^{6}} \qquad \text{savings} = \text{cost}_{\text{baseline}} - \text{cost}
```

Here t is the token count and p the price in USD per million tokens from `models.yaml`. The baseline prices the same token counts at the premium default model; local models cost 0 unless you set a notional compute price.

- Use the premium default as the baseline, not the most expensive flagship, or the savings figure is inflated.
- Label savings as estimates: a cloud model would have written a different number of output tokens.

**Dashboard panels** (one per metric in the diagram)

| Diagram metric | Grafana panel | Source |
| --- | --- | --- |
| Cost Savings (USD) | Total and % vs baseline, plus savings per day | `requests` |
| Routing Decisions (%) | Share of local, premium, fallback and cache hits | `requests` |
| Latency (P95) | p95 by tier over time | Prometheus histogram |
| Token Usage (Input/Output) | Input vs output tokens by model | `requests` |
| Quality Feedback (Users) | Thumbs-up rate by tier | `feedback` joined to `requests` |
| Model Performance (Accuracy) | Router accuracy, under-routing and answer quality per eval run | `eval_runs` |

**Prometheus metrics** at `/metrics`: `router_requests_total{tier, model, reason, status}`, the `router_latency_seconds{tier, model}` histogram, `router_tokens_total{model, direction}` and `router_savings_usd_total`.

## Tech stack and key decisions

Python and FastAPI throughout, Ollama for local models, and one OpenAI-compatible client for every provider. Each choice trades scale for simplicity and visibility, which is the right trade for a showcase.

| Decision | Choice | Why | Alternative considered |
| --- | --- | --- | --- |
| API framework | FastAPI on Python 3.12, Pydantic v2 | Async, typed validation and a free OpenAPI docs page to show; Python has the tokenizer and ML tools | Express (Node) |
| API shape | OpenAI-compatible `/v1/chat/completions` | Existing SDKs work unchanged, which is the strongest demo moment | A custom schema |
| Local runtime | Ollama in Docker | One command to pull and serve models, OpenAI-compatible API, runs on CPU | vLLM (faster, needs a GPU); llama.cpp server |
| Provider access | One `OpenAICompatibleProvider` behind a small `Provider` interface | Ollama, OpenAI and [Gemini](https://ai.google.dev/gemini-api/docs/openai) (beta) all accept OpenAI-format calls, so a new model is a YAML entry | One SDK per provider; LiteLLM |
| Anthropic | Its [OpenAI SDK compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) in v1, a native adapter later if needed | Anthropic positions the layer for testing and comparing models, which fits a showcase | Native SDK from day one |
| Complexity scoring | Weighted keyword rules | Deterministic, explainable, free and easy to unit-test | A small LLM as classifier; embeddings with logistic regression (stretch) |
| Token estimate | `tiktoken` with `o200k_base` for every model | Close enough for routing; actual cost uses each provider's reported usage | Per-model tokenizers |
| Analytics store | PostgreSQL 16 | Matches the diagram, Grafana reads it directly, SQL computes percentiles | SQLite |
| Dashboards | Grafana provisioned as code (Postgres and Prometheus sources) | Looks production-grade for little code, versioned in the repo | A custom React dashboard |
| Demo client | Streamlit playground | A usable UI in a single Python file | React or Next.js |
| Cache | Redis exact-match with a TTL | Simple, and cache hits show up as savings | Semantic cache (stretch) |
| Orchestration | Docker Compose | One machine, one command | Kubernetes (stretch) |
| Tooling | `uv`, `ruff`, `pytest` | Fast installs, one linter and formatter, standard tests | pip, black and flake8 |

## Repo layout and configuration

One repo, one Compose file and two YAML files that hold every model, price, weight and threshold, so nothing you will want to tune is hardcoded.

```text
slm-router/
├── docker-compose.yml
├── Makefile                # make models, make up, make eval
├── .env.example            # keys and URLs; .env stays out of git
├── config/
│   ├── models.yaml         # model catalog: tier, endpoint, prices, context window
│   └── routing.yaml        # threshold, weights, keywords, hard rules
├── app/
│   ├── main.py             # FastAPI app and endpoints
│   ├── router/             # tokens.py, classifier.py, scoring.py, policy.py
│   ├── providers/          # base.py (Provider interface), openai_compat.py
│   ├── aggregator/         # formatter.py, guardrails.py, costs.py
│   ├── storage/            # db.py (Postgres), cache.py (Redis)
│   └── metrics.py          # Prometheus counters and histograms
├── ui/playground.py        # Streamlit demo client
├── eval/
│   ├── prompts.jsonl       # labelled prompts: task, expected tier
│   ├── run_eval.py         # accuracy, savings, quality
│   └── load_demo.py        # replays prompts to fill the dashboard
├── infra/
│   ├── postgres/init.sql
│   ├── prometheus/prometheus.yml
│   └── grafana/            # provisioned data sources and dashboard JSON
├── tests/
└── docs/                   # architecture image, ADRs, demo script, results
```

**`config/models.yaml`** (excerpt)

```yaml
baseline: premium-default          # savings are measured against this model

models:
  - id: phi3-mini
    tier: local
    base_url: http://ollama:11434/v1
    model: phi3:mini
    context_window: 8192             # match OLLAMA_CONTEXT_LENGTH
    usd_per_1m_input: 0
    usd_per_1m_output: 0

  - id: premium-default
    tier: premium
    base_url: https://api.openai.com/v1   # or the Gemini / Anthropic OpenAI-compatible URL
    api_key_env: OPENAI_API_KEY
    model: <current model ID>
    context_window: <from the model docs>
    usd_per_1m_input: <from the pricing page>
    usd_per_1m_output: <from the pricing page>
    priced_on: <date you copied the prices>
```

The diagram's GPT-4o and Claude 3 labels are dated placeholders. Pick current model IDs from each provider's docs when you build, and record the date you copied the prices.

**`config/routing.yaml`** (excerpt)

```yaml
threshold: 30
task_base: {chitchat: 0, qa: 10, summarise: 15, code: 35, analysis: 40}   # all ten types
modifiers:
  input_length: {bands: [200, 1000, 3000], points: [0, 5, 10, 20]}
  reasoning_cues: {each: 5, max: 15}
  multi_part: {each: 5, max: 10}
local_models: {chitchat: phi3-mini, summarise: mistral-7b, default: llama3-8b}
premium_models: {default: premium-default}
privacy_mode: false                # true = prompts with PII never leave the machine
timeouts_s: {local: 60, premium: 60}
cache_ttl_s: 3600
```

**`.env.example`**: `ROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_BASE_URL`, `DATABASE_URL`, `REDIS_URL`.

## Roadmap

Six phases, about 16 to 22 focused days in total. Each phase ends at a gate you can demo, so the project is showable from Phase 1 onwards.

&#91;embedded content: Roadmap · 6 phases, effort in focused days\]

Effort is in focused days, not calendar time, so stretch it to fit evenings and weekends; the checklists below are the work inside each band.

### Phase 0 · Foundations (1 to 2 days)

- [ ] Repo with `uv`, `ruff`, `pytest`, `.gitignore` and `.env.example`
- [ ] FastAPI skeleton with `/health` and config loading from `config/*.yaml`
- [ ] `docker-compose.yml` with `api` and `ollama`, plus a `make models` target that pulls Phi-3 Mini, Llama 3.1 8B and Mistral 7B
- [ ] Pass-through endpoint that forwards a chat request to Ollama

**Done when:** `docker compose up` plus one curl returns an answer from a local model.

### Phase 1 · Routing core (4 to 5 days)

- [ ] `Provider` interface and `OpenAICompatibleProvider`, wired to Ollama and one cloud provider
- [ ] `models.yaml` loader: tiers, endpoints, prices, context windows
- [ ] Token estimate (`tiktoken`), task classifier (keyword rules) and complexity scorer with signal breakdown
- [ ] Routing policy: override, capability, context fit, then threshold, with model choice inside the tier (the privacy rule plugs in first once Phase 2 adds PII detection)
- [ ] `POST /v1/chat/completions` (non-streaming) with the `router` block, and the `POST /v1/route` dry run
- [ ] Table-driven unit tests for classifier, scorer and rule order (about 30 cases), with providers mocked

**Done when:** dry runs on 10 hand-picked prompts land in the expected tier, and real calls return routing metadata.

### Phase 2 · Aggregator and data (3 to 4 days)

- [ ] One response shape for every provider, plus the `X-Router-*` headers
- [ ] Guardrails: input size limit, regex PII detection (emails, phone, card and ID numbers) with masking and privacy routing, empty-answer check
- [ ] Timeouts, local-to-cloud fallback and one premium retry
- [ ] Postgres schema (`requests`, `feedback`, `eval_runs`) and background writes
- [ ] Cost, baseline cost and savings calculation
- [ ] `/v1/feedback`, `/v1/stats` and `/v1/models`

**Done when:** every request leaves a row with tokens, latency, cost and savings, and stopping Ollama mid-demo produces a logged fallback instead of an error.

### Phase 3 · Dashboard and playground (3 to 4 days)

- [ ] `/metrics` with Prometheus instrumentation, plus `prometheus.yml`
- [ ] Grafana provisioned from files: both data sources and the six-panel dashboard
- [ ] Streamlit playground: chat, a routing card per answer (tier, model, score bar, signals, latency, cost, savings), thumbs up and down, a force-tier toggle and dry-run mode
- [ ] `load_demo.py` to replay a mixed prompt set and fill the dashboard

**Done when:** after one `load_demo.py` run all six panels show data, and the playground explains every answer.

### Phase 4 · Evaluate and optimise (3 to 4 days)

- [ ] About 150 labelled prompts in `eval/prompts.jsonl` across all task types, including deliberate borderline cases
- [ ] `run_eval.py`: router accuracy, under- vs over-routing, local share, savings vs baseline, p95 latency per tier
- [ ] Quality check on 40 to 50 prompts: the premium model grades each local answer against its own (better, same, worse), plus a hand spot-check
- [ ] Threshold sweep from 20 to 45 through the free dry-run endpoint; pick the threshold and adjust weights
- [ ] Results written to `eval_runs` and `docs/results.md`
- [ ] Redis exact-match cache with a TTL

**Done when:** a results table and a cost-versus-accuracy chart justify the threshold in `routing.yaml`.

Under-routing (a complex prompt sent local) hurts answer quality, while over-routing only costs money. Report the two separately and tune to keep under-routing low.

### Phase 5 · Showcase polish (2 to 3 days)

- [ ] README: one-line pitch, the architecture image, a 60 to 90 second demo GIF, how routing works with one worked example, the results table, stack and decisions, what you would do next
- [ ] Screenshots of the dashboard and playground
- [ ] GitHub Actions running `ruff` and `pytest` on every push (no API keys needed, providers mocked)
- [ ] Short ADRs in `docs/adr/` for the main decisions
- [ ] Final sweep: pinned versions, and no keys or real prompts anywhere in git history

**Done when:** someone landing on the repo understands what it does and what it achieved within two minutes.

## Demo script

Seven steps show every capability in one pass; steps 1, 3, 6 and 7 are enough for the README GIF.

| # | Do this | What it shows |
| --- | --- | --- |
| 1 | Ask "Hi! What can you do?" | Local answer from Phi-3 Mini, near-instant, $0 |
| 2 | Ask "Summarise this article in 3 bullets: …" | Local answer from Mistral 7B, with the savings on the routing card |
| 3 | Ask "Compare Postgres and MongoDB for event sourcing, step by step." | Premium model, score 45 and its breakdown |
| 4 | With privacy mode on in routing.yaml, ask "Rewrite this note to jane@example.com so it sounds friendlier: …" | PII detected, kept local, email masked in the stored preview |
| 5 | Repeat step 2 | Cache hit with near-zero latency |
| 6 | Run `docker compose stop ollama`, then repeat step 1 | Automatic fallback to the cloud, flagged on the card |
| 7 | Switch to Grafana | Savings, routing mix, p95 latency, feedback and eval accuracy |

Before a live demo, preload the models and set `OLLAMA_KEEP_ALIVE=-1`. By default Ollama unloads a model after 5 idle minutes ([Ollama FAQ](https://docs.ollama.com/faq)), and the reload makes the first answer slow.

## Risks and mitigations

The biggest risk is a slow or wrong local answer in front of an audience; the rest are about cost, credibility and scope.

| Risk | Mitigation |
| --- | --- |
| Local models are slow on CPU | Use the small models for the demo, run Ollama natively on a Mac, keep models loaded, and rely on timeouts plus fallback |
| Keyword rules misroute prompts | The labelled eval set and threshold sweep, the override header, and the logged signal breakdown for debugging |
| Cloud spend during development | Test routing through the free dry-run endpoint, keep `max_tokens` low, cache, and use a budget alert where the provider offers one |
| Savings figures look inflated | Premium default as the baseline, numbers labelled as estimates, method written up in the README |
| Model IDs and prices go stale | Everything lives in `models.yaml` with a `priced_on` date; nothing is hardcoded |
| Keys or personal data end up on GitHub | `.env` gitignored, masked prompt previews only, and a history check before the repo goes public |
| Scope creep | Nothing from the stretch list starts before Phase 5's gate |

## Stretch goals after v1

None of these start before Phase 5's gate; each makes a good follow-up commit once the showcase is done.

- Learned classifier: embeddings plus logistic regression trained on the eval set, compared head to head with the keyword rules
- Cascade routing: answer locally first and escalate when the answer fails a quality check
- Streaming responses (server-sent events) through the router
- Semantic cache keyed on embedding similarity
- Kubernetes: the same stack on a local kind or k3d cluster
- Budget guard: a daily spend cap that forces local routing once reached
- Native Anthropic adapter, and a code-specialised local model for code prompts

## Sources

- [Ollama: OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama FAQ](https://docs.ollama.com/faq): GPU in Docker, default context window, model keep-alive
- [Gemini API: OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [Claude API: OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
