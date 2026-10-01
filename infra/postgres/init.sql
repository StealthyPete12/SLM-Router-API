-- Analytics schema for the SLM router. Run once by the postgres container on an
-- empty data volume (/docker-entrypoint-initdb.d); there is no migration tool.
-- To apply a changed schema to an existing stack: `make reset-data` (destructive).
--
-- The dashboard and the evaluation report read these same rows, so their numbers agree.
-- Prompts are never stored in full: only a PII-masked preview and a SHA-256 hash.

CREATE TABLE IF NOT EXISTS requests (
    id                text        PRIMARY KEY,  -- the request_id returned to the client
    created_at        timestamptz NOT NULL DEFAULT now(),
    client            text        NOT NULL DEFAULT 'api',  -- X-Router-Client, e.g. demo-loader
    task_type         text        NOT NULL,
    complexity_score  integer     NOT NULL CHECK (complexity_score BETWEEN 0 AND 100),
    threshold         integer     NOT NULL,
    signals           jsonb       NOT NULL,     -- score breakdown: task_base, input_length, ...
    tier              text        NOT NULL CHECK (tier IN ('local', 'premium')),  -- answered by
    reason            text        NOT NULL,     -- the routing decision, e.g. score_below_threshold
    model_id          text        NOT NULL,     -- models.yaml id that answered
    model             text,                     -- provider-facing model name
    provider          text        NOT NULL,     -- ollama, openai, ... or mock in mock mode
    input_tokens      integer     CHECK (input_tokens >= 0),
    output_tokens     integer     CHECK (output_tokens >= 0),
    latency_ms        integer     CHECK (latency_ms >= 0),
    -- Estimates in USD. NULL when the model (or the baseline) has no price in models.yaml.
    cost_usd          numeric(14, 8),
    baseline_cost_usd numeric(14, 8),
    savings_usd       numeric(14, 8),
    cache_hit         boolean     NOT NULL DEFAULT false,  -- exact-match cache (Phase 4)
    fallback_from     text,                     -- the local model id that failed, if any
    fallback_cause    text,                     -- timeout, unreachable, error, empty_answer, ...
    pii_detected      boolean     NOT NULL DEFAULT false,
    status            text        NOT NULL CHECK (status IN ('ok', 'error')),
    status_code       integer     NOT NULL,
    error             text,
    prompt_preview    text        NOT NULL,     -- last user message, PII-masked, 200 chars
    prompt_hash       text        NOT NULL      -- SHA-256 of every message
);

CREATE INDEX IF NOT EXISTS requests_created_at_idx ON requests (created_at);
CREATE INDEX IF NOT EXISTS requests_client_idx ON requests (client);

-- One rating per answer: a second vote for the same request replaces the first.
CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial   PRIMARY KEY,
    request_id  text        NOT NULL UNIQUE REFERENCES requests (id) ON DELETE CASCADE,
    rating      smallint    NOT NULL CHECK (rating IN (-1, 1)),
    comment     text,                           -- PII-masked
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz
);

CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON feedback (created_at);

-- One row per evaluation run. Written by eval/run_eval.py (Phase 4); empty until then.
CREATE TABLE IF NOT EXISTS eval_runs (
    id                bigserial   PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    threshold         integer     NOT NULL,
    dataset_size      integer     NOT NULL CHECK (dataset_size > 0),
    -- All percentages are 0 to 100.
    router_accuracy   numeric(5, 2),  -- prompts routed to their labelled tier
    under_routed_pct  numeric(5, 2),  -- complex prompts sent local (hurts quality)
    over_routed_pct   numeric(5, 2),  -- simple prompts sent premium (only costs money)
    local_share       numeric(5, 2),
    savings_pct       numeric(6, 2),  -- estimated, vs the all-premium baseline
    quality_score     numeric(5, 2),  -- graded local answers judged same or better
    notes             text
);

CREATE INDEX IF NOT EXISTS eval_runs_created_at_idx ON eval_runs (created_at);
