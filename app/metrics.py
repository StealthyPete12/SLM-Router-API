"""Prometheus metrics, served at GET /metrics and scraped by infra/prometheus/prometheus.yml.

Each app instance owns its registry, so tests can build many apps in one process.
The `model` label is the models.yaml id (phi3-mini, premium-default, ...).
"""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

# Seconds. Wide on purpose: a cached or mocked answer takes milliseconds, a local
# model on CPU can take a minute.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "router_requests",
            "Chat completions handled, by the tier and model that answered (or failed).",
            ["tier", "model", "reason", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "router_latency_seconds",
            "End-to-end latency of successful chat completions, fallback included.",
            ["tier", "model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.tokens = Counter(
            "router_tokens",
            "Tokens processed, as reported by the provider (estimated when it reports none).",
            ["model", "direction"],
            registry=self.registry,
        )
        self.savings = Counter(
            "router_savings_usd",
            "Estimated USD saved against the baseline model (priced requests only).",
            registry=self.registry,
        )
        self.cost = Counter(
            "router_cost_usd",
            "Estimated USD spent (priced requests only).",
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "router_fallbacks",
            "Local answers that failed and were retried on the premium model.",
            ["from_model", "cause"],
            registry=self.registry,
        )
        self.feedback = Counter(
            "router_feedback",
            "Feedback submissions.",
            ["rating"],
            registry=self.registry,
        )
        self.cache = Counter(
            "router_cache",
            "Exact-match cache operations: lookups (hit, miss, error, bypass) and stores "
            "(ok, error).",
            ["operation", "result"],
            registry=self.registry,
        )
        self.storage_errors = Counter(
            "router_storage_errors",
            "Failed writes or reads against the analytics store.",
            ["operation"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
