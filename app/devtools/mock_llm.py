"""A fake OpenAI-compatible model server for the mock and offline stacks. DEV ONLY.

It lets the whole stack (routing, fallback, logging, metrics, dashboards, the
playground) run without cloud credentials or downloaded models. Every answer
says it is simulated, latency is simulated from the settings below, and the
`provider` column of each stored request reads `mock`, so mock traffic can never
pass for real results.

    MOCK_TIER          local | premium: only changes the answer text and delays
    MOCK_MODELS        comma-separated model names it serves (others get 404)
    MOCK_DELAY_MS      base delay per answer
    MOCK_MS_PER_TOKEN  extra delay per output token

Run: uvicorn app.devtools.mock_llm:create_app --factory --port 8080
"""

import asyncio
import hashlib
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException

_FILLER = (
    "the router keeps simple prompts on a local model and sends demanding ones to the cloud "
    "this answer is simulated so the dashboard and playground can be exercised without real "
    "models every number it produces is labelled as mock data"
)
_WORDS = _FILLER.split(" ")


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _tokens(text: str) -> int:
    """Rough token count (4 characters each); good enough for a fake usage block."""
    return max(1, len(text) // 4)


def create_app() -> FastAPI:
    tier = os.environ.get("MOCK_TIER", "premium")
    models = [m.strip() for m in os.environ.get("MOCK_MODELS", "mock-premium").split(",") if m]
    base_delay = float(os.environ.get("MOCK_DELAY_MS", "600")) / 1000
    per_token = float(os.environ.get("MOCK_MS_PER_TOKEN", "5")) / 1000

    app = FastAPI(title=f"Mock {tier} LLM (development only)")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "tier": tier}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in models],
        }

    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model")
        if model not in models:
            raise HTTPException(404, f"model {model!r} not found")
        messages = body.get("messages") or []
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        seed = _seed(f"{model}:{prompt}")

        # Deterministic length: premium answers run longer, max_tokens caps both.
        words = (24 if tier == "local" else 60) + seed % (60 if tier == "local" else 140)
        if body.get("max_tokens"):
            words = min(words, max(1, int(body["max_tokens"] * 0.75)))
        filler = " ".join(_WORDS[(seed + i) % len(_WORDS)] for i in range(words))
        content = (
            f"[Simulated {tier} answer from {model}; mock mode, no model was called.] {filler}."
        )
        completion_tokens = _tokens(content)

        jitter = 0.7 + (seed % 61) / 100  # 0.7x to 1.3x
        await asyncio.sleep((base_delay + per_token * completion_tokens) * jitter)

        prompt_tokens = sum(_tokens(str(m.get("content") or "")) + 3 for m in messages) + 3
        return {
            "id": f"mock-{seed:08x}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app
