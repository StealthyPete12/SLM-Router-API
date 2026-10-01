"""Replays the demo prompt set through a running router to fill the dashboard.

    uv run python -m eval.load_demo                 # the main set, against http://localhost:8000
    uv run python -m eval.load_demo --set fallback  # while the local backend is stopped
    make demo                                       # both, stopping the local backend in between

This is demo data, not an evaluation: the prompts in eval/demo_prompts.jsonl are
fixed fixtures (fictional names, example.com addresses, 555 numbers), the
thumbs up and down are synthetic and say so in their comment, and every request
carries `X-Router-Client: demo-loader` so dashboards can filter it out. Router
accuracy needs labelled prompts and arrives with eval/run_eval.py in Phase 4.

The API key comes from --key or ROUTER_API_KEY and is never printed.
Exit codes: 0 all good, 1 some requests failed, 2 the run could not start.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

PROMPTS = Path(__file__).resolve().parent / "demo_prompts.jsonl"
CLIENT_LABEL = "demo-loader"
SETS = ("main", "fallback")
FEEDBACK_COMMENT = "synthetic demo rating from eval/load_demo.py ({id}); not a user judgement"


class DemoDataError(ValueError):
    """The prompt file is malformed."""


class PreflightError(Exception):
    """The API is not ready for a demo run; the message says what to fix."""


@dataclass(frozen=True)
class DemoPrompt:
    id: str
    set: str
    prompt: str
    tier: str | None = None  # forced with X-Router-Tier
    feedback: int | None = None  # synthetic rating sent after the answer
    note: str | None = None


@dataclass
class Summary:
    sent: int = 0
    ok: int = 0
    tiers: Counter[str] = field(default_factory=Counter)
    models: Counter[str] = field(default_factory=Counter)
    tasks: Counter[str] = field(default_factory=Counter)
    fallbacks: int = 0
    pii: int = 0
    feedback: Counter[str] = field(default_factory=Counter)
    savings_usd: float = 0.0
    unpriced: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.sent - self.ok


def load_prompts(path: Path = PROMPTS, sets: tuple[str, ...] = SETS) -> list[DemoPrompt]:
    """Parse and validate the JSONL prompt file; keep the prompts in `sets`, in file order."""
    prompts: list[DemoPrompt] = []
    seen: set[str] = set()
    allowed = {f for f in DemoPrompt.__dataclass_fields__}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DemoDataError(f"{where}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DemoDataError(f"{where}: expected an object")
        unknown = set(row) - allowed
        if unknown:
            raise DemoDataError(f"{where}: unknown fields {sorted(unknown)}")
        for name in ("id", "set", "prompt"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise DemoDataError(f"{where}: `{name}` must be a non-empty string")
        if row["id"] in seen:
            raise DemoDataError(f"{where}: duplicate id {row['id']!r}")
        if row["set"] not in SETS:
            raise DemoDataError(f"{where}: `set` must be one of {SETS}")
        if row.get("tier") not in (None, "local", "premium"):
            raise DemoDataError(f"{where}: `tier` must be local or premium")
        if row.get("feedback") not in (None, 1, -1):
            raise DemoDataError(f"{where}: `feedback` must be 1 or -1")
        seen.add(row["id"])
        prompt = DemoPrompt(**row)
        if prompt.set in sets:
            prompts.append(prompt)
    if not prompts:
        raise DemoDataError(f"{path.name}: no prompts in sets {sets}")
    return prompts


def preflight(client: httpx.Client, prompts: list[DemoPrompt], fallback_run: bool) -> list[str]:
    """Check the API before sending anything. Returns warnings; raises PreflightError."""
    try:
        health = client.get("/health").json()
    except (httpx.HTTPError, ValueError) as exc:
        message = f"cannot reach the router at {client.base_url} ({type(exc).__name__})"
        raise PreflightError(f"{message}. Is it up? `make up`") from exc
    stats = client.get("/v1/stats")
    if stats.status_code == 401:
        raise PreflightError("the API key was rejected: set ROUTER_API_KEY to the router's key")
    if stats.status_code == 503 and not health.get("auth_configured"):
        raise PreflightError(
            "the router has no ROUTER_API_KEY configured; set it in .env and `make up`"
        )

    warnings = []
    premium = health.get("premium", {})
    if not premium.get("configured"):
        # Premium is needed for premium-routed prompts and for every fallback.
        raise PreflightError(
            "the premium model is not configured (missing: "
            f"{', '.join(premium.get('missing', []))}). Set PREMIUM_* in .env, or use the "
            "labelled mock model: STACK_MODE=mock in .env, then `make up`"
        )
    if premium.get("provider") == "mock":
        warnings.append(
            "MOCK MODE: premium answers are simulated and prices illustrative; "
            "these numbers are demo data, not results"
        )
    ollama = health.get("ollama", {})
    if fallback_run and ollama.get("reachable"):
        warnings.append(
            "the local backend still answers, so these prompts will probably not fall back; "
            "stop it first (`make demo` does this for you)"
        )
    if not fallback_run and not ollama.get("reachable"):
        warnings.append("the local backend is unreachable: local prompts will fall back to premium")
    elif not fallback_run and ollama.get("models_missing"):
        warnings.append(f"local models not pulled: {', '.join(ollama['models_missing'])}")
    storage = health.get("storage", {})
    if storage.get("backend") == "memory":
        warnings.append("the API has no DATABASE_URL: rows stay in its memory, not in Postgres")
    elif storage and not storage.get("reachable"):
        warnings.append("Postgres is unreachable: the dashboard will not see these requests")
    return warnings


def send(client: httpx.Client, prompt: DemoPrompt, max_tokens: int) -> dict[str, Any]:
    headers = {"X-Router-Tier": prompt.tier} if prompt.tier else {}
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": prompt.prompt}],
            "max_tokens": max_tokens,
        },
        headers=headers,
    )
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:200]}")
    return response.json()


def run(
    client: httpx.Client,
    prompts: list[DemoPrompt],
    max_tokens: int,
    out: Any = sys.stdout,  # noqa: ANN401 - any writable text stream
) -> Summary:
    summary = Summary()
    width = max(len(p.id) for p in prompts)
    for index, prompt in enumerate(prompts, 1):
        summary.sent += 1
        prefix = f"[{index:>3}/{len(prompts)}] {prompt.id:<{width}}"
        try:
            body = send(client, prompt, max_tokens)
        except (httpx.HTTPError, RuntimeError) as exc:
            summary.failures.append(f"{prompt.id}: {exc}")
            print(f"{prefix}  FAILED {exc}", file=out, flush=True)
            continue
        router = body["router"]
        summary.ok += 1
        summary.tiers[router["tier"]] += 1
        summary.models[router["model_id"]] += 1
        summary.tasks[router["task_type"]] += 1
        summary.fallbacks += bool(router.get("fallback_from"))
        summary.pii += bool(router.get("pii_detected"))
        if router.get("savings_usd") is None:
            summary.unpriced += 1
        else:
            summary.savings_usd += router["savings_usd"]
        flags = [
            f"fallback from {router['fallback_from']}" if router.get("fallback_from") else "",
            "forced" if router["reason"] == "caller_override" else "",
            "PII" if router.get("pii_detected") else "",
        ]
        savings = router.get("savings_usd")
        print(
            f"{prefix}  {router['tier']:<8}{router['model_id']:<16}{router['task_type']:<10}"
            f"score {router['complexity_score']:>3}  {router.get('latency_ms', 0):>6} ms  "
            f"saved {'unknown' if savings is None else f'${savings:.6f}'}"
            f"{'  [' + ', '.join(f for f in flags if f) + ']' if any(flags) else ''}",
            file=out,
            flush=True,
        )
        if prompt.feedback is not None:
            rating = client.post(
                "/v1/feedback",
                json={
                    "request_id": body["id"],
                    "rating": prompt.feedback,
                    "comment": FEEDBACK_COMMENT.format(id=prompt.id),
                },
            )
            if rating.is_error:
                summary.failures.append(f"{prompt.id}: feedback HTTP {rating.status_code}")
            else:
                summary.feedback["up" if prompt.feedback == 1 else "down"] += 1
    return summary


def report(summary: Summary, elapsed_s: float, out: Any = sys.stdout) -> None:  # noqa: ANN401
    def counts(counter: Counter[str]) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(counter.items())) or "none"

    savings = (
        "unknown (no prices configured)"
        if summary.unpriced == summary.ok
        else f"${summary.savings_usd:.6f} (estimated"
        + (f"; {summary.unpriced} answers unpriced)" if summary.unpriced else ")")
    )
    lines = [
        "",
        f"Sent {summary.sent} prompts in {elapsed_s:.1f}s: {summary.ok} answered, "
        f"{summary.failed} failed",
        f"  tiers:     {counts(summary.tiers)}",
        f"  models:    {counts(summary.models)}",
        f"  tasks:     {counts(summary.tasks)}",
        f"  fallbacks: {summary.fallbacks}   PII detected: {summary.pii}",
        f"  feedback:  {summary.feedback['up']} up, {summary.feedback['down']} down (synthetic)",
        f"  savings:   {savings}",
    ]
    lines += [f"  failure:   {failure}" for failure in summary.failures]
    print("\n".join(lines), file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill the dashboard with demo traffic.")
    parser.add_argument("--url", default=os.environ.get("ROUTER_API_URL", "http://localhost:8000"))
    parser.add_argument("--key", default=None, help="default: ROUTER_API_KEY")
    parser.add_argument("--set", choices=[*SETS, "all"], default="main")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=128, help="per answer; keep it low")
    parser.add_argument("--timeout", type=float, default=180, help="seconds per request")
    args = parser.parse_args(argv)

    key = args.key or os.environ.get("ROUTER_API_KEY")
    if not key:
        print("error: no API key; set ROUTER_API_KEY (e.g. in .env) or pass --key", file=sys.stderr)
        return 2
    sets = SETS if args.set == "all" else (args.set,)
    try:
        prompts = load_prompts(args.prompts, sets)
    except (OSError, DemoDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {key}", "X-Router-Client": CLIENT_LABEL}
    with httpx.Client(
        base_url=args.url.rstrip("/"), headers=headers, timeout=args.timeout
    ) as client:
        try:
            warnings = preflight(client, prompts, fallback_run=sets == ("fallback",))
        except PreflightError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for warning in warnings:
            print(f"note: {warning}")
        print(f"Sending {len(prompts)} prompts ({', '.join(sets)}) to {args.url}\n")
        started = time.perf_counter()
        summary = run(client, prompts, args.max_tokens)
    report(summary, time.perf_counter() - started)

    if sets == ("fallback",) and summary.ok and not summary.fallbacks:
        print("error: no request fell back; was the local backend stopped?", file=sys.stderr)
        return 1
    return 1 if summary.failures else 0


if __name__ == "__main__":
    sys.exit(main())
