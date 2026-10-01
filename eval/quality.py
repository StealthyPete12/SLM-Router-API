"""Answer-quality check: the premium model grades local answers against its own. PAID.

    uv run python -m eval.run_eval quality --estimate-only                 # free: the cost cap
    uv run python -m eval.run_eval quality --confirm-paid --max-cost-usd 1  # calls the cloud
    make eval-quality-estimate / make eval-quality-paid CONFIRM_PAID=yes

For each prompt in the quality subset (`quality: true` in eval/prompts.jsonl):

1. POST /v1/route (free) asks where the router sends it. Prompts routed premium
   need no grading: the router already gives them the premium answer.
2. The local answer: /v1/chat/completions with X-Router-Tier: local, so a local
   failure is reported instead of silently falling back to the cloud.
3. The premium reference: the same request with X-Router-Tier: premium.
4. The grade: the premium model sees both answers as "A" and "B" in a seeded random
   order (against position bias) and returns JSON {"verdict": "A" | "B" | "same",
   "reason": ...}, mapped back to the local answer being better, same or worse.

Everything goes through the running router, so provider keys stay in the API
container and every call is logged (X-Router-Client: eval-quality / eval-grader).
Calls send Cache-Control: no-cache, so latencies are real model calls.

This is model-based grading, not ground truth: the grader is one of the two
contestants, may prefer its own style, and sees no reference answer. A blind hand
spot-check (eval/results/spot_check.md and .csv) measures how far to trust it.
quality_score = answers graded better or same / answers graded, in percent.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import load_config
from app.router.tokens import TokenEstimator
from app.storage.memory import percentile_cont
from eval.dataset import PROMPTS, DatasetError, EvalPrompt, dataset_sha256, load_dataset

PREMIUM_MODEL_ID = "premium-default"
CLIENT_ANSWER = "eval-quality"
CLIENT_GRADER = "eval-grader"
SPOT_CHECK_SIZE = 12
GRADER_SYSTEM = (
    "You are a strict, impartial evaluator of AI assistant answers. You judge substance, "
    "not style or length."
)
GRADER_TEMPLATE = """Two assistants answered the same request. Decide which answer serves the \
request better. Judge correctness first, then completeness against what was asked, then \
clarity. Do not reward length for its own sake. If neither is clearly better, say "same".

<request>
{request}
</request>

<answer_a>
{a}
</answer_a>

<answer_b>
{b}
</answer_b>

Reply with JSON only, exactly: {{"verdict": "A" | "B" | "same", "reason": "<one sentence>"}}"""
GRADER_OVERHEAD_TOKENS = 220  # template and system prompt, rounded up


class QualityError(Exception):
    """The quality check cannot run; the message says what to fix."""


@dataclass(frozen=True)
class Settings:
    max_tokens: int
    grader_max_tokens: int
    seed: int
    max_cost_usd: float


def local_position(seed: int, prompt_id: str) -> str:
    """Where the local answer goes ("A" or "B"), fixed per prompt and seed."""
    return "A" if random.Random(f"{seed}:{prompt_id}").random() < 0.5 else "B"


def parse_verdict(raw: str, local_is: str) -> tuple[str, str]:
    """Map the grader's JSON to (better | same | worse | invalid, reason)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
        verdict = str(data["verdict"]).strip()
        reason = str(data.get("reason", "")).strip()
    except (ValueError, KeyError, TypeError):
        return "invalid", "grader reply was not the requested JSON"
    if verdict.lower() == "same":
        return "same", reason
    if verdict.upper() in ("A", "B"):
        return ("better" if verdict.upper() == local_is else "worse"), reason
    return "invalid", f"unknown verdict {verdict!r}"


def request_text(prompt: EvalPrompt) -> str:
    return "\n\n".join(f"[{role}] {content}" for role, content in prompt.messages)


def estimate_cost(
    prompts: list[EvalPrompt],
    tokens: TokenEstimator,
    prices: tuple[float, float] | None,
    s: Settings,
) -> dict[str, Any]:
    """Upper bound if every prompt is graded: one premium answer and one grading call each."""
    input_tokens = output_tokens = 0
    for p in prompts:
        prompt_tokens = tokens.estimate_messages([c for _, c in p.messages])
        cap = min(p.max_tokens or s.max_tokens, s.max_tokens)
        input_tokens += prompt_tokens  # premium reference
        output_tokens += cap
        input_tokens += prompt_tokens + 2 * cap + GRADER_OVERHEAD_TOKENS  # grader
        output_tokens += s.grader_max_tokens
    usd = None
    if prices is not None:
        usd = round((input_tokens * prices[0] + output_tokens * prices[1]) / 1e6, 4)
    return {
        "prompts": len(prompts),
        "premium_calls_max": 2 * len(prompts),
        "input_tokens_max": input_tokens,
        "output_tokens_max": output_tokens,
        "usd_max": usd,
    }


class RouterClient:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def _post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        response = self.client.post(path, json=body, headers=headers)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise QualityError(f"{path} answered HTTP {response.status_code}: {str(detail)[:300]}")
        return response.json()

    def route(self, prompt: EvalPrompt) -> dict[str, Any]:
        return self._post("/v1/route", prompt.payload(), {})["router"]

    def answer(self, prompt: EvalPrompt, tier: str, max_tokens: int) -> dict[str, Any]:
        body = prompt.payload() | {"max_tokens": min(prompt.max_tokens or max_tokens, max_tokens)}
        headers = {
            "X-Router-Tier": tier,
            "X-Router-Client": CLIENT_ANSWER,
            "Cache-Control": "no-cache",
        }
        return self._post("/v1/chat/completions", body, headers)

    def grade(self, prompt: EvalPrompt, a: str, b: str, max_tokens: int) -> dict[str, Any]:
        body = {
            "model": PREMIUM_MODEL_ID,
            "messages": [
                {"role": "system", "content": GRADER_SYSTEM},
                {
                    "role": "user",
                    "content": GRADER_TEMPLATE.format(request=request_text(prompt), a=a, b=b),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"X-Router-Client": CLIENT_GRADER, "Cache-Control": "no-cache"}
        return self._post("/v1/chat/completions", body, headers)


def preflight(client: httpx.Client) -> dict[str, Any]:
    """Check the router can run a quality pass; return what it reports about itself."""
    try:
        health = client.get("/health").json()
        models = client.get("/v1/models")
    except (httpx.HTTPError, ValueError) as exc:
        raise QualityError(f"cannot reach the router at {client.base_url}: {exc!r}") from exc
    if models.status_code in (401, 503):
        raise QualityError("the router rejected the API key; set ROUTER_API_KEY")
    cards = {m["id"]: m for m in models.json()["data"]}
    premium = cards.get(PREMIUM_MODEL_ID)
    if premium is None or not premium.get("configured"):
        raise QualityError(
            f"{PREMIUM_MODEL_ID} is not configured: set PREMIUM_MODEL and PREMIUM_API_KEY in .env"
        )
    if not health.get("ollama", {}).get("reachable"):
        raise QualityError("the local backend is unreachable: start it (`make up`, `make models`)")
    mock = any(card.get("owned_by") == "mock" for card in cards.values())
    prices = None
    if premium.get("usd_per_1m_input") is not None and premium.get("usd_per_1m_output") is not None:
        prices = (float(premium["usd_per_1m_input"]), float(premium["usd_per_1m_output"]))
    return {"mock": mock, "prices": prices, "premium_model": premium.get("model")}


def _content(response: dict[str, Any]) -> str:
    return "".join(c["message"].get("content") or "" for c in response["choices"])


def evaluate_item(rc: RouterClient, prompt: EvalPrompt, s: Settings) -> dict[str, Any]:
    item: dict[str, Any] = {"id": prompt.id, "task": prompt.task, "expected": prompt.expected_tier}
    decision = rc.route(prompt)
    item |= {
        "routed": decision["tier"],
        "score": decision["complexity_score"],
        "threshold": decision["threshold"],
    }
    if decision["tier"] != "local":
        return item | {"status": "routed_premium"}
    try:
        local = rc.answer(prompt, "local", s.max_tokens)
        premium = rc.answer(prompt, "premium", s.max_tokens)
        local_is = local_position(s.seed, prompt.id)
        local_text, premium_text = _content(local), _content(premium)
        a, b = (local_text, premium_text) if local_is == "A" else (premium_text, local_text)
        graded = rc.grade(prompt, a, b, s.grader_max_tokens)
    except QualityError as exc:
        return item | {"status": "failed", "error": str(exc)}
    raw = _content(graded)
    verdict, reason = parse_verdict(raw, local_is)
    return item | {
        "status": "graded",
        "local_model": local["router"]["model_id"],
        "premium_model": premium["model"],
        "local_answer": local_text,
        "premium_answer": premium_text,
        "local_position": local_is,
        "grader_raw": raw,
        "verdict": verdict,
        "reason": reason,
        "local_latency_ms": local["router"].get("latency_ms"),
        "premium_latency_ms": premium["router"].get("latency_ms"),
        "cost_usd": sum(r["router"].get("cost_usd") or 0 for r in (local, premium, graded)),
        "request_ids": [local["id"], premium["id"], graded["id"]],
    }


def summarise_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [i for i in items if i["status"] == "graded"]
    verdicts = {v: sum(i["verdict"] == v for i in graded) for v in ("better", "same", "worse")}
    invalid = sum(i["verdict"] == "invalid" for i in graded)
    judged = sum(verdicts.values())

    def p95(key: str) -> float | None:
        values = [float(i[key]) for i in graded if i.get(key) is not None]
        result = percentile_cont(values, 0.95)
        return None if result is None else round(result, 1)

    return {
        "selected": len(items),
        "routed_local": sum(i.get("routed") == "local" for i in items),
        "routed_premium": sum(i["status"] == "routed_premium" for i in items),
        "failed": sum(i["status"] == "failed" for i in items),
        "graded": judged,
        "invalid": invalid,
        **verdicts,
        "quality_score_pct": round(100 * (verdicts["better"] + verdicts["same"]) / judged, 2)
        if judged
        else None,
        "p95_local_latency_ms": p95("local_latency_ms"),
        "p95_premium_latency_ms": p95("premium_latency_ms"),
        "latency_samples": len(graded),
        "cost_usd": round(sum(i.get("cost_usd") or 0 for i in items), 6),
    }


def write_spot_check(results: Path, data: dict[str, Any], prompts: dict[str, EvalPrompt]) -> None:
    """A blind sample for a person to grade: A/B as the grader saw them, no verdicts shown."""
    graded = [i for i in data["items"] if i["status"] == "graded" and i["verdict"] != "invalid"]
    sample = sorted(
        random.Random(data["settings"]["seed"]).sample(graded, min(SPOT_CHECK_SIZE, len(graded))),
        key=lambda i: i["id"],
    )
    lines = [
        "# Answer-quality spot check (blind)",
        "",
        "For each prompt, decide which answer serves the request better: A, B or same.",
        "Write it in the `human_verdict` column of spot_check.csv, then run",
        "`make eval-spot-check`. The model's verdicts are not shown here on purpose.",
        "",
    ]
    for item in sample:
        a, b = (
            (item["local_answer"], item["premium_answer"])
            if item["local_position"] == "A"
            else (item["premium_answer"], item["local_answer"])
        )
        lines += [
            f"## {item['id']}",
            "",
            "**Request**",
            "",
            "```text",
            request_text(prompts[item["id"]])[:4000],
            "```",
            "",
            "**Answer A**",
            "",
            "```text",
            a,
            "```",
            "",
            "**Answer B**",
            "",
            "```text",
            b,
            "```",
            "",
        ]
    (results / "spot_check.md").write_text("\n".join(lines), encoding="utf-8")
    with (results / "spot_check.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "human_verdict", "note"])
        for item in sample:
            writer.writerow([item["id"], "", ""])


def spot_check_summary(results: Path) -> dict[str, Any]:
    data = json.loads((results / "quality.json").read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}
    rows = list(csv.DictReader((results / "spot_check.csv").open(encoding="utf-8")))
    checked = agree = 0
    disagreements = []
    for row in rows:
        human = row["human_verdict"].strip()
        if not human:
            continue
        item = by_id[row["id"]]
        mapped, _ = parse_verdict(json.dumps({"verdict": human}), item["local_position"])
        if mapped == "invalid":
            raise QualityError(f"{row['id']}: human_verdict must be A, B or same, got {human!r}")
        checked += 1
        if mapped == item["verdict"]:
            agree += 1
        else:
            disagreements.append(
                {"id": row["id"], "model": item["verdict"], "human": mapped, "note": row["note"]}
            )
    return {
        "sampled": len(rows),
        "checked": checked,
        "agree": agree,
        "agreement_pct": round(100 * agree / checked, 1) if checked else None,
        "disagreements": disagreements,
    }


def spot_check_main(results: Path) -> int:
    if not (results / "spot_check.csv").exists():
        print("error: no spot_check.csv yet; run the paid quality check first", file=sys.stderr)
        return 2
    try:
        summary = spot_check_summary(results)
    except QualityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (results / "spot_check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not summary["checked"]:
        print(f"{summary['sampled']} prompts sampled, none graded by hand yet (spot_check.csv)")
        return 1
    print(
        f"hand-checked {summary['checked']} of {summary['sampled']}: the grader agreed on "
        f"{summary['agree']} ({summary['agreement_pct']}%)"
    )
    for d in summary["disagreements"]:
        print(f"  {d['id']}: model said {d['model']}, you said {d['human']}")
    return 0


def main(argv: list[str], results: Path, dataset: Path = PROMPTS) -> int:
    parser = argparse.ArgumentParser(prog="run_eval quality", description="PAID quality check.")
    parser.add_argument("--url", default=os.environ.get("ROUTER_API_URL", "http://localhost:8000"))
    parser.add_argument("--estimate-only", action="store_true", help="print the cost cap and stop")
    parser.add_argument("--confirm-paid", action="store_true", help="allow cloud calls that cost")
    parser.add_argument("--max-cost-usd", type=float, default=1.00, help="refuse above this cap")
    parser.add_argument("--allow-unpriced", action="store_true", help="run with no premium prices")
    parser.add_argument("--allow-mock", action="store_true", help="smoke-test on mock models")
    parser.add_argument("--max-tokens", type=int, default=400, help="cap per answer")
    parser.add_argument("--grader-max-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N quality prompts")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    s = Settings(args.max_tokens, args.grader_max_tokens, args.seed, args.max_cost_usd)

    key = os.environ.get("ROUTER_API_KEY")
    if not key:
        print("error: set ROUTER_API_KEY (the router's key, never a provider key)", file=sys.stderr)
        return 2
    config = load_config()
    try:
        prompts = [p for p in load_dataset(dataset, config) if p.quality][: args.limit]
    except DatasetError as exc:
        print(
            f"error: invalid dataset; run `make eval-validate` ({exc.problems[0]})", file=sys.stderr
        )
        return 2

    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=args.timeout) as http:
        try:
            info = preflight(http)
        except QualityError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        estimate = estimate_cost(prompts, TokenEstimator(config.routing.tokens), info["prices"], s)
        cap = (
            "unknown (premium unpriced)"
            if estimate["usd_max"] is None
            else f"${estimate['usd_max']}"
        )
        print(
            f"{len(prompts)} quality prompts; at most {estimate['premium_calls_max']} premium "
            f"calls, {estimate['input_tokens_max']} input and "
            f"{estimate['output_tokens_max']} output tokens: cost cap {cap}"
        )
        if info["mock"]:
            print("MOCK MODELS: answers are simulated; the output is a smoke test, not a result")
        if args.estimate_only:
            return 0
        if info["mock"] and not args.allow_mock:
            print(
                "error: the router serves mock models; pass --allow-mock for a smoke test",
                file=sys.stderr,
            )
            return 2
        if not info["mock"]:
            if not args.confirm_paid:
                print(
                    "error: this calls the paid premium model; pass --confirm-paid", file=sys.stderr
                )
                return 2
            if estimate["usd_max"] is None and not args.allow_unpriced:
                print(
                    "error: premium prices are unset, so the cost cannot be capped; set "
                    "PREMIUM_USD_PER_1M_* or pass --allow-unpriced",
                    file=sys.stderr,
                )
                return 2
            if estimate["usd_max"] is not None and estimate["usd_max"] > s.max_cost_usd:
                print(
                    f"error: cost cap {cap} exceeds --max-cost-usd {s.max_cost_usd}",
                    file=sys.stderr,
                )
                return 2

        rc = RouterClient(http)
        items = []
        for index, prompt in enumerate(prompts, 1):
            try:
                item = evaluate_item(rc, prompt, s)
            except QualityError as exc:
                item = {"id": prompt.id, "task": prompt.task, "status": "failed", "error": str(exc)}
            items.append(item)
            print(
                f"[{index:>2}/{len(prompts)}] {prompt.id:<15} {item['status']:<15} "
                f"{item.get('verdict', '')}"
            )

    data = {
        "kind": "quality_eval",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": dataset.name, "sha256": dataset_sha256(dataset)},
        # The running router's threshold, which may differ from this checkout's routing.yaml.
        "threshold": next((i["threshold"] for i in items if "threshold" in i), None),
        "mock": info["mock"],
        "premium_model": info["premium_model"],
        "settings": vars(s) | {"grader": PREMIUM_MODEL_ID},
        "estimate": estimate,
        "summary": summarise_items(items),
        "items": items,
    }
    results.mkdir(parents=True, exist_ok=True)
    name = "quality-mock.json" if info["mock"] else "quality.json"
    (results / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if not info["mock"]:
        write_spot_check(results, data, {p.id: p for p in prompts})
    summary = data["summary"]
    print(
        f"graded {summary['graded']}: {summary['better']} better, {summary['same']} same, "
        f"{summary['worse']} worse ({summary['invalid']} invalid, {summary['failed']} failed); "
        f"quality score {summary['quality_score_pct']}%; spent ${summary['cost_usd']}"
    )
    print(f"wrote {results / name}")
    return 1 if summary["failed"] else 0
