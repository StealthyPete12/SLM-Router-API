"""Phase 1 acceptance: hand-picked prompts and the route each one must get.

The tests run them through POST /v1/route (tests/test_acceptance.py). Run them
yourself, with no model called and no cloud cost:

    uv run python -m scripts.acceptance                  # in-process, no server needed
    uv run python -m scripts.acceptance --url http://localhost:8000 --key "$ROUTER_API_KEY"
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import httpx

ARTICLE = """\
Tea has been grown in the hills of southern China for thousands of years, and the plant itself \
is a modest evergreen shrub called Camellia sinensis. Left alone it becomes a small tree, but \
growers keep it pruned to waist height so that pickers can reach the young shoots. The leaves \
that end up in a cup are usually the top two leaves and a bud, picked by hand every week or two \
during the growing season.

The flavour of the finished tea depends on what happens after picking. Green tea is heated soon \
after harvest, which stops the leaves from oxidising and keeps their grassy, fresh character. \
Black tea is rolled and left to oxidise fully, turning the leaves dark and giving the drink its \
malty depth. Oolong sits between the two, with leaves bruised and partly oxidised, and white tea \
is simply withered and dried with as little handling as possible.

Climate matters as much as processing. Tea likes warm days, cool nights, steady rain and soil that \
drains well, which is why so many famous gardens cling to steep slopes. Plants grown at higher \
altitude grow more slowly, and many drinkers believe the slower growth concentrates the flavour. \
Darjeeling, in the foothills of the Himalayas, is the classic example: its first spring harvest \
is prized for a light, floral cup that fetches high prices at auction.

Tea travelled along trade routes long before it reached Europe. Monks and merchants carried \
compressed bricks of leaves across Central Asia, where the bricks were sometimes used as money. \
Dutch traders brought tea to Europe in the early seventeenth century, and within a few decades it \
had become fashionable in London coffee houses. By the nineteenth century, British demand was so \
large that the East India Company began growing tea in Assam and later in Ceylon, now Sri Lanka.

Those colonial estates changed the industry. Large plantations replaced small family gardens, and \
workers were brought in from other regions, often under harsh conditions that historians still \
write about today. Machines arrived for rolling and cutting leaves, which made black tea cheaper \
and more uniform. The tea bag, popularised in the early twentieth century, pushed the market \
further toward small, broken leaves that brew quickly.

Today tea is the most widely consumed drink in the world after water. China and India produce most \
of it, followed by Kenya, Sri Lanka and Turkey. Kenya is notable because almost all of its tea is \
grown by small farmers who sell their leaves to cooperative factories. Turkey drinks nearly all of \
the tea it grows, served strong and sweet in small tulip-shaped glasses throughout the day.

Habits around the drink vary enormously. In Japan the tea ceremony turns the preparation of \
powdered green tea into a slow, deliberate ritual. In Morocco green tea is brewed with fresh mint \
and plenty of sugar and poured from a height to create foam. In Britain milk is usually added, and \
an afternoon cup with biscuits remains a small daily ritual for millions of people.

Interest in speciality tea has grown again in recent years. Small producers sell single-estate \
leaves directly to drinkers, and tasting notes now read much like those for wine or coffee. At the \
same time, rising temperatures and less predictable rain are putting pressure on traditional \
growing regions, and some farmers are moving their gardens higher up the hills to keep the cool \
conditions the plant prefers.

For most people, though, tea is not a hobby but a pause in the day. A pot shared with friends, a \
mug carried to a desk or a flask poured on a cold platform all serve the same purpose, and that \
quiet, ordinary role may be the reason the drink has lasted so long.
"""

LONG_NOTES = "The team met to go over the quarterly numbers and agreed on next steps. " * 700

FIX_BUG = """\
Fix the bug in this function:
```python
def average(values):
    return sum(values) / len(values) + 1
```"""


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    tier: str
    model_id: str
    task_type: str
    score: int | None = None  # exact score, when the design fixes one
    min_score: int | None = None
    reason: str | None = None
    model: str = "auto"
    headers: dict[str, str] = field(default_factory=dict)


CASES = [
    # The four worked examples from the design document.
    Case("chitchat", "Hi! What can you do?", "local", "phi3-mini", "chitchat", score=0),
    Case(
        "summarise article",
        f"Summarise this article in 3 bullets:\n\n{ARTICLE}",
        "local",
        "mistral-7b",
        "summarise",
        score=20,
    ),
    Case("fix bug", FIX_BUG, "premium", "premium-default", "code", min_score=35),
    Case(
        "compare databases",
        "Compare Postgres and MongoDB for event sourcing, step by step.",
        "premium",
        "premium-default",
        "analysis",
        score=45,
    ),
    # Six more, one per remaining task family plus the hard rules.
    Case(
        "translate",
        "Translate into French: Good morning, the meeting starts at nine.",
        "local",
        "mistral-7b",
        "translate",
        score=15,
    ),
    Case(
        "rewrite",
        "Rewrite this message so it sounds more professional: hey cant make it tmrw, "
        "lets push the meeting to friday",
        "local",
        "mistral-7b",
        "rewrite",
        score=10,
    ),
    Case(
        "extract",
        "Extract all the people and dates from this text: Alice met Bob on 3 March 2024 "
        "in Paris, and they signed the lease with Carol on 12 April.",
        "local",
        "llama3-8b",
        "extract",
        score=15,
    ),
    Case(
        "creative",
        "Write a short poem about autumn leaves falling in the park.",
        "local",
        "llama3-8b",
        "creative",
        score=20,
    ),
    Case(
        "math",
        "Solve for x: 3x + 7 = 22, and explain why each step works.",
        "premium",
        "premium-default",
        "math",
        score=45,
    ),
    Case(
        "forced local",
        "Compare Postgres and MongoDB for event sourcing, step by step.",
        "local",
        "llama3-8b",
        "analysis",
        score=45,
        reason="caller_override",
        headers={"X-Router-Tier": "local"},
    ),
    # Extras: rule precedence and model choice.
    Case(
        "forced local, too long",
        f"Summarise these meeting notes:\n\n{LONG_NOTES}",
        "premium",
        "premium-default",
        "summarise",
        reason="context_window_exceeded",
        headers={"X-Router-Tier": "local"},
    ),
    Case(
        "simple question",
        "What is the capital of France?",
        "local",
        "phi3-mini",
        "qa",
        score=10,
    ),
    Case(
        "explicit model",
        "Compare Postgres and MongoDB for event sourcing, step by step.",
        "local",
        "mistral-7b",
        "analysis",
        reason="explicit_model",
        model="mistral-7b",
    ),
]


def check(case: Case, router: dict) -> list[str]:
    """Return the mismatches between a /v1/route `router` block and the expectation."""
    problems = []
    for key, expected in [
        ("tier", case.tier),
        ("model_id", case.model_id),
        ("task_type", case.task_type),
        ("reason", case.reason),
        ("complexity_score", case.score),
    ]:
        if expected is not None and router[key] != expected:
            problems.append(f"{key}={router[key]!r}, expected {expected!r}")
    if case.min_score is not None and router["complexity_score"] < case.min_score:
        problems.append(f"score {router['complexity_score']} < {case.min_score}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="a running API; default runs the app in-process")
    parser.add_argument("--key", default=os.environ.get("ROUTER_API_KEY", "acceptance"))
    args = parser.parse_args()

    if args.url:
        client: httpx.Client = httpx.Client(base_url=args.url, timeout=30)
    else:
        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.main import create_app

        os.environ.setdefault("LOG_LEVEL", "WARNING")
        client = TestClient(create_app(settings=Settings(router_api_key=args.key)))

    failures = 0
    print(f"{'case':<24}{'tier':<9}{'model':<17}{'task':<11}{'score':>5}  reason")
    with client:
        for case in CASES:
            response = client.post(
                "/v1/route",
                json={"model": case.model, "messages": [{"role": "user", "content": case.prompt}]},
                headers={"Authorization": f"Bearer {args.key}", **case.headers},
            )
            response.raise_for_status()
            r = response.json()["router"]
            problems = check(case, r)
            failures += bool(problems)
            print(
                f"{case.name:<24}{r['tier']:<9}{r['model_id']:<17}{r['task_type']:<11}"
                f"{r['complexity_score']:>5}  {r['reason']}"
                + (f"  MISMATCH: {'; '.join(problems)}" if problems else "")
            )
    print(f"\n{len(CASES) - failures}/{len(CASES)} as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
