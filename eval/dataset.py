"""The labelled evaluation set (eval/prompts.jsonl): loading and validation.

    uv run python -m eval.dataset              # validate and print the composition
    make eval-validate

One JSON object per line:

    id             stable id, "<task>-<nnn>"
    task           the task a person would say the prompt is (one of the ten task types)
    expected_tier  local | premium: the tier the prompt *needs* (see Labelling below)
    prompt         the user message, or
    messages       a full conversation [{role, content}], ending with a user message
    tags           edge-case tags from TAGS
    rationale      why the label is what it is
    max_tokens     optional, sent with the request
    forced_tier    optional X-Router-Tier (rows tagged `override`)
    privacy_mode   optional, true = evaluate with routing.yaml privacy_mode on
    quality        optional, true = part of the paid answer-quality subset

Labelling: `local` means a 7-8B local model (Phi-3 Mini, Mistral 7B, Llama 3.1 8B)
would likely give an adequate answer; `premium` means a small model would likely be
noticeably worse (multi-step reasoning, non-trivial code, judgement-heavy analysis,
long high-quality output, precise language work). Rows decided by a hard rule
(context window, privacy, override) carry the tier the policy must produce. Labels
were written before the router was run on the set and are not edited to fit it.

All personal data is synthetic: reserved domains (example.com), 555-01xx phone
numbers, published test card numbers and never-issued ID ranges. Validation
rejects anything else that looks like PII, and anything that looks like a secret.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from app.aggregator.guardrails import PiiScanner
from app.config import TASK_TYPES, TIERS, AppConfig, TaskType, Tier, load_config
from app.schemas import ChatCompletionRequest, ChatMessage

PROMPTS = Path(__file__).resolve().parent / "prompts.jsonl"

TAGS = frozenset(
    {
        "obvious_local",  # clearly fine on a local model
        "obvious_premium",  # clearly needs the premium model
        "borderline",  # a judgement call: either label is defensible
        "long_input",  # 200+ input tokens
        "multi_part",  # several questions or numbered requirements
        "reasoning_cue",  # "step by step", "explain why", "pros and cons", ...
        "output_demand",  # "detailed", 500+ words, a large max_tokens
        "code_block",  # a fenced code block in the prompt
        "deep_conversation",  # more than 6 user/assistant messages
        "context_window",  # too large for the local context window
        "override",  # a forced tier (X-Router-Tier)
        "privacy",  # synthetic PII in the prompt
        "implicit_task",  # the task is clear but none of its usual keywords appear
        "system_prompt",  # the conversation starts with a system message
    }
)
FIELDS = frozenset(
    {
        "id",
        "task",
        "expected_tier",
        "prompt",
        "messages",
        "tags",
        "rationale",
        "max_tokens",
        "forced_tier",
        "privacy_mode",
        "quality",
    }
)
ROLES = frozenset(get_args(ChatMessage.model_fields["role"].annotation)) - {"tool"}
ID_PATTERN = re.compile(r"^(?P<task>[a-z]+)-\d{3}$")

# Distribution the set must keep, so edits cannot quietly hollow it out.
MIN_ROWS, MAX_ROWS = 120, 200
MIN_PER_TASK = 8
MIN_TIER_SHARE = 0.20
QUALITY_ROWS = (40, 50)
MIN_TAGS = {
    "obvious_local": 10,
    "obvious_premium": 5,
    "borderline": 15,
    "long_input": 5,
    "multi_part": 5,
    "reasoning_cue": 5,
    "output_demand": 3,
    "code_block": 5,
    "deep_conversation": 3,
    "context_window": 1,
    "override": 2,
    "privacy": 2,
    "implicit_task": 5,
}

# Anything shaped like a credential fails validation outright.
SECRET_PATTERNS = {
    "OpenAI-style key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "Anthropic key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "Slack token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "password assignment": r"(?i)\bpassword\s*[:=]\s*\S{6,}",
}

# Synthetic PII must come from ranges reserved for examples, so it can never be real.
RESERVED_EMAIL = re.compile(r"@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|\w+\.(?:test|invalid))$")
TEST_CARDS = frozenset({"4111111111111111", "5555555555554444", "378282246310005"})


def _reserved(kind: str, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if kind == "email":
        return RESERVED_EMAIL.search(value) is not None
    if kind == "phone":
        return "55501" in digits  # North American 555-0100 to 555-0199 are fictional
    if kind == "card":
        return digits in TEST_CARDS
    if kind == "us_ssn":
        return digits.startswith(("9", "000", "666"))  # never issued
    if kind == "uk_nino":
        return value.replace(" ", "").upper().startswith("QQ")  # the official example prefix
    return False  # iban and any new kind: no synthetic range is known, so reject


class DatasetError(ValueError):
    """The dataset is malformed; `problems` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    task: TaskType
    expected_tier: Tier
    messages: tuple[tuple[str, str], ...]  # (role, content)
    tags: frozenset[str]
    rationale: str
    max_tokens: int | None = None
    forced_tier: Tier | None = None
    privacy_mode: bool = False
    quality: bool = False
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def text(self) -> str:
        return "\n".join(content for _, content in self.messages)

    @property
    def last_user(self) -> str:
        return next(content for role, content in reversed(self.messages) if role == "user")

    def request(self) -> ChatCompletionRequest:
        """The chat request this row stands for (model "auto")."""
        return ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role=role, content=content) for role, content in self.messages],
            max_tokens=self.max_tokens,
        )

    def payload(self) -> dict[str, Any]:
        """The same request as JSON, for the HTTP API."""
        body: dict[str, Any] = {
            "model": "auto",
            "messages": [{"role": r, "content": c} for r, c in self.messages],
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return body


def _row_problems(row: Any, where: str) -> list[str]:  # noqa: ANN401 - parsed JSON
    if not isinstance(row, dict):
        return [f"{where}: expected a JSON object"]
    problems = []
    unknown = set(row) - FIELDS
    if unknown:
        problems.append(f"{where}: unknown fields {sorted(unknown)}")

    row_id = row.get("id")
    match = ID_PATTERN.match(row_id) if isinstance(row_id, str) else None
    if match is None:
        problems.append(f"{where}: `id` must look like '<task>-<nnn>', got {row_id!r}")
    if row.get("task") not in TASK_TYPES:
        problems.append(f"{where}: `task` must be one of {list(TASK_TYPES)}")
    elif match and match["task"] != row["task"]:
        problems.append(f"{where}: id prefix {match['task']!r} does not match task {row['task']!r}")
    if row.get("expected_tier") not in TIERS:
        problems.append(f"{where}: `expected_tier` must be local or premium")

    has_prompt, has_messages = "prompt" in row, "messages" in row
    if has_prompt == has_messages:
        problems.append(f"{where}: give exactly one of `prompt` or `messages`")
    elif has_prompt and (not isinstance(row["prompt"], str) or not row["prompt"].strip()):
        problems.append(f"{where}: `prompt` must be a non-empty string")
    elif has_messages:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            problems.append(f"{where}: `messages` must be a non-empty list")
        else:
            for i, message in enumerate(messages):
                ok = (
                    isinstance(message, dict)
                    and set(message) == {"role", "content"}
                    and message["role"] in ROLES
                    and isinstance(message["content"], str)
                    and message["content"].strip()
                )
                if not ok:
                    problems.append(
                        f"{where}: messages[{i}] must be {{role, content}} with a role in "
                        f"{sorted(ROLES)} and non-empty content"
                    )
            if isinstance(messages[-1], dict) and messages[-1].get("role") != "user":
                problems.append(f"{where}: the last message must be from the user")

    tags = row.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        problems.append(f"{where}: `tags` must be a list of strings")
        tags = []
    elif sorted(set(tags) - TAGS):
        problems.append(f"{where}: unknown tags {sorted(set(tags) - TAGS)}")
    elif len(set(tags)) != len(tags):
        problems.append(f"{where}: repeated tags")
    if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
        problems.append(f"{where}: `rationale` must explain the label")

    max_tokens = row.get("max_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        problems.append(f"{where}: `max_tokens` must be a positive integer")
    forced = row.get("forced_tier")
    if forced is not None and forced not in TIERS:
        problems.append(f"{where}: `forced_tier` must be local or premium")
    if (forced is not None) != ("override" in tags):
        problems.append(f"{where}: `forced_tier` and the `override` tag go together")
    for flag in ("privacy_mode", "quality"):
        if not isinstance(row.get(flag, False), bool):
            problems.append(f"{where}: `{flag}` must be true or false")
    if row.get("privacy_mode") and "privacy" not in tags:
        problems.append(f"{where}: rows with privacy_mode need the `privacy` tag")
    if row.get("quality") and "privacy" in tags:
        problems.append(
            f"{where}: privacy rows cannot be in the quality subset (it calls the cloud)"
        )
    return problems


def _to_prompt(row: dict[str, Any]) -> EvalPrompt:
    if "prompt" in row:
        messages: tuple[tuple[str, str], ...] = (("user", row["prompt"]),)
    else:
        messages = tuple((m["role"], m["content"]) for m in row["messages"])
    return EvalPrompt(
        id=row["id"],
        task=row["task"],
        expected_tier=row["expected_tier"],
        messages=messages,
        tags=frozenset(row.get("tags", [])),
        rationale=row["rationale"],
        max_tokens=row.get("max_tokens"),
        forced_tier=row.get("forced_tier"),
        privacy_mode=row.get("privacy_mode", False),
        quality=row.get("quality", False),
    )


def content_problems(prompts: list[EvalPrompt], scanner: PiiScanner) -> list[str]:
    """Duplicates, secrets and PII outside the reserved synthetic ranges."""
    problems = []
    seen: dict[str, str] = {}
    for p in prompts:
        normalised = " ".join(f"{r}:{c}" for r, c in p.messages).lower().split()
        key = hashlib.sha256(" ".join(normalised).encode()).hexdigest()
        if key in seen:
            problems.append(f"{p.id}: same messages as {seen[key]}")
        seen.setdefault(key, p.id)

        for name, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, p.text):
                problems.append(f"{p.id}: contains something that looks like a {name}")
        kinds = scanner.detect(p.text)
        if kinds and "privacy" not in p.tags:
            problems.append(
                f"{p.id}: looks like it contains PII ({', '.join(kinds)}); tag it privacy"
            )
        for kind in kinds:
            for match in re.finditer(scanner.config.patterns[kind], p.text):
                if not _reserved(kind, match.group()):
                    problems.append(
                        f"{p.id}: {kind} {match.group()!r} is not from a reserved synthetic range"
                    )
    return problems


def distribution_problems(prompts: list[EvalPrompt]) -> list[str]:
    problems = []
    n = len(prompts)
    if not MIN_ROWS <= n <= MAX_ROWS:
        problems.append(f"dataset has {n} rows; keep it between {MIN_ROWS} and {MAX_ROWS}")
    tasks = Counter(p.task for p in prompts)
    for task in TASK_TYPES:
        if tasks[task] < MIN_PER_TASK:
            problems.append(f"task {task!r} has {tasks[task]} rows; at least {MIN_PER_TASK}")
    tiers = Counter(p.expected_tier for p in prompts)
    for tier in TIERS:
        if n and tiers[tier] / n < MIN_TIER_SHARE:
            problems.append(f"only {tiers[tier]} of {n} rows expect {tier}")
    tags = Counter(t for p in prompts for t in p.tags)
    for tag, minimum in MIN_TAGS.items():
        if tags[tag] < minimum:
            problems.append(f"tag {tag!r} on {tags[tag]} rows; at least {minimum}")
    quality = sum(p.quality for p in prompts)
    if not QUALITY_ROWS[0] <= quality <= QUALITY_ROWS[1]:
        problems.append(f"quality subset has {quality} rows; keep it within {QUALITY_ROWS}")
    return problems


def load_dataset(
    path: Path = PROMPTS, config: AppConfig | None = None, check_distribution: bool = True
) -> list[EvalPrompt]:
    """Parse and validate the whole file; raise DatasetError listing every problem."""
    config = config or load_config()
    problems: list[str] = []
    prompts: list[EvalPrompt] = []
    ids: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: not valid JSON ({exc.msg})")
            continue
        row_problems = _row_problems(row, where)
        if row_problems:
            problems += row_problems
            continue
        if row["id"] in ids:
            problems.append(f"{where}: duplicate id {row['id']!r} (first on line {ids[row['id']]})")
            continue
        ids[row["id"]] = number
        prompts.append(_to_prompt(row))
    if not prompts and not problems:
        problems.append(f"{path.name}: no rows")
    problems += content_problems(prompts, PiiScanner(config.routing.pii))
    if check_distribution and not problems:
        problems += distribution_problems(prompts)
    if problems:
        raise DatasetError(problems)
    return prompts


def dataset_sha256(path: Path = PROMPTS) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def composition(prompts: list[EvalPrompt]) -> dict[str, Any]:
    """Counts used by the report: per task and tier, per tag, and the quality subset."""
    by_task: dict[str, dict[str, int]] = {}
    for task in TASK_TYPES:
        rows = [p for p in prompts if p.task == task]
        by_task[task] = {
            "total": len(rows),
            "local": sum(p.expected_tier == "local" for p in rows),
            "premium": sum(p.expected_tier == "premium" for p in rows),
            "borderline": sum("borderline" in p.tags for p in rows),
            "quality": sum(p.quality for p in rows),
        }
    tags = Counter(t for p in prompts for t in p.tags)
    return {
        "rows": len(prompts),
        "local": sum(p.expected_tier == "local" for p in prompts),
        "premium": sum(p.expected_tier == "premium" for p in prompts),
        "quality": sum(p.quality for p in prompts),
        "by_task": by_task,
        "tags": dict(sorted(tags.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate eval/prompts.jsonl.")
    parser.add_argument("--path", type=Path, default=PROMPTS)
    args = parser.parse_args(argv)
    try:
        prompts = load_dataset(args.path)
    except DatasetError as exc:
        print(f"{len(exc.problems)} problem(s) in {args.path}:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    comp = composition(prompts)
    print(f"{args.path.name}: {comp['rows']} prompts, valid")
    print(f"  expected tier: {comp['local']} local, {comp['premium']} premium")
    print(f"  quality subset: {comp['quality']}")
    print(f"  {'task':<10} {'total':>5} {'local':>5} {'prem.':>5} {'bord.':>5} {'qual.':>5}")
    for task, c in comp["by_task"].items():
        print(
            f"  {task:<10} {c['total']:>5} {c['local']:>5} {c['premium']:>5} "
            f"{c['borderline']:>5} {c['quality']:>5}"
        )
    print("  tags: " + ", ".join(f"{t} {n}" for t, n in comp["tags"].items()))
    print(f"  sha256: {dataset_sha256(args.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
