"""Deterministic keyword classifier: labels a prompt with one of ten task types.

Every rule comes from routing.yaml `classifier`. The result carries the evidence
(which keyword or pattern matched, where, and for how many points), so a
decision can always be explained. See routing.yaml for how points are counted.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import ClassifierConfig, TaskType


@dataclass(frozen=True)
class Evidence:
    task: TaskType
    signal: str  # the keyword, pattern or "code block" that matched
    where: str  # "instruction", "body" or "pattern"
    points: int


@dataclass(frozen=True)
class Classification:
    task_type: TaskType
    # How the winner was chosen: "matched", "tie_break", "short_message" or "fallback".
    decided_by: str
    evidence: list[Evidence] = field(default_factory=list)
    totals: dict[TaskType, int] = field(default_factory=dict)

    def evidence_for(self, task: TaskType) -> list[Evidence]:
        return [e for e in self.evidence if e.task == task]


@lru_cache(maxsize=1024)
def keyword_regex(keyword: str) -> re.Pattern[str]:
    """Case-insensitive whole-word match; also used by scoring for cue keywords."""
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=256)
def pattern_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def has_code_block(text: str, config: ClassifierConfig) -> bool:
    return pattern_regex(config.code_block_pattern).search(text) is not None


def strip_code_blocks(text: str, config: ClassifierConfig) -> str:
    """The prose of a message: keywords and cues are matched outside code."""
    return pattern_regex(config.code_block_pattern).sub(" ", text)


def instruction_of(text: str, config: ClassifierConfig) -> str:
    """The first non-empty line, capped: where the user states what they want."""
    first_line = next((line for line in text.strip().splitlines() if line.strip()), "")
    return first_line[: config.instruction_max_chars]


def classify(text: str, config: ClassifierConfig) -> Classification:
    """Label `text` (the last user message) with a task type and the evidence for it."""
    prose = strip_code_blocks(text, config)
    instruction = instruction_of(prose, config)
    weights = config.weights

    evidence: list[Evidence] = []
    for task, rules in config.tasks.items():
        for keyword in rules.keywords:
            regex = keyword_regex(keyword)
            if regex.search(instruction):
                evidence.append(Evidence(task, keyword, "instruction", weights.instruction))
            elif regex.search(prose):
                evidence.append(Evidence(task, keyword, "body", weights.body))
        for pattern in rules.patterns:
            match = pattern_regex(pattern).search(text)
            if match:
                evidence.append(Evidence(task, match.group(0), "pattern", weights.pattern))
    if has_code_block(text, config):
        evidence.append(Evidence("code", "code block", "pattern", weights.pattern))

    totals: dict[TaskType, int] = {}
    for item in evidence:
        totals[item.task] = totals.get(item.task, 0) + item.points

    if totals:
        best = max(totals.values())
        leaders = [task for task in config.priority if totals.get(task) == best]
        decided_by = "matched" if len(leaders) == 1 else "tie_break"
        return Classification(leaders[0], decided_by, evidence, totals)

    if len(prose.split()) <= config.chitchat_max_words:
        return Classification("chitchat", "short_message", evidence, totals)
    return Classification(config.fallback_task, "fallback", evidence, totals)
