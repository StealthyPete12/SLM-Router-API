"""Complexity score: task base points plus modifiers, capped at routing.yaml `max_score`.

Each modifier is a small pure function returning (points, evidence), so the
breakdown in every response shows exactly how the score was reached.
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from app.config import (
    ClassifierConfig,
    ConversationDepthModifier,
    InputLengthModifier,
    Modifiers,
    MultiPartModifier,
    OutputDemandModifier,
    RoutingConfig,
    TaskType,
)
from app.router.classifier import has_code_block, keyword_regex, pattern_regex, strip_code_blocks

Detail = dict[str, Any]


@dataclass(frozen=True)
class ScoreInputs:
    task_type: TaskType
    text: str  # the last user message
    input_tokens: int  # estimate for the whole request
    conversation_messages: int  # user and assistant messages
    max_tokens: int | None


@dataclass(frozen=True)
class Score:
    total: int  # capped at max_score
    uncapped: int
    signals: dict[str, int]  # points per signal, every signal present
    details: dict[str, Detail]  # what triggered each signal


def input_length_points(tokens: int, rule: InputLengthModifier) -> tuple[int, Detail]:
    band = bisect_right(rule.bands, tokens)
    return rule.points[band], {"input_tokens": tokens}


def reasoning_cue_points(prose: str, modifiers: Modifiers) -> tuple[int, Detail]:
    rule = modifiers.reasoning_cues
    cues = [k for k in rule.keywords if keyword_regex(k).search(prose)]
    return min(rule.each * len(cues), rule.max), {"cues": cues}


def multi_part_points(prose: str, rule: MultiPartModifier) -> tuple[int, Detail]:
    questions = len(pattern_regex(rule.question_pattern).findall(prose))
    numbered = len(pattern_regex(rule.numbered_item_pattern).findall(prose))
    extra = max(questions, numbered, 1) - 1
    detail = {"questions": questions, "numbered_items": numbered, "extra_parts": extra}
    return min(rule.each * extra, rule.max), detail


def output_demand_points(
    prose: str, max_tokens: int | None, rule: OutputDemandModifier
) -> tuple[int, Detail]:
    triggers = [k for k in rule.keywords if keyword_regex(k).search(prose)]
    for match in pattern_regex(rule.words_requested_pattern).finditer(prose):
        words = int(match.group(1).replace(",", ""))
        if words >= rule.min_words_requested:
            triggers.append(f"{words} words requested")
    if max_tokens is not None and max_tokens >= rule.min_max_tokens:
        triggers.append(f"max_tokens={max_tokens}")
    return (rule.points if triggers else 0), {"triggers": triggers}


def conversation_depth_points(messages: int, rule: ConversationDepthModifier) -> tuple[int, Detail]:
    level = sum(messages > limit for limit in rule.more_than)
    points = rule.points[level - 1] if level else 0
    return points, {"messages": messages}


def code_in_input_points(
    text: str, task_type: TaskType, points: int, classifier: ClassifierConfig
) -> tuple[int, Detail]:
    present = has_code_block(text, classifier)
    applies = present and task_type != "code"
    return (points if applies else 0), {"code_block": present, "task_is_code": task_type == "code"}


def score_prompt(inputs: ScoreInputs, config: RoutingConfig) -> Score:
    """Score one request. Pure: the same inputs and config always give the same score."""
    modifiers = config.modifiers
    prose = strip_code_blocks(inputs.text, config.classifier)

    parts: dict[str, tuple[int, Detail]] = {
        "task_base": (config.task_base[inputs.task_type], {"task_type": inputs.task_type}),
        "input_length": input_length_points(inputs.input_tokens, modifiers.input_length),
        "reasoning_cues": reasoning_cue_points(prose, modifiers),
        "multi_part": multi_part_points(prose, modifiers.multi_part),
        "output_demand": output_demand_points(prose, inputs.max_tokens, modifiers.output_demand),
        "conversation_depth": conversation_depth_points(
            inputs.conversation_messages, modifiers.conversation_depth
        ),
        "code_in_input": code_in_input_points(
            inputs.text, inputs.task_type, modifiers.code_in_input.points, config.classifier
        ),
    }
    signals = {name: points for name, (points, _) in parts.items()}
    uncapped = sum(signals.values())
    return Score(
        total=min(uncapped, config.max_score),
        uncapped=uncapped,
        signals=signals,
        details={name: detail for name, (_, detail) in parts.items()},
    )
