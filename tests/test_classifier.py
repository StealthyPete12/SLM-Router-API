"""Table-driven tests for the keyword task classifier."""

import pytest

from app.config import AppConfig
from app.router.classifier import classify

CODE = "```python\nx = compute(1)\n```"


@pytest.mark.parametrize(
    ("prompt", "task", "decided_by", "signal"),
    [
        ("Hi! What can you do?", "chitchat", "matched", "hi"),
        ("ok cool", "chitchat", "short_message", None),
        ("What is the capital of France?", "qa", "matched", "what is"),
        ("Tell me something interesting about the history of Rome", "qa", "fallback", None),
        ("Rephrase this sentence: the cat sat on the mat", "rewrite", "matched", "rephrase"),
        ("Summarize the text below in two lines", "summarise", "matched", "summarize"),
        ("tl;dr this thread please", "summarise", "matched", "tl;dr"),
        (
            "Put this sentence in Spanish: where is the station",
            "translate",
            "tie_break",
            "in spanish",
        ),
        ("Classify each review as positive or negative", "extract", "matched", "classify"),
        ("Write a slogan for a bakery", "creative", "matched", "slogan"),
        ("Why does this throw?\n" + CODE, "code", "matched", "code block"),
        ("Calculate the derivative of x^2 + 3x", "math", "matched", "calculate"),
        ("Design a caching strategy for our API", "analysis", "matched", "design"),
        # Ties go to the more demanding task (routing.yaml priority).
        ("Hello, compare tea", "analysis", "tie_break", "compare"),
    ],
)
def test_task_types(config: AppConfig, prompt, task, decided_by, signal):
    result = classify(prompt, config.routing.classifier)

    assert result.task_type == task
    assert result.decided_by == decided_by
    if signal is not None:
        assert signal in [e.signal for e in result.evidence_for(task)]


def test_instruction_outweighs_pasted_body(config: AppConfig):
    # "compare" and "plan" appear only in the pasted text, "summarise" in the instruction.
    prompt = "Summarise this email:\nWe should compare vendors and plan the move."

    result = classify(prompt, config.routing.classifier)

    assert result.task_type == "summarise"
    assert {e.where for e in result.evidence_for("analysis")} == {"body"}


def test_keywords_inside_code_blocks_are_ignored(config: AppConfig):
    prompt = "Explain this snippet\n```\n# translate and summarise\n```"

    result = classify(prompt, config.routing.classifier)

    assert result.evidence_for("translate") == []
    assert result.evidence_for("summarise") == []


def test_whole_words_only(config: AppConfig):
    # "this" contains "hi", "classify" contains "class": neither may match.
    result = classify("this is about a classifying approach", config.routing.classifier)

    assert result.evidence_for("chitchat") == []


def test_is_deterministic(config: AppConfig):
    prompt = "Compare Postgres and MongoDB for event sourcing, step by step."
    assert classify(prompt, config.routing.classifier) == classify(
        prompt, config.routing.classifier
    )
