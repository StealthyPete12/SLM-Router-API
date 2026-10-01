"""Table-driven tests for the complexity score and each modifier."""

import pytest

from app.config import AppConfig
from app.router.scoring import ScoreInputs, score_prompt

CODE = "```python\nprint('hi')\n```"


def score(config: AppConfig, text="hello", task="chitchat", tokens=10, messages=1, max_tokens=None):
    return score_prompt(
        ScoreInputs(
            task_type=task,
            text=text,
            input_tokens=tokens,
            conversation_messages=messages,
            max_tokens=max_tokens,
        ),
        config.routing,
    )


@pytest.mark.parametrize(
    ("task", "points"),
    [
        ("chitchat", 0),
        ("qa", 10),
        ("rewrite", 10),
        ("summarise", 15),
        ("translate", 15),
        ("extract", 15),
        ("creative", 20),
        ("code", 35),
        ("math", 40),
        ("analysis", 40),
    ],
)
def test_task_base_points(config, task, points):
    assert score(config, task=task).signals["task_base"] == points


@pytest.mark.parametrize(
    ("tokens", "points"),
    [(0, 0), (199, 0), (200, 5), (999, 5), (1000, 10), (2999, 10), (3000, 20), (50_000, 20)],
)
def test_input_length_bands(config, tokens, points):
    assert score(config, tokens=tokens).signals["input_length"] == points


@pytest.mark.parametrize(
    ("text", "signal", "points"),
    [
        ("Explain why the sky is blue", "reasoning_cues", 5),
        ("Step by step, explain why, with pros and cons", "reasoning_cues", 15),
        ("Step by step, explain why, pros and cons, justify, edge cases", "reasoning_cues", 15),
        ("What is X? What is Y?", "multi_part", 5),
        ("A? B? C? D? E?", "multi_part", 10),
        ("Do this:\n1. parse\n2. validate\n3. store", "multi_part", 10),
        ("One question only?", "multi_part", 0),
        ("Write a detailed answer", "output_demand", 10),
        ("Write 800 words on tea", "output_demand", 10),
        ("Write 200 words on tea", "output_demand", 0),
        # Question marks inside a fenced code block are code, not questions.
        ("Is this valid?\n```js\nx = a ? b : c ? d : e\n```", "multi_part", 0),
    ],
)
def test_text_modifiers(config, text, signal, points):
    assert score(config, text=text).signals[signal] == points


def test_max_tokens_counts_as_output_demand(config):
    assert score(config, max_tokens=999).signals["output_demand"] == 0
    assert score(config, max_tokens=1000).signals["output_demand"] == 10


@pytest.mark.parametrize(("messages", "points"), [(1, 0), (6, 0), (7, 5), (12, 5), (13, 10)])
def test_conversation_depth(config, messages, points):
    assert score(config, messages=messages).signals["conversation_depth"] == points


def test_code_in_input_only_when_task_is_not_code(config):
    assert (
        score(config, text=f"Summarise this\n{CODE}", task="summarise").signals["code_in_input"]
        == 10
    )
    assert score(config, text=f"Fix this\n{CODE}", task="code").signals["code_in_input"] == 0


def test_breakdown_sums_to_total_and_lists_every_signal(config):
    result = score(config, text="Compare A and B step by step", task="analysis", tokens=500)

    assert set(result.signals) == {
        "task_base",
        "input_length",
        "reasoning_cues",
        "multi_part",
        "output_demand",
        "conversation_depth",
        "code_in_input",
    }
    assert result.total == sum(result.signals.values()) == 40 + 5 + 5
    assert result.details["reasoning_cues"]["cues"] == ["step by step"]


def test_score_is_capped(config):
    text = "Detailed report, step by step, explain why, pros and cons? A? B? C?\n" + CODE
    result = score(config, text=text, task="math", tokens=5000, messages=20)

    assert result.uncapped > 100
    assert result.total == 100
