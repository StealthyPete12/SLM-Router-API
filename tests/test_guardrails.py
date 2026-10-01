"""PII detection and masking, the Luhn check, and the empty-answer guardrail."""

import pytest

from app.aggregator.guardrails import PiiScanner, is_empty_answer, luhn_valid
from app.config import DEFAULT_CONFIG_DIR, load_config
from app.schemas import ChatCompletionResponse
from scripts.acceptance import CASES

# Fictional values only: example.com addresses, 555-01xx numbers, test card numbers.
SCANNER = PiiScanner(load_config(DEFAULT_CONFIG_DIR).routing.pii)


@pytest.mark.parametrize(
    ("text", "kinds", "masked"),
    [
        ("Write to jane@example.com today", ["email"], "Write to [EMAIL] today"),
        ("Call +1 555-010-0199 now", ["phone"], "Call [PHONE] now"),
        ("Call (020) 7946 0018 now", ["phone"], "Call [PHONE] now"),
        ("Card 4111 1111 1111 1111 expires", ["card"], "Card [CARD] expires"),
        ("SSN 123-45-6789 on file", ["us_ssn"], "SSN [US_SSN] on file"),
        ("NI number AB 12 34 56 C", ["uk_nino"], "NI number [UK_NINO]"),
        ("IBAN GB82 WEST 1234 5698 7654 32", ["iban"], "IBAN [IBAN]"),
        (
            "jane@example.com, card 5555555555554444",
            ["card", "email"],
            "[EMAIL], card [CARD]",
        ),
    ],
)
def test_detects_and_masks(text, kinds, masked):
    assert SCANNER.detect(text) == kinds
    assert SCANNER.mask(text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "Solve for x: 3x + 7 = 22",
        "They met on 3 March 2024 and again on 2024-04-12 at 09:30.",
        "Version 1.2.3 shipped to 42 users",
        "The meeting is in room 101",
    ],
)
def test_no_false_positives(text):
    assert SCANNER.detect(text) == []
    assert SCANNER.mask(text) == text


def test_acceptance_prompts_have_no_pii():
    for case in CASES:
        assert SCANNER.detect(case.prompt) == [], case.name


def test_card_numbers_must_pass_luhn():
    # Not a card; a long digit run is still masked (as a phone), erring on privacy.
    assert "card" not in SCANNER.detect("Card 4111 1111 1111 1112 fails the checksum")


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    assert luhn_valid("5555-5555-5555-4444")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("1234")


def _answer(content, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ChatCompletionResponse.model_validate(
        {"id": "x", "created": 0, "model": "m", "choices": [{"index": 0, "message": message}]}
    )


def test_empty_answer():
    assert is_empty_answer(_answer(""))
    assert is_empty_answer(_answer("   \n"))
    assert is_empty_answer(_answer(None))
    assert not is_empty_answer(_answer("Hello"))
    assert not is_empty_answer(_answer(None, [{"id": "1", "type": "function", "function": {}}]))
