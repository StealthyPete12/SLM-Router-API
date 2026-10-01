"""Input and output guardrails: size limit, regex PII detection and masking, empty answers."""

import re
from dataclasses import dataclass

from app.config import PiiConfig
from app.schemas import ChatCompletionResponse


def luhn_valid(number: str) -> bool:
    """True when the digits in `number` pass the Luhn checksum used by card numbers."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class PiiScanner:
    """Finds and masks the PII kinds configured in routing.yaml `pii`."""

    config: PiiConfig

    def _matches(self, kind: str, text: str) -> list[re.Match[str]]:
        found = re.finditer(self.config.patterns[kind], text)
        if kind in self.config.luhn_checked:
            return [m for m in found if luhn_valid(m.group())]
        return list(found)

    def detect(self, text: str) -> list[str]:
        """The kinds of PII in `text`, in configured order. Matches the router's PiiDetector."""
        kinds = []
        for kind in self.config.patterns:
            if self._matches(kind, text):
                kinds.append(kind)
                # Mask before the next kind, so one value is never counted twice
                # (a card number also looks like a phone number).
                text = self._mask_kind(kind, text)
        return kinds

    def _mask_kind(self, kind: str, text: str) -> str:
        for match in reversed(self._matches(kind, text)):
            text = f"{text[: match.start()]}[{kind.upper()}]{text[match.end() :]}"
        return text

    def mask(self, text: str) -> str:
        for kind in self.config.patterns:
            text = self._mask_kind(kind, text)
        return text


def input_chars(contents: list[str]) -> int:
    return sum(len(c) for c in contents)


def is_empty_answer(answer: ChatCompletionResponse) -> bool:
    """No text and no tool calls in any choice: nothing the client could use."""
    return not any(
        (c.message.content or "").strip() or c.message.tool_calls for c in answer.choices
    )
