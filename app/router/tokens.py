"""Input-token estimate for routing decisions.

One tokenizer (tiktoken, routing.yaml `tokens.encoding`) is used for every model.
That is close enough to decide length bands and context-window fit; final costs
use the usage each provider reports.
"""

from collections.abc import Iterable
from functools import lru_cache

import tiktoken

from app.config import TokensConfig


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class TokenEstimator:
    def __init__(self, config: TokensConfig) -> None:
        self._config = config
        self._encoding = _encoding(config.encoding)

    def count(self, text: str) -> int:
        """Tokens in a piece of text. Special-token strings are counted as plain text."""
        return len(self._encoding.encode(text, disallowed_special=()))

    def estimate_messages(self, contents: Iterable[str]) -> int:
        """Tokens a chat request sends: each message's content plus fixed overheads."""
        per_message = self._config.per_message_overhead
        total = sum(self.count(content) + per_message for content in contents)
        return total + self._config.reply_overhead
