"""Privacy hook for the first hard rule.

The router calls a PiiDetector on every message. The default is the regex
PiiScanner in app/aggregator/guardrails.py (emails, phone, card and ID numbers,
configured in routing.yaml `pii`). The policy honours the result: a prompt with
PII and routing.yaml `privacy_mode: true` stays local even when the caller
forces premium, and never falls back to the cloud.
"""

from collections.abc import Callable

# Returns the kinds of PII found in the text; an empty list means none.
PiiDetector = Callable[[str], list[str]]
