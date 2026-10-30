"""Scrub secrets before they reach a prompt, a tool result, or the audit log.

API keys are read from the environment only to build the denylist. Tool
handlers do not receive them, and nothing in this module prints them.
"""

from __future__ import annotations

import os
import re
from typing import Any

SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "MASSIVE_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
)

_KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
)

REDACTED = "[REDACTED]"


class Redactor:
    def __init__(self, extra_needles: tuple[str, ...] = ()) -> None:
        needles: list[str] = []
        for name in SECRET_ENV_NAMES:
            value = os.environ.get(name, "").strip()
            if len(value) >= 4:
                needles.append(value)
        needles.extend(needle for needle in extra_needles if len(needle) >= 4)
        # Longer needles first so a short prefix does not leave a tail behind.
        self._needles = tuple(sorted(set(needles), key=len, reverse=True))

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._scrub_text(value)
        if isinstance(value, dict):
            return {self._scrub_text(str(key)): self.scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.scrub(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.scrub(item) for item in value)
        return value

    def _scrub_text(self, text: str) -> str:
        for needle in self._needles:
            text = text.replace(needle, REDACTED)
        for pattern in _KEY_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text
