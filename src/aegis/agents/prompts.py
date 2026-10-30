"""Load versioned system prompts from disk."""

from __future__ import annotations

import hashlib
from pathlib import Path

from aegis.errors import ResearchError

_ROLES = ("orchestrator", "researcher", "implementer", "naysayer")
_ROOT = Path(__file__).resolve().parent / "prompts"


def load_prompt(role: str, version: str) -> str:
    if role not in _ROLES:
        raise ResearchError(f"Unknown agent role {role!r}.")
    path = _ROOT / role / f"{version}.md"
    if not path.is_file():
        raise ResearchError(f"No system prompt for {role} at version {version}.")
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else ""
    if first.strip() != f"Prompt-Version: {version}":
        raise ResearchError(f"{path.name} must start with 'Prompt-Version: {version}'.")
    return text


def prompt_hash(role: str, version: str) -> str:
    return hashlib.sha256(load_prompt(role, version).encode("utf-8")).hexdigest()
