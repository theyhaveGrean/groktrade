"""Emergency stop shared by the running process and ``aegis paper kill``.

``RISK_KILL_SWITCH=true`` stops the process that loaded it. A kill file stops
every process that reads the same path, including one started later. Deleting
the file is a manual step; this package does not clear it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aegis.config import Settings


def kill_file(settings: Settings) -> Path:
    if settings.paper_kill_file is not None:
        return settings.paper_kill_file
    return settings.paper_dir / "KILL"


def kill_reason(settings: Settings) -> str | None:
    """Why trading must stop, or ``None`` when neither switch is armed."""
    if settings.risk_kill_switch:
        return "RISK_KILL_SWITCH"
    path = kill_file(settings)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "kill file"
    return text.splitlines()[0][:200]


def arm_kill_file(settings: Settings, reason: str) -> Path:
    path = kill_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    path.write_text(f"{reason.strip() or 'aegis paper kill'}\n{stamp}\n", encoding="utf-8")
    return path
