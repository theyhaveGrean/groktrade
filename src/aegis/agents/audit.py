"""Append-only JSONL log of every agent message and tool call."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.agents.redaction import Redactor


class AuditLog:
    def __init__(self, path: Path, run_id: str, redactor: Redactor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self._redactor = redactor
        self._handle = path.open("a", encoding="utf-8")

    def write(self, event: str, agent: str | None, payload: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "agent": agent,
            "payload": self._redactor.scrub(payload),
        }
        self._handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
