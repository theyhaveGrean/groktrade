"""Local paper book. The broker remains the source of truth; this is what we reconcile."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegis.errors import AegisError

STATE_VERSION = 1


@dataclass
class PaperState:
    path: Path
    version: int = STATE_VERSION
    session_date: str | None = None
    day_start_equity: float = 0.0
    halted_session: str | None = None
    mismatch: bool = False
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PaperState:
        if not path.exists():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AegisError(
                f"Paper state at {path} is not valid JSON. Refusing to trade over it."
            ) from exc
        version = int(payload.get("version", 0))
        if version != STATE_VERSION:
            raise AegisError(
                f"Paper state at {path} is version {version}. This build reads version {STATE_VERSION}."
            )
        return cls(
            path=path,
            version=version,
            session_date=payload.get("session_date"),
            day_start_equity=float(payload.get("day_start_equity") or 0.0),
            halted_session=payload.get("halted_session"),
            mismatch=bool(payload.get("mismatch")),
            positions=dict(payload.get("positions") or {}),
            orders=dict(payload.get("orders") or {}),
        )

    def clear_exposure(self) -> None:
        self.positions.clear()
        self.orders.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_date": self.session_date,
            "day_start_equity": self.day_start_equity,
            "halted_session": self.halted_session,
            "mismatch": self.mismatch,
            "positions": self.positions,
            "orders": self.orders,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
