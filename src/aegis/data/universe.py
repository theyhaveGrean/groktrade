"""Configurable trading universe with point-in-time membership.

``end: null`` means the name is still listed. A delisted symbol stays in the
file with an ``end`` date so later research can include it without pretending
it traded forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")


@dataclass(frozen=True)
class Membership:
    symbol: str
    start: date
    end: date | None
    listing_mic: str = "XNAS"
    name: str = ""

    @property
    def active(self) -> bool:
        return self.end is None


def validate_symbol(symbol: str) -> str:
    if not _SYMBOL.match(symbol):
        raise ValueError(f"Invalid ticker {symbol!r}.")
    return symbol


def packaged_universe_path() -> Path:
    return Path(__file__).with_name("default_universe.yaml")


def resolve_universe_path(configured: Path) -> Path:
    if configured.exists():
        return configured
    packaged = packaged_universe_path()
    if packaged.exists():
        return packaged
    raise FileNotFoundError(
        f"Universe file not found at {configured}. Set UNIVERSE_PATH or run from the repository root."
    )


def load_universe(path: Path) -> tuple[Membership, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("universe") or []
    members: list[Membership] = []
    for row in rows:
        end = row.get("end")
        members.append(
            Membership(
                symbol=validate_symbol(str(row["symbol"])),
                name=str(row.get("name") or ""),
                listing_mic=str(row.get("listing_mic") or "XNAS"),
                start=date.fromisoformat(str(row["start"])),
                end=None if end in (None, "null", "") else date.fromisoformat(str(end)),
            )
        )
    return tuple(members)


def select_as_of(members: tuple[Membership, ...] | list[Membership], as_of: date) -> tuple[Membership, ...]:
    """Members whose listing window contains ``as_of`` (inclusive)."""
    chosen = [
        member
        for member in members
        if member.start <= as_of and (member.end is None or as_of <= member.end)
    ]
    return tuple(sorted(chosen, key=lambda member: member.symbol))


def symbol_window(member: Membership, start: date, end: date) -> tuple[date, date] | None:
    """Intersect a backfill range with one membership. ``None`` if they do not overlap."""
    window_start = max(start, member.start)
    window_end = end if member.end is None else min(end, member.end)
    if window_start > window_end:
        return None
    return window_start, window_end
