"""Chunk a backfill so each Massive call stays inside the 50,000-minute scan cap.

``limit`` on the aggregates endpoint is a scan budget of base minutes, not a
page size. Extended hours are about 960 minutes per NYSE session (04:00–20:00
America/New_York). A 70-calendar-day window is about 48 sessions and ~46,000
base minutes, so ``limit=50000`` usually returns the window in one response.
The client still follows ``next_url`` if a window runs long.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from aegis.config import MAX_AGGREGATE_PAGE_LIMIT

# Minutes Massive scans for one extended session. Used only to size chunks.
EXTENDED_SESSION_MINUTES = 960
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def estimated_base_minutes(calendar_days: int) -> float:
    trading_days = calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR
    return trading_days * EXTENDED_SESSION_MINUTES


def assert_chunk_fits_page_limit(calendar_days: int, page_limit: int = MAX_AGGREGATE_PAGE_LIMIT) -> None:
    scanned = estimated_base_minutes(calendar_days)
    if scanned > page_limit:
        raise ValueError(
            f"A chunk of {calendar_days} calendar days scans about {scanned:.0f} base minutes, "
            f"above the Massive limit of {page_limit}. Lower MASSIVE_CHUNK_CALENDAR_DAYS."
        )


def group_missing_sessions(
    sessions: list[date],
    have: set[date],
    *,
    max_calendar_days: int,
) -> list[tuple[date, date]]:
    """Group missing sessions into request ranges.

    A range never spans more than ``max_calendar_days`` and never covers a
    session that is already cached. Restarting a backfill therefore
    re-requests only the hole, not the days already on disk.
    """
    if max_calendar_days < 1:
        raise ValueError("max_calendar_days must be positive.")
    missing = [day for day in sessions if day not in have]
    if not missing:
        return []
    session_set = set(sessions)
    groups: list[tuple[date, date]] = []
    current: list[date] = [missing[0]]
    for day in missing[1:]:
        span = (day - current[0]).days + 1
        between = [item for item in session_set if current[-1] < item < day]
        cached_between = any(item in have for item in between)
        if cached_between or span > max_calendar_days:
            groups.append((current[0], current[-1]))
            current = [day]
        else:
            current.append(day)
    groups.append((current[0], current[-1]))
    return groups


@dataclass(frozen=True)
class BackfillPlan:
    start: date
    end: date
    symbols: tuple[str, ...]
    bar_requests: int
    cached_sessions: int
    missing_sessions: int
    corporate_action_requests: int
    rate_limit_per_minute: float
    chunk_calendar_days: int
    page_limit: int
    warnings: tuple[str, ...]

    @property
    def total_requests(self) -> int:
        return self.bar_requests + self.corporate_action_requests

    @property
    def estimated_minutes(self) -> float:
        if self.rate_limit_per_minute <= 0:
            return 0.0
        return self.total_requests / self.rate_limit_per_minute

    def render(self) -> str:
        lines = [
            f"Window {self.start.isoformat()} .. {self.end.isoformat()}  "
            f"({len(self.symbols)} symbols)",
            f"Cached sessions: {self.cached_sessions}",
            f"Missing sessions: {self.missing_sessions}",
            f"Bar requests: {self.bar_requests} "
            f"(one call per chunk, up to {self.chunk_calendar_days} calendar days, "
            f"limit={self.page_limit})",
            f"Corporate-action requests: {self.corporate_action_requests}",
            f"Total requests: {self.total_requests}",
            f"Estimated wall time: {self.estimated_minutes:.1f} minutes "
            f"at {self.rate_limit_per_minute:g} requests/minute.",
        ]
        scanned = estimated_base_minutes(self.chunk_calendar_days)
        lines.append(
            f"Chunk scan budget: about {scanned:.0f} base minutes "
            f"(cap {self.page_limit})."
        )
        for warning in self.warnings:
            lines.append(f"Note: {warning}")
        if self.bar_requests == 0 and self.corporate_action_requests == 0:
            lines.append("Cache is complete for this window. Nothing to download.")
        return "\n".join(lines)
