"""Protected final holdout.

Research and optimization must call :func:`assert_no_holdout_overlap` before
reading bars. Backfill and data-quality checks are maintenance paths: they may
store holdout dates so a later, separate evaluation can use them once. That
evaluation runner is intentionally not part of this build.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from aegis.errors import HoldoutViolation


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Date range start {self.start} is after end {self.end}.")

    def overlaps(self, other: DateRange) -> bool:
        return self.start <= other.end and other.start <= self.end

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


def assert_no_holdout_overlap(requested: DateRange, holdout: DateRange, *, purpose: str) -> None:
    """Refuse the request when it shares any session with the holdout."""
    if requested.overlaps(holdout):
        raise HoldoutViolation(
            f"Refusing to read {requested} for {purpose}: it overlaps the protected "
            f"holdout {holdout}. Research and optimization cannot use that window. "
            "Move the request so it ends the day before HOLDOUT_START."
        )


def assert_dates_outside_holdout(dates: set[date], holdout: DateRange, *, purpose: str) -> None:
    """Refuse when any concrete bar date falls inside the holdout."""
    leaked = sorted(day for day in dates if holdout.contains(day))
    if leaked:
        raise HoldoutViolation(
            f"Refusing to read {len(leaked)} holdout session(s) starting {leaked[0].isoformat()} "
            f"for {purpose}. The protected window is {holdout}."
        )
