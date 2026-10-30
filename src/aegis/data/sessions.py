"""NYSE regular-session helpers.

Massive minute aggregates cover pre-market and after-hours. Research bars and
the risk clock use the regular session only. Free-tier history stops at the
last completed session before today's Eastern date, because the plan is
delayed end-of-day data rather than a live feed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from aegis.errors import PlanWindowError

NY = ZoneInfo("America/New_York")
# After the cash close, wait before treating today's session as complete on a paid plan.
CLOSE_BUFFER = timedelta(minutes=15)


@lru_cache(maxsize=1)
def nyse_calendar():
    return xcals.get_calendar("XNYS")


def to_eastern(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware. Refusing to guess UTC.")
    return ts.astimezone(NY)


def eastern_date(ts: datetime) -> date:
    return to_eastern(ts).date()


def sessions_between(start: date, end: date) -> list[date]:
    """NYSE sessions in ``[start, end]``, inclusive."""
    if start > end:
        return []
    calendar = nyse_calendar()
    index = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in index]


def is_session(day: date) -> bool:
    return bool(nyse_calendar().is_session(pd.Timestamp(day)))


def session_bounds_utc(day: date) -> tuple[datetime, datetime]:
    """Regular-session open inclusive and close exclusive, as UTC datetimes."""
    calendar = nyse_calendar()
    if not calendar.is_session(pd.Timestamp(day)):
        raise ValueError(f"{day.isoformat()} is not an NYSE session.")
    open_ts = calendar.session_open(pd.Timestamp(day)).to_pydatetime()
    close_ts = calendar.session_close(pd.Timestamp(day)).to_pydatetime()
    return open_ts, close_ts


def expected_minute_opens(day: date) -> list[datetime]:
    """One timestamp per regular-session minute, bar-open convention."""
    open_ts, close_ts = session_bounds_utc(day)
    opens: list[datetime] = []
    cursor = open_ts
    while cursor < close_ts:
        opens.append(cursor)
        cursor += timedelta(minutes=1)
    return opens


def latest_requestable_session(now: datetime, *, plan: str) -> date:
    """Latest session the pipeline will request.

    Free plan: the last NYSE session strictly before today's Eastern date.
    Paid plan: today's session once the close plus a short buffer has passed,
    otherwise the previous completed session.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware.")
    if plan not in {"free", "paid"}:
        raise ValueError(f"Unknown Massive plan {plan!r}.")
    today = now.astimezone(NY).date()
    calendar = nyse_calendar()
    today_is_session = bool(calendar.is_session(pd.Timestamp(today)))
    if plan == "paid" and today_is_session:
        close_ts = calendar.session_close(pd.Timestamp(today)).to_pydatetime()
        if now.astimezone(NY) >= close_ts.astimezone(NY) + CLOSE_BUFFER:
            return today
    if today_is_session:
        return calendar.previous_session(pd.Timestamp(today)).date()
    return calendar.date_to_session(pd.Timestamp(today), direction="previous").date()


def clamp_history_window(
    start: date,
    end: date,
    *,
    latest_end: date,
    lookback_days: int,
) -> tuple[date, date, tuple[str, ...]]:
    """Pull ``start``/``end`` into the plan's available history. Warn when clipping."""
    warnings: list[str] = []
    clipped_end = end
    if clipped_end > latest_end:
        warnings.append(
            f"Clamped end from {end.isoformat()} to {latest_end.isoformat()} "
            f"({lookback_days}-day delayed history; the in-progress session is not requested)."
        )
        clipped_end = latest_end
    earliest = latest_end - timedelta(days=lookback_days)
    clipped_start = start
    if clipped_start < earliest:
        warnings.append(
            f"Clamped start from {start.isoformat()} to {earliest.isoformat()} "
            f"(lookback is {lookback_days} days, about two years on the free tier)."
        )
        clipped_start = earliest
    if clipped_start > clipped_end:
        raise PlanWindowError(
            f"Nothing to fetch: start {clipped_start.isoformat()} is after the latest "
            f"requestable session {clipped_end.isoformat()}."
        )
    return clipped_start, clipped_end, tuple(warnings)
