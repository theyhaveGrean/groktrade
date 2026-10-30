"""A two-session price path that does not need an API key.

Day one declines, then rallies hard enough for a fast EMA to cross a slow EMA
and for a 2.5R target to be reachable. Day two is flat so the report has a
second daily return. Both sessions are regular NYSE hours in January 2024,
outside the default holdout.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from aegis.data.cache import BAR_SCHEMA

NY = ZoneInfo("America/New_York")
DEMO_DAYS = (date(2024, 1, 3), date(2024, 1, 4))


# Weekdays used by `aegis optimize demo`. Two rolling folds fit in this span
# with train_days=3, test_days=3, embargo_days=2, step_days=5, end=2024-01-15.
OPTIMIZE_DEMO_DAYS = (
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 15),
)


def synthetic_bars(symbol: str = "AAPL", sessions: tuple[date, ...] | list[date] | None = None) -> pl.DataFrame:
    days = tuple(sessions) if sessions is not None else DEMO_DAYS
    # Extra sessions repeat the trend day so a walk-forward window can trade.
    # The default two-day demo keeps a flat second session.
    trend_every_session = sessions is not None
    rows: list[dict[str, object]] = []
    for index, day in enumerate(days):
        if trend_every_session or index == 0:
            closes = _day_one_closes()
        else:
            closes = _flat_closes(_day_one_closes()[-1])
        rows.extend(_rows(symbol, day, closes))
    frame = pl.DataFrame(rows).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("transactions").cast(pl.Int64),
    )
    return frame.select(list(BAR_SCHEMA)).sort("ts")


def _day_one_closes() -> list[float]:
    price = 100.0
    closes: list[float] = []
    for minute in range(390):
        if minute < 40:
            price -= 0.08
        elif minute < 140:
            price += 0.10
        closes.append(round(price, 4))
    return closes


def _flat_closes(price: float) -> list[float]:
    return [round(price, 4)] * 390


def _rows(symbol: str, day: date, closes: list[float]) -> list[dict[str, object]]:
    open_local = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
    previous = closes[0]
    rows: list[dict[str, object]] = []
    for offset, close in enumerate(closes):
        ts = (open_local + timedelta(minutes=offset)).astimezone(timezone.utc)
        open_ = previous
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        rows.append(
            {
                "symbol": symbol,
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100_000.0,
                "vwap": (high + low + close) / 3,
                "transactions": 100,
            }
        )
        previous = close
    return rows
