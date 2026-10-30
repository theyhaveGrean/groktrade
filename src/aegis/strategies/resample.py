"""Aggregate cached 1-minute bars up to a coarser regular-session timeframe.

Buckets are aligned to 9:30 America/New_York and never cross a session, so a
5-minute bar is five regular-session minutes rather than an hour-of-day slice
that could mix the close with the next morning.
"""

from __future__ import annotations

import polars as pl

from aegis.data.cache import BAR_SCHEMA

_TIMEFRAMES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}
_ALIASES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min", "1h": "60min"}
_SESSION_OPEN_MINUTE = 9 * 60 + 30


def timeframe_minutes(value: str) -> int:
    key = value.strip().lower().replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in _TIMEFRAMES:
        allowed = ", ".join(sorted(_TIMEFRAMES))
        raise ValueError(f"Unknown bar timeframe {value!r}. Use one of: {allowed}.")
    return _TIMEFRAMES[key]


def canonical_timeframe(value: str) -> str:
    minutes = timeframe_minutes(value)
    return next(name for name, size in _TIMEFRAMES.items() if size == minutes)


def resample_bars(bars: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Return OHLCV bars at ``timeframe``. ``1min`` is a copy of the input."""
    minutes = timeframe_minutes(timeframe)
    if bars.is_empty() or minutes == 1:
        return bars
    work = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et")
    )
    work = work.with_columns(
        pl.col("_et").dt.date().alias("_session"),
        (
            pl.col("_et").dt.hour().cast(pl.Int32) * 60
            + pl.col("_et").dt.minute().cast(pl.Int32)
            - _SESSION_OPEN_MINUTE
        ).alias("_offset"),
    )
    work = work.filter((pl.col("_offset") >= 0) & (pl.col("_offset") < 390))
    if work.is_empty():
        return bars.clear()
    work = work.with_columns((pl.col("_offset") // minutes).alias("_bucket"))
    grouped = work.group_by(["symbol", "_session", "_bucket"], maintain_order=True).agg(
        pl.col("ts").first().alias("ts"),
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        ((pl.col("vwap") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
        pl.col("transactions").sum().alias("transactions"),
    )
    return grouped.select(list(BAR_SCHEMA)).sort(["symbol", "ts"])
