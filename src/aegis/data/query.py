"""DuckDB scan plus Polars adjustment. This is the research read path.

:func:`load_research_bars` refuses the protected holdout before it opens a
file. Maintenance reads (validation, cache inspection) opt out explicitly.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from aegis.data.adjustments import apply_dividend_adjustments, apply_split_adjustments
from aegis.data.cache import BarCache, empty_bars
from aegis.data.sessions import session_bounds_utc
from aegis.holdout import DateRange, assert_dates_outside_holdout


def load_research_bars(
    cache: BarCache,
    symbols: list[str],
    start: date,
    end: date,
    holdout: DateRange,
    *,
    adjusted: bool = True,
    adjust_dividends: bool = False,
    as_of: date | None = None,
    regular_hours_only: bool = True,
) -> pl.DataFrame:
    """Bars for research. Raises :class:`aegis.errors.HoldoutViolation` on overlap."""
    frame = cache.read_bars(symbols, start, end, purpose="research", holdout=holdout)
    frame = _prepare(
        cache,
        frame,
        symbols,
        adjusted=adjusted,
        adjust_dividends=adjust_dividends,
        as_of=as_of or end,
        regular_hours_only=regular_hours_only,
    )
    if not frame.is_empty():
        dates = set(
            frame.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date())
            .to_series()
            .to_list()
        )
        assert_dates_outside_holdout(dates, holdout, purpose="research bar read")
    return frame


def load_maintenance_bars(
    cache: BarCache,
    symbols: list[str],
    start: date,
    end: date,
    *,
    regular_hours_only: bool = False,
) -> pl.DataFrame:
    """Unadjusted cache read for validation. Allowed to see holdout dates."""
    frame = cache.read_bars(symbols, start, end, purpose="maintenance")
    if regular_hours_only:
        frame = filter_regular_hours(frame)
    return frame


def _prepare(
    cache: BarCache,
    frame: pl.DataFrame,
    symbols: list[str],
    *,
    adjusted: bool,
    adjust_dividends: bool,
    as_of: date,
    regular_hours_only: bool,
) -> pl.DataFrame:
    if frame.is_empty():
        return empty_bars()
    if regular_hours_only:
        frame = filter_regular_hours(frame)
    if frame.is_empty():
        return empty_bars()
    if not adjusted:
        return frame.sort(["symbol", "ts"])
    splits = cache.read_splits(symbols)
    frame = apply_split_adjustments(frame, splits, as_of)
    if adjust_dividends:
        dividends = cache.read_dividends(symbols)
        frame = apply_dividend_adjustments(frame, dividends, splits, as_of)
    return frame.sort(["symbol", "ts"])


def filter_regular_hours(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep prints inside the NYSE regular session, including early closes."""
    if frame.is_empty():
        return frame
    stamped = frame.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_session")
    )
    sessions = stamped["_session"].unique().to_list()
    bounds: list[dict[str, object]] = []
    for session in sessions:
        try:
            open_ts, close_ts = session_bounds_utc(session)
        except ValueError:
            continue
        bounds.append({"_session": session, "_open": open_ts, "_close": close_ts})
    if not bounds:
        return frame.clear()
    bound_frame = pl.DataFrame(bounds).with_columns(
        pl.col("_open").cast(pl.Datetime("us", "UTC")),
        pl.col("_close").cast(pl.Datetime("us", "UTC")),
    )
    joined = stamped.join(bound_frame, on="_session", how="inner")
    kept = joined.filter((pl.col("ts") >= pl.col("_open")) & (pl.col("ts") < pl.col("_close")))
    return kept.drop(["_session", "_open", "_close"]).sort(["symbol", "ts"])
