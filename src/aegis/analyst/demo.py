"""Synthetic result bundle so the analyst report can be printed with no API key.

The trades, curve, folds, and bars are constructed here. No strategy class
runs. Costs match the platform schedule, so the 1x/2x/3x table moves.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from aegis.analyst.costs import round_trip_cost
from aegis.analyst.input import AnalysisInput, FoldInput
from aegis.backtest.report import TradeRecord

NY = ZoneInfo("America/New_York")
_COMMISSION = 0.005
_SLIPPAGE_BPS = 5.0


def demo_analysis_input() -> AnalysisInput:
    """Four January 2024 sessions, outside the default holdout, plus three OOS folds."""
    days = (date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8))
    bars = _demo_bars(days)
    trades = (
        _trade("AAPL", days[0], 100.0, 101.0, 100, 0.50, "target"),
        _trade("AAPL", days[1], 101.0, 99.50, 80, -0.75, "stop"),
        _trade("AAPL", days[2], 99.50, 100.10, 60, 0.40, "target"),
        _trade("AAPL", days[3], 100.10, 100.40, 40, 0.25, "flatten"),
    )
    equity = 100_000.0
    curve: list[tuple[datetime, float]] = []
    for day, trade in zip(days, trades, strict=True):
        equity += trade.pnl
        curve.append((_at(day, 16, 0), equity))
    folds = (
        _fold(0, date(2024, 1, 2), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 12), 101_000.0, 1.5, 1_000.0),
        _fold(1, date(2024, 1, 8), date(2024, 1, 12), date(2024, 1, 16), date(2024, 1, 19), 99_500.0, -0.4, -500.0),
        _fold(2, date(2024, 1, 16), date(2024, 1, 19), date(2024, 1, 22), date(2024, 1, 26), 100_000.0, None, None),
    )
    return AnalysisInput(
        strategy="demo",
        symbols=("AAPL",),
        starting_equity=100_000.0,
        equity_curve=tuple(curve),
        trades=trades,
        bars=bars,
        commission_per_share=_COMMISSION,
        slippage_bps=_SLIPPAGE_BPS,
        parameters={"source": "synthetic"},
        folds=folds,
    )


def _trade(
    symbol: str,
    day: date,
    entry: float,
    exit_price: float,
    quantity: int,
    r_multiple: float,
    reason: str,
) -> TradeRecord:
    cost = float(round_trip_cost(quantity, entry, exit_price, _COMMISSION, _SLIPPAGE_BPS))
    gross = (exit_price - entry) * quantity
    return TradeRecord(
        symbol=symbol,
        entry_ts=_at(day, 10, 0),
        exit_ts=_at(day, 15, 0),
        entry_price=entry,
        exit_price=exit_price,
        quantity=quantity,
        stop_price=round(entry - 2.0, 2),
        target_price=round(entry + 5.0, 2),
        pnl=gross - cost,
        r_multiple=r_multiple,
        exit_reason=reason,
    )


def _fold(
    index: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    ending: float,
    r_multiple: float | None,
    pnl: float | None,
) -> FoldInput:
    trades: tuple[TradeRecord, ...] = ()
    if pnl is not None and r_multiple is not None:
        trades = (
            TradeRecord(
                symbol="AAPL",
                entry_ts=_at(test_start, 10, 0),
                exit_ts=_at(test_end, 15, 0),
                entry_price=100.0,
                exit_price=100.0,
                quantity=10,
                stop_price=98.0,
                target_price=105.0,
                pnl=pnl,
                r_multiple=r_multiple,
                exit_reason="flatten",
            ),
        )
    return FoldInput(
        fold_index=index,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        starting_equity=100_000.0,
        equity_curve=((_at(test_end, 16, 0), ending),),
        trades=trades,
        symbols=("AAPL",),
    )


def _demo_bars(days: tuple[date, ...]) -> pl.DataFrame:
    """Up, down, a wide chop, and a mild up day. Regimes are not all the same."""
    frames = [
        _session(days[0], [100.0 + 0.05 * step for step in range(20)]),
        _session(days[1], [101.0 - 0.08 * step for step in range(20)]),
        _session(days[2], _chop(99.4)),
        _session(days[3], [99.5 + 0.02 * step for step in range(20)]),
    ]
    return pl.concat(frames).sort("ts")


def _chop(start: float) -> list[float]:
    closes = [start + (0.8 if step % 2 else 0.0) for step in range(20)]
    closes[-1] = start
    return closes


def _session(day: date, closes: list[float]) -> pl.DataFrame:
    open_local = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
    previous = closes[0]
    rows: list[dict[str, object]] = []
    for offset, close in enumerate(closes):
        ts = (open_local + timedelta(minutes=offset)).astimezone(timezone.utc)
        open_ = previous
        high = max(open_, close) + 0.02
        low = min(open_, close) - 0.02
        rows.append(
            {
                "symbol": "AAPL",
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100_000.0 + 1_000.0 * (offset % 5),
            }
        )
        previous = close
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))


def _at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(timezone.utc)
