"""Ex-post trend and volatility buckets for a finished window.

Trend uses that session's equal-weight last-close / first-close return.
Volatility uses the equal-weight standard deviation of one-minute returns,
split at the median of this window. Both labels describe the sample after
the fact. They are not features a strategy is allowed to trade on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from aegis.backtest.report import TradeRecord
from aegis.data.sessions import eastern_date
from aegis.errors import AegisError

# Five basis points. Smaller sessions are labeled range rather than trend.
RANGE_BAND = 0.0005


@dataclass(frozen=True)
class RegimeSlice:
    regime: str
    sessions: int
    strategy_days: int
    sum_daily_return: float
    mean_daily_return: float | None
    trade_count: int
    trade_pnl: float

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "sessions": self.sessions,
            "strategy_days": self.strategy_days,
            "sum_daily_return": self.sum_daily_return,
            "mean_daily_return": self.mean_daily_return,
            "trade_count": self.trade_count,
            "trade_pnl": self.trade_pnl,
        }


@dataclass
class _Bucket:
    sessions: int = 0
    strategy_days: int = 0
    sum_daily_return: float = 0.0
    trade_count: int = 0
    trade_pnl: float = 0.0


def regime_breakdown(
    bars: pl.DataFrame | None,
    daily_returns: list[tuple[date, float]],
    trades: tuple[TradeRecord, ...] | list[TradeRecord],
) -> tuple[RegimeSlice, ...]:
    """Attribute strategy days and entry-day trade pnl to market regimes.

    Without bars there is no market label, so the breakdown is empty rather
    than a single unlabeled bucket.
    """
    labels = classify_sessions(bars)
    if not labels:
        return ()
    buckets: dict[str, _Bucket] = {}
    for session, regime in labels.items():
        buckets.setdefault(regime, _Bucket()).sessions += 1
    for session, daily_return in daily_returns:
        regime = labels.get(session, "unlabeled")
        bucket = buckets.setdefault(regime, _Bucket())
        bucket.strategy_days += 1
        bucket.sum_daily_return += daily_return
    for trade in trades:
        regime = labels.get(eastern_date(trade.entry_ts), "unlabeled")
        bucket = buckets.setdefault(regime, _Bucket())
        bucket.trade_count += 1
        bucket.trade_pnl += trade.pnl
    slices = [
        RegimeSlice(
            regime=name,
            sessions=bucket.sessions,
            strategy_days=bucket.strategy_days,
            sum_daily_return=bucket.sum_daily_return,
            mean_daily_return=(
                bucket.sum_daily_return / bucket.strategy_days if bucket.strategy_days else None
            ),
            trade_count=bucket.trade_count,
            trade_pnl=bucket.trade_pnl,
        )
        for name, bucket in buckets.items()
        if bucket.sessions or bucket.strategy_days or bucket.trade_count
    ]
    return tuple(sorted(slices, key=lambda item: item.regime))


def classify_sessions(bars: pl.DataFrame | None) -> dict[date, str]:
    """Map each Eastern session to ``trend_up|vol_high`` style labels."""
    if bars is None or bars.is_empty():
        return {}
    missing = {"symbol", "ts", "close"} - set(bars.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise AegisError(f"Regime bars are missing column(s): {joined}.")
    timezone_name = getattr(bars.schema["ts"], "time_zone", None)
    if timezone_name is None:
        raise AegisError("Regime bars need timezone-aware timestamps.")
    ordered = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    )
    ordered = ordered.with_columns(
        pl.when(pl.col("session") == pl.col("session").shift(1).over("symbol"))
        .then(pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
        .otherwise(None)
        .alias("minute_return")
    )
    per_symbol = ordered.group_by(["symbol", "session"], maintain_order=True).agg(
        pl.col("close").first().alias("first_close"),
        pl.col("close").last().alias("last_close"),
        pl.col("minute_return").std().alias("vol"),
    )
    by_session: dict[date, list[tuple[float, float]]] = {}
    for row in per_symbol.iter_rows(named=True):
        first = float(row["first_close"])
        last = float(row["last_close"])
        if first <= 0:
            continue
        vol = 0.0 if row["vol"] is None else float(row["vol"])
        by_session.setdefault(row["session"], []).append((last / first - 1.0, vol))
    if not by_session:
        return {}
    session_rows = []
    for session, parts in by_session.items():
        session_rows.append(
            (
                session,
                sum(item[0] for item in parts) / len(parts),
                sum(item[1] for item in parts) / len(parts),
            )
        )
    vols = sorted(item[2] for item in session_rows)
    mid = len(vols) // 2
    median_vol = vols[mid] if len(vols) % 2 else (vols[mid - 1] + vols[mid]) / 2
    return {session: _label(ret, vol, median_vol) for session, ret, vol in session_rows}


def _label(session_return: float, vol: float, median_vol: float) -> str:
    if session_return > RANGE_BAND:
        trend = "trend_up"
    elif session_return < -RANGE_BAND:
        trend = "trend_down"
    else:
        trend = "range"
    vol_name = "vol_high" if vol >= median_vol else "vol_low"
    return f"{trend}|{vol_name}"
