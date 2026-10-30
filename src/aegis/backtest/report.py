"""Standardized performance report for the EMA baseline.

R-multiples are gross price R against the initial ATR stop:
``(exit - entry) / (entry - stop)`` for a long. PnL is net of the commission
and slippage model. Buy-and-hold is the equal-weight close-to-close return of
the symbols in the run, fully invested, without costs. Sharpe uses daily
equity changes and ``sqrt(252)`` with a zero risk-free rate. It is ``None``
when fewer than two daily returns exist or the standard deviation is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: int
    stop_price: float
    target_price: float
    pnl: float
    r_multiple: float | None
    exit_reason: str


@dataclass(frozen=True)
class PerformanceReport:
    strategy: str
    symbols: tuple[str, ...]
    start: datetime | None
    end: datetime | None
    starting_equity: float
    ending_equity: float
    total_return: float
    sharpe: float | None
    sharpe_observations: int
    max_drawdown: float
    win_rate: float
    avg_r_multiple: float | None
    trade_count: int
    benchmark_return: float
    excess_return: float
    trades: tuple[TradeRecord, ...]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbols": list(self.symbols),
            "start": _iso(self.start),
            "end": _iso(self.end),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "sharpe_observations": self.sharpe_observations,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_r_multiple": self.avg_r_multiple,
            "trade_count": self.trade_count,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "trades": [
                {
                    "symbol": trade.symbol,
                    "entry_ts": _iso(trade.entry_ts),
                    "exit_ts": _iso(trade.exit_ts),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "stop_price": trade.stop_price,
                    "target_price": trade.target_price,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                    "exit_reason": trade.exit_reason,
                }
                for trade in self.trades
            ],
        }

    def render(self) -> str:
        lines = [
            f"Strategy: {self.strategy}",
            f"Symbols: {', '.join(self.symbols) if self.symbols else '-'}",
            f"Window: {_iso(self.start)} .. {_iso(self.end)}",
            f"Starting equity: {self.starting_equity:,.2f}",
            f"Ending equity: {self.ending_equity:,.2f}",
            f"Total return: {_pct(self.total_return)}",
            f"Sharpe: {_sharpe_text(self.sharpe, self.sharpe_observations)}",
            f"Max drawdown: {_pct(self.max_drawdown)}",
            f"Win rate: {_pct(self.win_rate)}",
            f"Avg R-multiple: {_num(self.avg_r_multiple)}",
            f"Trade count: {self.trade_count}",
            f"Buy-and-hold return: {_pct(self.benchmark_return)}",
            f"Excess return vs buy-and-hold: {_pct(self.excess_return)}",
        ]
        if self.trades:
            lines.append("Trades:")
            for trade in self.trades:
                r_text = "n/a" if trade.r_multiple is None else f"{trade.r_multiple:.2f}R"
                lines.append(
                    f"  {trade.symbol} {trade.exit_reason} qty={trade.quantity} "
                    f"entry={trade.entry_price:.2f} exit={trade.exit_price:.2f} "
                    f"pnl={trade.pnl:.2f} {r_text}"
                )
        return "\n".join(lines)


def build_report(
    *,
    strategy: str,
    symbols: tuple[str, ...],
    starting_equity: float,
    equity_curve: list[tuple[datetime, float]],
    trades: list[TradeRecord],
    bars: pl.DataFrame,
) -> PerformanceReport:
    ending = equity_curve[-1][1] if equity_curve else starting_equity
    total_return = _return(starting_equity, ending)
    sharpe, observations = _sharpe(equity_curve, starting_equity)
    drawdown = _max_drawdown([starting_equity, *[point[1] for point in equity_curve]])
    wins = sum(1 for trade in trades if trade.pnl > 0)
    win_rate = wins / len(trades) if trades else 0.0
    r_values = [trade.r_multiple for trade in trades if trade.r_multiple is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else None
    benchmark = buy_and_hold_return(bars)
    start = equity_curve[0][0] if equity_curve else None
    end = equity_curve[-1][0] if equity_curve else None
    return PerformanceReport(
        strategy=strategy,
        symbols=symbols,
        start=start,
        end=end,
        starting_equity=starting_equity,
        ending_equity=ending,
        total_return=total_return,
        sharpe=sharpe,
        sharpe_observations=observations,
        max_drawdown=drawdown,
        win_rate=win_rate,
        avg_r_multiple=avg_r,
        trade_count=len(trades),
        benchmark_return=benchmark,
        excess_return=total_return - benchmark,
        trades=tuple(trades),
    )


def buy_and_hold_return(bars: pl.DataFrame) -> float:
    """Equal-weight close-to-close return. Zero when a symbol has no bars."""
    if bars.is_empty():
        return 0.0
    returns: list[float] = []
    for symbol in bars["symbol"].unique().sort().to_list():
        part = bars.filter(pl.col("symbol") == symbol).sort("ts")
        if part.height < 2:
            continue
        first = float(part["close"][0])
        last = float(part["close"][-1])
        if first > 0:
            returns.append(last / first - 1.0)
    if not returns:
        return 0.0
    return sum(returns) / len(returns)


def _sharpe(
    equity_curve: list[tuple[datetime, float]],
    starting_equity: float,
) -> tuple[float | None, int]:
    if starting_equity <= 0 or not equity_curve:
        return None, 0
    by_day: dict = {}
    for ts, equity in equity_curve:
        by_day[_eastern_date(ts)] = equity
    days = sorted(by_day)
    returns: list[float] = []
    previous = starting_equity
    for day in days:
        equity = by_day[day]
        returns.append(equity / previous - 1.0)
        previous = equity
    if len(returns) < 2:
        return None, len(returns)
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    if variance <= 0:
        return None, len(returns)
    return mean / math.sqrt(variance) * math.sqrt(252), len(returns)


def _max_drawdown(equities: list[float]) -> float:
    peak = equities[0] if equities else 0.0
    worst = 0.0
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def _return(start: float, end: float) -> float:
    if start <= 0:
        return 0.0
    return end / start - 1.0


def _eastern_date(ts: datetime):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(NY).date()


def _iso(ts: datetime | None) -> str:
    if ts is None:
        return "-"
    return ts.astimezone(timezone.utc).isoformat()


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _sharpe_text(value: float | None, observations: int) -> str:
    if value is None:
        return f"n/a ({observations} daily returns)"
    return f"{value:.2f} ({observations} daily returns)"
