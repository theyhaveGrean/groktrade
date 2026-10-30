"""Standardized performance report.

R-multiples are net of costs. Realized PnL includes commission and the
slippage charged as a fee, and R divides that PnL by the initial risk in
dollars (stop distance times shares). Gross price R,
``(exit - entry) / (entry - stop)``, is not what the report prints. Buy-and-hold
is the equal-weight close-to-close return of the symbols in the run, fully
invested, without costs. Sharpe uses daily equity changes and ``sqrt(252)``
with a zero risk-free rate. It is ``None`` when fewer than two daily returns
exist or the standard deviation is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import BaseModel, ConfigDict

from aegis.backtest.risk import RiskLimits

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
    equity_curve: tuple[tuple[datetime, float], ...] = ()
    schema_version: str = "1.1"
    parameters: dict[str, Any] = field(default_factory=dict)
    commission_per_share: float | None = None
    slippage_bps: float | None = None
    risk_limits: dict[str, Any] = field(default_factory=dict)
    risk_engine_installed: bool = True
    bar_timeframe: str = "1min"

    def to_dict(self) -> dict:
        daily = _daily_rows(self.equity_curve, self.starting_equity)
        returns = [row["return"] for row in daily]
        profit_factor, expectancy, loss_streak = _trade_stats(self.trades)
        payload = {
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
            "schema_version": self.schema_version,
            "parameters": dict(self.parameters),
            "commission_per_share": self.commission_per_share,
            "slippage_bps": self.slippage_bps,
            "risk_limits": dict(self.risk_limits),
            "risk_engine_installed": self.risk_engine_installed,
            "bar_timeframe": self.bar_timeframe,
            "profit_factor": profit_factor,
            "sortino": _sortino(returns),
            "expectancy": expectancy,
            "max_consecutive_losses": loss_streak,
            "max_drawdown_duration_days": _drawdown_duration_days(
                [self.starting_equity, *[point[1] for point in _last_equity_by_day(self.equity_curve)]]
            ),
            "daily_returns": daily,
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
        return payload

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
            f"Avg R-multiple (net of costs): {_num(self.avg_r_multiple)}",
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
    parameters: dict[str, Any] | None = None,
    commission_per_share: float | None = None,
    slippage_bps: float | None = None,
    risk_limits: dict[str, Any] | None = None,
    risk_engine_installed: bool = True,
    bar_timeframe: str = "1min",
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
        equity_curve=tuple(equity_curve),
        parameters=dict(parameters or {}),
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
        risk_limits=dict(risk_limits or {}),
        risk_engine_installed=risk_engine_installed,
        bar_timeframe=bar_timeframe,
    )


class BacktestReport(BaseModel):
    """JSON document the Backtesting Analyst agent reads.

    Built from :meth:`PerformanceReport.to_dict`. Costs and the risk-limit
    snapshot are descriptive. The deterministic risk engine is still what
    approved every order.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.1"
    strategy: str
    symbols: list[str]
    start: str
    end: str
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
    parameters: dict[str, Any] = {}
    commission_per_share: float | None = None
    slippage_bps: float | None = None
    risk_limits: dict[str, Any] = {}
    risk_engine_installed: bool = True
    bar_timeframe: str = "1min"
    profit_factor: float | None = None
    sortino: float | None = None
    expectancy: float | None = None
    max_consecutive_losses: int = 0
    max_drawdown_duration_days: int = 0
    daily_returns: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    @classmethod
    def from_performance(cls, report: PerformanceReport) -> BacktestReport:
        return cls.model_validate(report.to_dict())


def risk_limits_snapshot(limits: RiskLimits) -> dict[str, Any]:
    """Copy of the limits that were installed. Not a live handle."""
    return {
        "max_risk_per_trade": limits.max_risk_per_trade,
        "max_positions": limits.max_positions,
        "max_position_equity_fraction": limits.max_position_equity_fraction,
        "daily_loss_cutoff": limits.daily_loss_cutoff,
        "reward_to_risk": limits.reward_to_risk,
        "atr_multiplier": limits.atr_multiplier,
        "entry_start": limits.entry_start.strftime("%H:%M"),
        "entry_end": limits.entry_end.strftime("%H:%M"),
        "flatten_time": limits.flatten_time.strftime("%H:%M"),
        "kill_switch": bool(limits.kill_switch),
    }


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


def _last_equity_by_day(equity_curve: tuple[tuple[datetime, float], ...] | list) -> list[tuple[datetime, float]]:
    by_day: dict = {}
    order: list = []
    for ts, equity in equity_curve:
        day = _eastern_date(ts)
        if day not in by_day:
            order.append(day)
        by_day[day] = (ts, equity)
    return [by_day[day] for day in order]


def _daily_rows(equity_curve, starting_equity: float) -> list[dict[str, Any]]:
    if starting_equity <= 0 or not equity_curve:
        return []
    rows: list[dict[str, Any]] = []
    previous = starting_equity
    for ts, equity in _last_equity_by_day(equity_curve):
        rows.append({"date": _eastern_date(ts).isoformat(), "return": equity / previous - 1.0})
        previous = equity
    return rows


def _sortino(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    downside = [min(value, 0.0) ** 2 for value in returns]
    variance = sum(downside) / (len(returns) - 1)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(252)


def _trade_stats(trades: tuple[TradeRecord, ...]) -> tuple[float | None, float | None, int]:
    if not trades:
        return None, None, 0
    wins = sum(trade.pnl for trade in trades if trade.pnl > 0)
    losses = sum(trade.pnl for trade in trades if trade.pnl < 0)
    profit_factor = None if losses == 0 else wins / abs(losses)
    expectancy = sum(trade.pnl for trade in trades) / len(trades)
    streak = 0
    worst = 0
    for trade in trades:
        if trade.pnl < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return profit_factor, expectancy, worst


def _drawdown_duration_days(equities: list[float]) -> int:
    if not equities:
        return 0
    peak = equities[0]
    peak_index = 0
    worst = 0.0
    duration = 0
    for index, equity in enumerate(equities):
        if equity >= peak:
            peak = equity
            peak_index = index
        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > worst:
                worst = drawdown
                duration = index - peak_index
    return duration
