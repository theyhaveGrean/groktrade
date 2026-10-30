"""Standardized backtest report: JSON and markdown from one result bundle.

The headline performance block uses the same fields as the phase-2 report.
Benchmark, walk-forward, cost multiples, and regimes sit beside it. Building
the report refuses any timestamp or fold window that overlaps the holdout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from aegis.analyst.costs import scale_equity_curve, scale_trades
from aegis.analyst.input import SCHEMA_VERSION, AnalysisInput
from aegis.analyst.regimes import RegimeSlice, regime_breakdown
from aegis.analyst.walkforward import WalkForwardSummary, summarize_walk_forward
from aegis.backtest.report import build_report, buy_and_hold_return
from aegis.data.sessions import eastern_date
from aegis.errors import AegisError
from aegis.holdout import DateRange, assert_dates_outside_holdout, assert_no_holdout_overlap

NOTES = (
    "R-multiples are gross price R against the recorded stop. Cost scenarios rescale net pnl and the equity curve; they do not change R.",
    "Buy-and-hold is the equal-weight close-to-close return of the supplied bars, fully invested, without costs.",
    "Walk-forward numbers use only the out-of-sample folds supplied with the result. This report does not run a backtest.",
    "Trend and volatility regimes are ex-post labels on this window. A session within 5 bps of flat is range. Volatility is split at the sample median.",
    "This report does not change the deterministic risk engine.",
)


@dataclass(frozen=True)
class BenchmarkComparison:
    name: str
    benchmark_return: float | None
    strategy_return: float
    excess_return: float | None
    note: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "benchmark_return": self.benchmark_return,
            "strategy_return": self.strategy_return,
            "excess_return": self.excess_return,
            "note": self.note,
        }


@dataclass(frozen=True)
class CostScenario:
    multiplier: float
    total_return: float
    sharpe: float | None
    sharpe_observations: int
    max_drawdown: float
    win_rate: float
    ending_equity: float
    avg_r_multiple: float | None
    trade_count: int
    excess_return: float | None

    def to_dict(self) -> dict:
        return {
            "multiplier": self.multiplier,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "sharpe_observations": self.sharpe_observations,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "ending_equity": self.ending_equity,
            "avg_r_multiple": self.avg_r_multiple,
            "trade_count": self.trade_count,
            "excess_return": self.excess_return,
        }


@dataclass(frozen=True)
class AnalystReport:
    strategy: str
    symbols: tuple[str, ...]
    parameters: dict
    performance: dict
    benchmark: BenchmarkComparison
    walk_forward: WalkForwardSummary
    cost_sensitivity: tuple[CostScenario, ...]
    regimes: tuple[RegimeSlice, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "strategy": self.strategy,
            "symbols": list(self.symbols),
            "parameters": self.parameters,
            "performance": self.performance,
            "benchmark": self.benchmark.to_dict(),
            "walk_forward": self.walk_forward.to_dict(),
            "cost_sensitivity": [scenario.to_dict() for scenario in self.cost_sensitivity],
            "regimes": [regime.to_dict() for regime in self.regimes],
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        performance = self.performance
        lines = [
            f"# Backtest analyst report: {self.strategy}",
            "",
            "## Performance",
            "",
            f"- Symbols: {', '.join(self.symbols) if self.symbols else '-'}",
            f"- Window: {performance.get('start', '-')} .. {performance.get('end', '-')}",
            f"- Starting equity: {_num(performance.get('starting_equity'))}",
            f"- Ending equity: {_num(performance.get('ending_equity'))}",
            f"- Total return: {_pct(performance.get('total_return'))}",
            f"- Sharpe: {_sharpe_text(performance.get('sharpe'), performance.get('sharpe_observations', 0))}",
            f"- Max drawdown: {_pct(performance.get('max_drawdown'))}",
            f"- Win rate: {_pct(performance.get('win_rate'))}",
            f"- Avg R-multiple: {_r(performance.get('avg_r_multiple'))}",
            f"- Trade count: {performance.get('trade_count', 0)}",
            "",
            "## Benchmark",
            "",
            f"- Name: {self.benchmark.name}",
            f"- Buy-and-hold return: {_pct(self.benchmark.benchmark_return)}",
            f"- Strategy return: {_pct(self.benchmark.strategy_return)}",
            f"- Excess return: {_pct(self.benchmark.excess_return)}",
            f"- Note: {self.benchmark.note}",
            "",
            "## Walk-forward out of sample",
            "",
        ]
        summary = self.walk_forward
        if not summary.folds:
            lines.append("No out-of-sample folds were supplied.")
        else:
            lines.extend(
                [
                    f"- Folds: {len(summary.folds)}",
                    f"- Mean OOS return: {_pct(summary.mean_return)}",
                    f"- Median OOS return: {_pct(summary.median_return)}",
                    f"- Min OOS return: {_pct(summary.min_return)}",
                    f"- Max OOS return: {_pct(summary.max_return)}",
                    f"- Fraction positive: {_pct(summary.fraction_positive)}",
                    f"- Compounded OOS return: {_pct(summary.compounded_return)}",
                    f"- OOS trades: {summary.total_trades}",
                    f"- Trade-weighted avg R: {_r(summary.mean_avg_r)}",
                    "",
                    "| Fold | Train | Test | Return | Sharpe | Max DD | Trades | Avg R | Excess |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for fold in summary.folds:
                lines.append(
                    "| {index} | {train} | {test} | {ret} | {sharpe} | {dd} | {trades} | {avg_r} | {excess} |".format(
                        index=fold.fold_index,
                        train=f"{fold.train_start.isoformat()}..{fold.train_end.isoformat()}",
                        test=f"{fold.test_start.isoformat()}..{fold.test_end.isoformat()}",
                        ret=_pct(fold.total_return),
                        sharpe=_sharpe_text(fold.sharpe, fold.sharpe_observations),
                        dd=_pct(fold.max_drawdown),
                        trades=fold.trade_count,
                        avg_r=_r(fold.avg_r_multiple),
                        excess=_pct(fold.excess_return),
                    )
                )
        lines.extend(["", "## Cost and slippage sensitivity", ""])
        lines.append("| Multiple | Total return | Sharpe | Max DD | Win rate | Ending equity | Avg R | Excess |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for scenario in self.cost_sensitivity:
            lines.append(
                "| {mult} | {ret} | {sharpe} | {dd} | {win} | {ending} | {avg_r} | {excess} |".format(
                    mult=_multiple(scenario.multiplier),
                    ret=_pct(scenario.total_return),
                    sharpe=_sharpe_text(scenario.sharpe, scenario.sharpe_observations),
                    dd=_pct(scenario.max_drawdown),
                    win=_pct(scenario.win_rate),
                    ending=f"{scenario.ending_equity:,.2f}",
                    avg_r=_r(scenario.avg_r_multiple),
                    excess=_pct(scenario.excess_return),
                )
            )
        lines.extend(["", "## Regime breakdown", ""])
        if not self.regimes:
            lines.append("No bars were supplied, so regimes were not labeled.")
        else:
            lines.append(
                "| Regime | Sessions | Strategy days | Sum of daily returns | Mean daily return | Trades | Trade pnl |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for regime in self.regimes:
                lines.append(
                    f"| {regime.regime} | {regime.sessions} | {regime.strategy_days} | "
                    f"{_pct(regime.sum_daily_return)} | {_pct(regime.mean_daily_return)} | "
                    f"{regime.trade_count} | {regime.trade_pnl:,.2f} |"
                )
        if self.parameters:
            lines.extend(["", "## Parameters", ""])
            for key in sorted(self.parameters):
                lines.append(f"- {key}: {self.parameters[key]}")
        lines.extend(["", "## Notes", ""])
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines)


def build_analyst_report(
    analysis: AnalysisInput,
    *,
    holdout: DateRange,
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> AnalystReport:
    """Build the report. ``holdout`` is required so a research path cannot skip it."""
    if not cost_multipliers:
        raise ValueError("cost_multipliers must not be empty.")
    if any(multiplier < 0 for multiplier in cost_multipliers):
        raise ValueError("Cost multipliers cannot be negative.")
    _refuse_holdout(analysis, holdout)
    curve = _resolved_curve(analysis)
    bars = analysis.bars
    performance = build_report(
        strategy=analysis.strategy,
        symbols=analysis.symbols,
        starting_equity=analysis.starting_equity,
        equity_curve=curve,
        trades=list(analysis.trades),
        bars=_bars_or_empty(bars),
    )
    benchmark = _benchmark(performance.total_return, bars)
    payload = performance.to_dict()
    payload["benchmark_return"] = benchmark.benchmark_return
    payload["excess_return"] = benchmark.excess_return
    scenarios = tuple(
        _cost_scenario(analysis, curve, multiplier, benchmark.benchmark_return) for multiplier in cost_multipliers
    )
    regimes = regime_breakdown(bars, _daily_returns(analysis.starting_equity, curve), analysis.trades)
    notes = list(NOTES)
    if bars is None or bars.is_empty():
        notes.append("Bars were not supplied. Benchmark and regime sections stay empty.")
    if not analysis.folds:
        notes.append("Walk-forward section is empty until OOS fold results are attached.")
    return AnalystReport(
        strategy=analysis.strategy,
        symbols=analysis.symbols,
        parameters=dict(analysis.parameters),
        performance=payload,
        benchmark=benchmark,
        walk_forward=summarize_walk_forward(analysis.folds, bars),
        cost_sensitivity=scenarios,
        regimes=regimes,
        notes=tuple(notes),
    )


def write_analyst_report(report: AnalystReport, directory: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "report.md"
    json_path = directory / "report.json"
    markdown_path.write_text(report.to_markdown() + "\n")
    json_path.write_text(report.to_json() + "\n")
    return markdown_path, json_path


def _cost_scenario(
    analysis: AnalysisInput,
    curve: list[tuple[datetime, float]],
    multiplier: float,
    benchmark_return: float | None,
) -> CostScenario:
    trades = scale_trades(
        analysis.trades,
        multiplier,
        commission_per_share=analysis.commission_per_share,
        slippage_bps=analysis.slippage_bps,
    )
    scaled_curve = scale_equity_curve(
        curve,
        analysis.trades,
        multiplier,
        commission_per_share=analysis.commission_per_share,
        slippage_bps=analysis.slippage_bps,
    )
    report = build_report(
        strategy=analysis.strategy,
        symbols=analysis.symbols,
        starting_equity=analysis.starting_equity,
        equity_curve=scaled_curve,
        trades=trades,
        bars=_bars_or_empty(None),
    )
    excess = None if benchmark_return is None else report.total_return - benchmark_return
    return CostScenario(
        multiplier=float(multiplier),
        total_return=report.total_return,
        sharpe=report.sharpe,
        sharpe_observations=report.sharpe_observations,
        max_drawdown=report.max_drawdown,
        win_rate=report.win_rate,
        ending_equity=report.ending_equity,
        avg_r_multiple=report.avg_r_multiple,
        trade_count=report.trade_count,
        excess_return=excess,
    )


def _benchmark(strategy_return: float, bars: pl.DataFrame | None) -> BenchmarkComparison:
    if bars is None or bars.is_empty():
        return BenchmarkComparison(
            name="equal_weight_buy_and_hold",
            benchmark_return=None,
            strategy_return=strategy_return,
            excess_return=None,
            note="No bars were supplied, so buy-and-hold was not computed.",
        )
    benchmark_return = buy_and_hold_return(bars)
    return BenchmarkComparison(
        name="equal_weight_buy_and_hold",
        benchmark_return=benchmark_return,
        strategy_return=strategy_return,
        excess_return=strategy_return - benchmark_return,
        note="Buy-and-hold is close-to-close and fully invested. It does not pay the strategy's costs or sit in cash intraday.",
    )


def _resolved_curve(analysis: AnalysisInput) -> list[tuple[datetime, float]]:
    if analysis.equity_curve:
        return sorted(analysis.equity_curve, key=lambda point: point[0])
    equity = analysis.starting_equity
    curve: list[tuple[datetime, float]] = []
    for trade in sorted(analysis.trades, key=lambda item: item.exit_ts):
        equity += trade.pnl
        curve.append((trade.exit_ts, equity))
    return curve


def _daily_returns(starting_equity: float, curve: list[tuple[datetime, float]]) -> list[tuple[date, float]]:
    marks = _end_of_day_equity(curve)
    returns: list[tuple[date, float]] = []
    previous = starting_equity
    for session, equity in marks:
        if previous > 0:
            returns.append((session, equity / previous - 1.0))
        else:
            returns.append((session, 0.0))
        previous = equity
    return returns


def _refuse_holdout(analysis: AnalysisInput, holdout: DateRange) -> None:
    dates = _dates_in_curve(analysis.equity_curve)
    dates.update(_dates_in_trades(analysis.trades))
    dates.update(_bar_dates(analysis.bars))
    for fold in analysis.folds:
        dates.update(_dates_in_curve(fold.equity_curve))
        dates.update(_dates_in_trades(fold.trades))
        _refuse_range(fold.train_start, fold.train_end, holdout, f"walk-forward train fold {fold.fold_index}")
        _refuse_range(fold.test_start, fold.test_end, holdout, f"walk-forward test fold {fold.fold_index}")
    assert_dates_outside_holdout(dates, holdout, purpose="backtest analyst report")


def _refuse_range(start: date, end: date, holdout: DateRange, purpose: str) -> None:
    assert_no_holdout_overlap(DateRange(start, end), holdout, purpose=purpose)


def _end_of_day_equity(curve: list[tuple[datetime, float]]) -> list[tuple[date, float]]:
    by_day: dict[date, float] = {}
    for ts, equity in curve:
        by_day[eastern_date(ts)] = equity
    return [(day, by_day[day]) for day in sorted(by_day)]


def _dates_in_curve(curve: tuple[tuple[datetime, float], ...]) -> set[date]:
    return {eastern_date(ts) for ts, _equity in curve}


def _dates_in_trades(trades: tuple) -> set[date]:
    dates: set[date] = set()
    for trade in trades:
        dates.add(eastern_date(trade.entry_ts))
        dates.add(eastern_date(trade.exit_ts))
    return dates


def _bar_dates(bars: pl.DataFrame | None) -> set[date]:
    if bars is None or bars.is_empty():
        return set()
    if "ts" not in bars.columns:
        raise AegisError("Bars are missing a ts column.")
    timezone_name = getattr(bars.schema["ts"], "time_zone", None)
    if timezone_name is None:
        raise AegisError("Bar timestamps must be timezone-aware.")
    return set(
        bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).to_series().to_list()
    )


def _bars_or_empty(bars: pl.DataFrame | None) -> pl.DataFrame:
    if bars is None:
        return pl.DataFrame(
            schema={"symbol": pl.String, "ts": pl.Datetime("us", "UTC"), "close": pl.Float64}
        )
    return bars


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _r(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _sharpe_text(value: float | None, observations: int | None) -> str:
    count = 0 if observations is None else observations
    if value is None:
        return f"n/a ({count} daily returns)"
    return f"{value:.2f} ({count} daily returns)"


def _multiple(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"
