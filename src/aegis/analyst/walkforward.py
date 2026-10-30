"""Summarize out-of-sample folds the caller already ran.

Fold curves are taken as given. This module does not split a single equity
curve and call it walk-forward, and it does not launch a backtest.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date

import polars as pl

from aegis.analyst.input import FoldInput
from aegis.backtest.report import build_report


@dataclass(frozen=True)
class FoldSummary:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    total_return: float
    sharpe: float | None
    sharpe_observations: int
    max_drawdown: float
    trade_count: int
    win_rate: float
    avg_r_multiple: float | None
    ending_equity: float
    benchmark_return: float | None
    excess_return: float | None

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "sharpe_observations": self.sharpe_observations,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "avg_r_multiple": self.avg_r_multiple,
            "ending_equity": self.ending_equity,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
        }


@dataclass(frozen=True)
class WalkForwardSummary:
    folds: tuple[FoldSummary, ...]
    mean_return: float | None
    median_return: float | None
    min_return: float | None
    max_return: float | None
    fraction_positive: float | None
    compounded_return: float | None
    total_trades: int
    mean_avg_r: float | None

    def to_dict(self) -> dict:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "fraction_positive": self.fraction_positive,
            "compounded_return": self.compounded_return,
            "total_trades": self.total_trades,
            "mean_avg_r": self.mean_avg_r,
        }


def summarize_walk_forward(
    folds: tuple[FoldInput, ...] | list[FoldInput],
    bars: pl.DataFrame | None,
) -> WalkForwardSummary:
    ordered = tuple(sorted(folds, key=lambda fold: fold.fold_index))
    summaries = tuple(_summarize_fold(fold, bars) for fold in ordered)
    returns = [fold.total_return for fold in summaries]
    if not summaries:
        return WalkForwardSummary(
            folds=(),
            mean_return=None,
            median_return=None,
            min_return=None,
            max_return=None,
            fraction_positive=None,
            compounded_return=None,
            total_trades=0,
            mean_avg_r=None,
        )
    compounded = 1.0
    for value in returns:
        compounded *= 1.0 + value
    weighted_r = 0.0
    weighted_n = 0
    for fold in summaries:
        if fold.avg_r_multiple is None or fold.trade_count == 0:
            continue
        weighted_r += fold.avg_r_multiple * fold.trade_count
        weighted_n += fold.trade_count
    return WalkForwardSummary(
        folds=summaries,
        mean_return=sum(returns) / len(returns),
        median_return=float(statistics.median(returns)),
        min_return=min(returns),
        max_return=max(returns),
        fraction_positive=sum(1 for value in returns if value > 0) / len(returns),
        compounded_return=compounded - 1.0,
        total_trades=sum(fold.trade_count for fold in summaries),
        mean_avg_r=weighted_r / weighted_n if weighted_n else None,
    )


def _summarize_fold(fold: FoldInput, bars: pl.DataFrame | None) -> FoldSummary:
    window = _test_bars(bars, fold.test_start, fold.test_end)
    report = build_report(
        strategy=f"fold_{fold.fold_index}",
        symbols=fold.symbols,
        starting_equity=fold.starting_equity,
        equity_curve=list(fold.equity_curve),
        trades=list(fold.trades),
        bars=window if window is not None else _empty_bars(),
    )
    benchmark = None if window is None else report.benchmark_return
    excess = None if benchmark is None else report.total_return - benchmark
    return FoldSummary(
        fold_index=fold.fold_index,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        total_return=report.total_return,
        sharpe=report.sharpe,
        sharpe_observations=report.sharpe_observations,
        max_drawdown=report.max_drawdown,
        trade_count=report.trade_count,
        win_rate=report.win_rate,
        avg_r_multiple=report.avg_r_multiple,
        ending_equity=report.ending_equity,
        benchmark_return=benchmark,
        excess_return=excess,
    )


def _test_bars(bars: pl.DataFrame | None, start: date, end: date) -> pl.DataFrame | None:
    if bars is None or bars.is_empty():
        return None
    marked = bars.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    )
    window = marked.filter((pl.col("session") >= start) & (pl.col("session") <= end)).drop("session")
    if window.is_empty():
        return None
    return window


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "ts": pl.Datetime("us", "UTC"),
            "close": pl.Float64,
        }
    )
