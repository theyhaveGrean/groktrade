"""Run the EMA baseline on bars the research path is allowed to see.

:func:`run_research_backtest` loads the cache through the holdout guard.
:func:`run_demo_backtest` uses the synthetic fixture so the report can be
produced with no API key. Both install one :class:`RiskEngine` before the
strategy starts; the strategy refuses to trade if that install is skipped.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from pydantic import ValidationError

from aegis.backtest.report import PerformanceReport, build_report
from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.backtest.synthetic import synthetic_bars
from aegis.config import Settings, load_settings
from aegis.data.cache import BarCache
from aegis.data.query import load_research_bars
from aegis.holdout import DateRange, assert_dates_outside_holdout
from aegis.strategies.ema import EmaCrossParams, EmaCrossover
from aegis.strategies.harness import run_signal_on_bars


def run_research_backtest(
    settings: Settings,
    *,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    strategy: str = "ema_cross",
    fast_period: int | None = None,
    slow_period: int | None = None,
    atr_period: int | None = None,
    bar_timeframe: str | None = None,
    params: dict | None = None,
) -> PerformanceReport:
    """Backtest cached bars. Refuses any window that overlaps the holdout."""
    settings.assert_paper_mode()
    from aegis.data.pipeline import resolve_members
    from aegis.strategies import get_strategy

    chosen = resolve_members(settings, symbols, end)
    names = [member.symbol for member in chosen]
    cache = BarCache(settings.data_cache_dir)
    bars = load_research_bars(cache, names, start, end, settings.holdout)
    plugin = get_strategy(strategy)
    raw = dict(params or {})
    if strategy == "ema_cross":
        raw.setdefault("fast_period", 20 if fast_period is None else fast_period)
        raw.setdefault("slow_period", 50 if slow_period is None else slow_period)
        raw.setdefault("atr_period", 14 if atr_period is None else atr_period)
    else:
        if fast_period is not None:
            raw["fast_period"] = fast_period
        if slow_period is not None:
            raw["slow_period"] = slow_period
        if atr_period is not None:
            raw["atr_period"] = atr_period
    if bar_timeframe is not None:
        raw["bar_timeframe"] = bar_timeframe
    parsed = plugin.params_model.model_validate(raw)
    return plugin.run(
        bars,
        parsed,
        holdout=settings.holdout,
        starting_equity=settings.starting_equity,
        commission_per_share=settings.commission_per_share,
        slippage_bps=settings.slippage_bps,
        kill_switch=settings.risk_kill_switch,
        log_level=settings.nautilus_log_level,
    )


def run_demo_backtest(settings: Settings | None = None, strategy: str = "ema_cross") -> PerformanceReport:
    """Synthetic backtest. The default is the two-day EMA fixture. No API key."""
    settings = settings or load_settings()
    settings.assert_paper_mode()
    if strategy in {"ema_cross", "ema_crossover"}:
        return run_ema_on_bars(
            synthetic_bars(),
            holdout=settings.holdout,
            starting_equity=settings.starting_equity,
            fast_period=5,
            slow_period=12,
            atr_period=10,
            commission_per_share=settings.commission_per_share,
            slippage_bps=settings.slippage_bps,
            risk_engine=RiskEngine(RiskLimits(kill_switch=settings.risk_kill_switch)),
            log_level=settings.nautilus_log_level,
        )
    from aegis.backtest.synthetic import DEMO_PARAMS, synthetic_bars_for
    from aegis.strategies import get_strategy

    plugin = get_strategy(strategy)
    parsed = plugin.params_model.model_validate(DEMO_PARAMS[strategy])
    return plugin.run(
        synthetic_bars_for(strategy),
        parsed,
        holdout=settings.holdout,
        starting_equity=settings.starting_equity,
        commission_per_share=settings.commission_per_share,
        slippage_bps=settings.slippage_bps,
        kill_switch=settings.risk_kill_switch,
        log_level=settings.nautilus_log_level,
    )


def run_ema_on_bars(
    bars: pl.DataFrame,
    *,
    holdout: DateRange,
    starting_equity: float = 100_000.0,
    fast_period: int = 20,
    slow_period: int = 50,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    bar_minutes: int = 1,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
    risk_engine: RiskEngine | None = None,
    install_risk: bool = True,
    log_level: str = "ERROR",
    venue_name: str = "XNAS",
) -> PerformanceReport:
    """Run one backtest. ``holdout`` is required so there is no unguarded path.

    ``atr_multiplier`` is the ATR stop multiple used when this function builds
    the risk engine. A caller that passes ``risk_engine`` keeps that engine's
    own limits, including its multiple.
    """
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive.")
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be at least 1.")
    dates: set[date] = set()
    if not bars.is_empty():
        dates = set(
            bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).to_series().to_list()
        )
    assert_dates_outside_holdout(dates, holdout, purpose="EMA backtest")
    symbols = tuple(bars["symbol"].unique().sort().to_list()) if not bars.is_empty() else ()
    timeframe = _TIMEFRAME_BY_MINUTES.get(bar_minutes, "1min")
    try:
        params = EmaCrossParams(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            bar_timeframe=timeframe,
        )
    except ValidationError:
        return build_report(
            strategy="ema_crossover",
            symbols=symbols,
            starting_equity=starting_equity,
            equity_curve=[],
            trades=[],
            bars=bars,
            bar_timeframe=timeframe,
            commission_per_share=commission_per_share,
            slippage_bps=slippage_bps,
            risk_engine_installed=install_risk,
        )
    return run_signal_on_bars(
        bars,
        holdout=holdout,
        signal_model=EmaCrossover,
        params=params,
        strategy_name="ema_crossover",
        starting_equity=starting_equity,
        atr_period=atr_period,
        bar_minutes=bar_minutes,
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
        risk_engine=risk_engine,
        install_risk=install_risk,
        log_level=log_level,
        venue_name=venue_name,
    )


_TIMEFRAME_BY_MINUTES = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "60min"}
