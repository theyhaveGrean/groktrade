"""Run the EMA baseline on bars the research path is allowed to see.

:func:`run_research_backtest` loads the cache through the holdout guard.
:func:`run_demo_backtest` uses the synthetic fixture so the report can be
produced with no API key. Both install one :class:`RiskEngine` before the
strategy starts; the strategy refuses to trade if that install is skipped.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import polars as pl
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money

from aegis.backtest.costs import EquityCommissionSlippage
from aegis.backtest.convert import RESEARCH_VENUE, bar_type_for, equity_instrument, frame_to_bars
from aegis.backtest.report import PerformanceReport, build_report
from aegis.backtest.risk import RiskEngine, RiskLimits, clear_risk_engine, install_risk_engine
from aegis.backtest.strategy import EmaCrossConfig, EmaCrossStrategy
from aegis.backtest.synthetic import synthetic_bars
from aegis.config import Settings, load_settings
from aegis.data.cache import BarCache
from aegis.data.query import load_research_bars
from aegis.holdout import DateRange, assert_dates_outside_holdout


def run_research_backtest(
    settings: Settings,
    *,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    fast_period: int = 20,
    slow_period: int = 50,
    atr_period: int = 14,
) -> PerformanceReport:
    """Backtest cached bars. Refuses any window that overlaps the holdout."""
    settings.assert_paper_mode()
    from aegis.data.pipeline import resolve_members

    chosen = resolve_members(settings, symbols, end)
    names = [member.symbol for member in chosen]
    cache = BarCache(settings.data_cache_dir)
    bars = load_research_bars(cache, names, start, end, settings.holdout)
    return run_ema_on_bars(
        bars,
        holdout=settings.holdout,
        starting_equity=settings.starting_equity,
        fast_period=fast_period,
        slow_period=slow_period,
        atr_period=atr_period,
        commission_per_share=settings.commission_per_share,
        slippage_bps=settings.slippage_bps,
        risk_engine=RiskEngine(RiskLimits(kill_switch=settings.risk_kill_switch)),
        log_level=settings.nautilus_log_level,
    )


def run_demo_backtest(settings: Settings | None = None) -> PerformanceReport:
    """Synthetic two-day EMA backtest. Does not read the cache or the network."""
    settings = settings or load_settings()
    settings.assert_paper_mode()
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
    venue_name: str = RESEARCH_VENUE,
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
            bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date())
            .to_series()
            .to_list()
        )
    assert_dates_outside_holdout(dates, holdout, purpose="EMA backtest")
    symbols = tuple(bars["symbol"].unique().sort().to_list()) if not bars.is_empty() else ()
    if bars.is_empty():
        return build_report(
            strategy="ema_crossover",
            symbols=symbols,
            starting_equity=starting_equity,
            equity_curve=[],
            trades=[],
            bars=bars,
        )
    if install_risk:
        install_risk_engine(risk_engine or RiskEngine(RiskLimits(atr_multiplier=atr_multiplier)))
    else:
        clear_risk_engine()
    engine: BacktestEngine | None = None
    try:
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("AEGIS-001"),
                logging=LoggingConfig(log_level=log_level),
            )
        )
        venue = Venue(venue_name)
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(starting_equity, USD)],
            default_leverage=Decimal("2"),
            fee_model=EquityCommissionSlippage(commission_per_share, slippage_bps),
            bar_adaptive_high_low_ordering=True,
            reject_stop_orders=False,
        )
        instrument_ids: list[str] = []
        bar_types: list[str] = []
        for symbol in symbols:
            instrument = equity_instrument(symbol, venue_name)
            bar_type = bar_type_for(symbol, venue_name, bar_minutes)
            engine.add_instrument(instrument)
            part = bars.filter(pl.col("symbol") == symbol).sort("ts")
            engine.add_data(frame_to_bars(part, instrument, bar_type))
            instrument_ids.append(str(instrument.id))
            bar_types.append(str(bar_type))
        strategy = EmaCrossStrategy(
            EmaCrossConfig(
                instrument_ids=instrument_ids,
                bar_types=bar_types,
                fast_period=fast_period,
                slow_period=slow_period,
                atr_period=atr_period,
            )
        )
        engine.add_strategy(strategy)
        engine.run()
        return build_report(
            strategy="ema_crossover",
            symbols=symbols,
            starting_equity=starting_equity,
            equity_curve=list(strategy.equity_curve),
            trades=list(strategy.trades),
            bars=bars,
        )
    finally:
        if engine is not None:
            engine.dispose()
        clear_risk_engine()
