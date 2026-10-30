"""Fixed execution harness.

Subscriptions, indicator registration, risk-engine sizing, bracket orders,
the entry window, the 15:55 flatten, the daily loss cutoff, and the kill
switch live here. Signal models only see :class:`~aegis.strategies.signal.BarView`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, timezone

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import ContingencyType, OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from aegis.backtest.report import PerformanceReport, TradeRecord, build_report, risk_limits_snapshot
from aegis.backtest.risk import AtrState, PortfolioSnapshot, RiskEngine, get_risk_engine
from aegis.data.sessions import eastern_date
from aegis.errors import RiskEngineNotInstalled
from aegis.strategies.params import StrategyParams
from aegis.strategies.signal import IndicatorSpec, Signal, SignalAction, SignalModel


class HarnessConfig(StrategyConfig, frozen=True):
    instrument_ids: list[str]
    bar_types: list[str]
    strategy_name: str
    signal_path: str
    params_path: str
    params_json: str
    atr_period: int = 14


@dataclass
class _Plan:
    stop: float
    target: float
    side: str


class HarnessStrategy(Strategy):
    """One Nautilus strategy for every registered signal."""

    def __init__(self, config: HarnessConfig) -> None:
        super().__init__(config)
        self._models: dict[str, SignalModel] = {}
        self._indicators: dict[str, dict[str, ExponentialMovingAverage]] = {}
        self._atr: dict[str, AtrState] = {}
        self._plans: dict[str, _Plan] = {}
        self._exit_reasons: dict[str, str] = {}
        self._halted_days: set = set()
        self._day = None
        self._day_start_equity = 0.0
        self._risk: RiskEngine | None = None
        self._disabled = False
        self._params: StrategyParams | None = None
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[TradeRecord] = []

    def on_start(self) -> None:
        try:
            params_cls = _load(self.config.params_path)
            model_cls = _load(self.config.signal_path)
            params = params_cls.model_validate_json(self.config.params_json)
        except Exception as exc:
            self.log.error(f"invalid strategy params; refusing to trade: {exc}")
            self._disabled = True
            return
        try:
            self._risk = get_risk_engine()
        except RiskEngineNotInstalled:
            self.log.error("deterministic risk engine is not installed; refusing to trade")
            self._disabled = True
            return
        self._params = params
        for instrument_id, bar_type in zip(self.config.instrument_ids, self.config.bar_types, strict=True):
            symbol = InstrumentId.from_str(instrument_id).symbol.value
            parsed = BarType.from_str(bar_type)
            model = model_cls(params)
            self._models[symbol] = model
            self._indicators[symbol] = {}
            for spec in model.indicator_specs():
                indicator = _build_indicator(spec)
                self.register_indicator_for_bars(parsed, indicator)
                self._indicators[symbol][spec.name] = indicator
            self.subscribe_bars(parsed)
            self._atr[symbol] = AtrState(self.config.atr_period)

    def on_bar(self, bar: Bar) -> None:
        if self._disabled or self._risk is None:
            return
        instrument_id = bar.bar_type.instrument_id
        symbol = instrument_id.symbol.value
        ts = _utc(bar.ts_event)
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)
        atr = self._atr[symbol].update(high, low, close)
        equity = self._equity(instrument_id.venue)
        self.equity_curve.append((ts, equity))
        self._roll_day(ts, equity)
        snapshot = PortfolioSnapshot(
            equity=equity,
            day_start_equity=self._day_start_equity,
            open_symbols=self._open_symbols(),
        )
        session = eastern_date(ts)
        view = _bar_view(symbol, ts, bar, atr, self._indicators[symbol])
        signal = self._models[symbol].on_bar(view)
        exit_reason = self._forced_exit_reason(ts, session, snapshot)
        if exit_reason is not None:
            self._flatten(instrument_id, exit_reason)
            return
        if self._in_position(instrument_id):
            if signal.action is SignalAction.EXIT:
                self._flatten(instrument_id, "signal")
            return
        if signal.action not in {SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT}:
            return
        if atr is None:
            return
        if self._busy(instrument_id):
            return
        side = "long" if signal.action is SignalAction.ENTER_LONG else "short"
        decision = self._risk.evaluate_entry(
            symbol=symbol,
            ts=ts,
            side=side,
            entry=close,
            atr=atr,
            snapshot=snapshot,
        )
        if decision.flatten:
            self._flatten(instrument_id, decision.reason)
            return
        if not decision.approved or decision.quantity < 1:
            return
        if decision.stop_price is None or decision.target_price is None:
            return
        stop, target = resolve_bracket(
            side=side,
            entry=close,
            engine_stop=decision.stop_price,
            engine_target=decision.target_price,
            hint_stop=signal.stop_price,
            hint_target=signal.target_price,
            reward_to_risk=self._risk.limits.reward_to_risk,
        )
        self._submit_bracket(instrument_id, side, decision.quantity, stop, target)

    def on_position_closed(self, event) -> None:
        symbol = event.instrument_id.symbol.value
        plan = self._plans.pop(symbol, None)
        stamped = self._exit_reasons.pop(symbol, None)
        entry = float(event.avg_px_open)
        exit_price = float(event.avg_px_close)
        reason = resolve_exit_reason(
            stamped=stamped,
            closing_tags=_closing_tags(self, event),
            exit_price=exit_price,
            stop=None if plan is None else plan.stop,
            target=None if plan is None else plan.target,
        )
        pnl = float(event.realized_pnl.as_double())
        quantity = int(event.peak_qty.as_double())
        # Net R: realized PnL already includes commission and the slippage
        # charged by the fee model. Dividing by initial risk dollars keeps
        # costs in the ratio instead of reporting the gross price excursion.
        if plan is None or quantity <= 0:
            r_multiple = None
        else:
            risk_dollars = abs(entry - plan.stop) * quantity
            r_multiple = pnl / risk_dollars if risk_dollars > 0 else None
        self.trades.append(
            TradeRecord(
                symbol=symbol,
                entry_ts=_utc(event.ts_opened),
                exit_ts=_utc(event.ts_closed),
                entry_price=entry,
                exit_price=exit_price,
                quantity=quantity,
                stop_price=plan.stop if plan else 0.0,
                target_price=plan.target if plan else 0.0,
                pnl=pnl,
                r_multiple=r_multiple,
                exit_reason=reason,
            )
        )

    def _submit_bracket(
        self,
        instrument_id: InstrumentId,
        side: str,
        quantity: int,
        stop: float,
        target: float,
    ) -> None:
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return
        symbol = instrument_id.symbol.value
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        order_list = self.order_factory.bracket(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=instrument.make_qty(quantity),
            contingency_type=ContingencyType.OCO,
            entry_order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            tp_price=instrument.make_price(target),
            tp_post_only=False,
            sl_trigger_price=instrument.make_price(stop),
            entry_tags=["ENTRY"],
            tp_tags=["TARGET"],
            sl_tags=["STOP"],
        )
        self.submit_order_list(order_list)
        self._plans[symbol] = _Plan(stop=stop, target=target, side=side)

    def _flatten(self, instrument_id: InstrumentId, reason: str) -> None:
        symbol = instrument_id.symbol.value
        if reason == "daily loss cutoff":
            if self.equity_curve:
                self._halted_days.add(eastern_date(self.equity_curve[-1][0]))
        has_orders = len(self.cache.orders_open(instrument_id=instrument_id)) > 0
        in_position = not self.portfolio.is_flat(instrument_id)
        # A flatten on a flat book must not stick to the next trade. A later
        # target fill was inheriting "daily loss cutoff" from that stamp.
        if not has_orders and not in_position:
            self._exit_reasons.pop(symbol, None)
            return
        self._exit_reasons[symbol] = reason
        self.cancel_all_orders(instrument_id)
        if in_position:
            self.close_all_positions(instrument_id, tags=[reason.upper().replace(" ", "_")])

    def _forced_exit_reason(self, ts: datetime, session, snapshot: PortfolioSnapshot) -> str | None:
        assert self._risk is not None
        if self._risk.killed:
            return "kill switch"
        if session in self._halted_days or self._risk.daily_loss_breached(snapshot):
            return "daily loss cutoff"
        if self._risk.must_flatten(ts):
            return "flatten"
        return None

    def _roll_day(self, ts: datetime, equity: float) -> None:
        session = eastern_date(ts)
        if self._day != session:
            self._day = session
            self._day_start_equity = equity
            self._exit_reasons.clear()

    def _equity(self, venue: Venue) -> float:
        account = self.portfolio.account(venue)
        from nautilus_trader.model.currencies import USD

        balance = float(account.balance_total(USD).as_double())
        unrealized = 0.0
        for money in self.portfolio.unrealized_pnls(venue).values():
            unrealized += float(money.as_double())
        return balance + unrealized

    def _open_symbols(self) -> frozenset[str]:
        symbols = {position.instrument_id.symbol.value for position in self.cache.positions_open()}
        for order in self.cache.orders_open():
            tags = list(order.tags) if order.tags else []
            if "ENTRY" in tags:
                symbols.add(order.instrument_id.symbol.value)
        return frozenset(symbols)

    def _in_position(self, instrument_id: InstrumentId) -> bool:
        return not self.portfolio.is_flat(instrument_id)

    def _busy(self, instrument_id: InstrumentId) -> bool:
        if self._in_position(instrument_id):
            return True
        return len(self.cache.orders_open(instrument_id=instrument_id)) > 0


def resolve_bracket(
    *,
    side: str,
    entry: float,
    engine_stop: float,
    engine_target: float,
    hint_stop: float | None,
    hint_target: float | None,
    reward_to_risk: float,
) -> tuple[float, float]:
    """Use the risk engine's stop and target unless a hint reduces risk.

    A wider stop is ignored. A target closer than ``reward_to_risk`` is
    ignored. Hints of ``None`` return the engine prices unchanged.
    """
    if hint_stop is None and hint_target is None:
        return engine_stop, engine_target
    stop = engine_stop
    target = engine_target
    if side == "long":
        if hint_stop is not None and engine_stop < hint_stop < entry:
            stop = hint_stop
        if hint_target is not None and hint_target > target:
            risk = entry - stop
            if risk > 0 and (hint_target - entry) / risk + 1e-12 >= reward_to_risk:
                target = hint_target
        return stop, target
    if side == "short":
        if hint_stop is not None and entry < hint_stop < engine_stop:
            stop = hint_stop
        if hint_target is not None and hint_target < target:
            risk = stop - entry
            if risk > 0 and (entry - hint_target) / risk + 1e-12 >= reward_to_risk:
                target = hint_target
        return stop, target
    return engine_stop, engine_target


def run_signal_on_bars(
    bars,
    *,
    holdout,
    signal_model: type[SignalModel],
    params: StrategyParams,
    strategy_name: str,
    starting_equity: float = 100_000.0,
    atr_period: int = 14,
    bar_minutes: int = 1,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
    risk_engine: RiskEngine | None = None,
    install_risk: bool = True,
    log_level: str = "ERROR",
    venue_name: str = "XNAS",
) -> PerformanceReport:
    """Run ``signal_model`` inside the harness. Bars are already resampled."""
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
    from aegis.backtest.convert import bar_type_for, equity_instrument, frame_to_bars
    from aegis.backtest.risk import RiskLimits, clear_risk_engine, install_risk_engine
    from aegis.holdout import assert_dates_outside_holdout

    if bar_minutes < 1:
        raise ValueError("bar_minutes must be at least 1.")
    dates: set[date] = set()
    if not bars.is_empty():
        dates = set(
            bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).to_series().to_list()
        )
    assert_dates_outside_holdout(dates, holdout, purpose=f"{strategy_name} backtest")
    symbols = tuple(bars["symbol"].unique().sort().to_list()) if not bars.is_empty() else ()
    active_engine = risk_engine or RiskEngine(RiskLimits(atr_multiplier=float(getattr(params, "atr_multiplier", 2.0))))
    report_kwargs = dict(
        strategy=strategy_name,
        symbols=symbols,
        starting_equity=starting_equity,
        parameters=params.model_dump(),
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
        risk_limits=risk_limits_snapshot(active_engine.limits),
        risk_engine_installed=install_risk,
        bar_timeframe=str(getattr(params, "bar_timeframe", f"{bar_minutes}min")),
    )
    if bars.is_empty():
        return build_report(**report_kwargs, equity_curve=[], trades=[], bars=bars)
    if install_risk:
        install_risk_engine(active_engine)
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
        strategy = HarnessStrategy(
            HarnessConfig(
                instrument_ids=instrument_ids,
                bar_types=bar_types,
                strategy_name=strategy_name,
                signal_path=f"{signal_model.__module__}:{signal_model.__name__}",
                params_path=f"{type(params).__module__}:{type(params).__name__}",
                params_json=params.model_dump_json(),
                atr_period=atr_period,
            )
        )
        engine.add_strategy(strategy)
        engine.run()
        return build_report(
            **report_kwargs,
            equity_curve=list(strategy.equity_curve),
            trades=list(strategy.trades),
            bars=bars,
        )
    finally:
        if engine is not None:
            engine.dispose()
        clear_risk_engine()


def _build_indicator(spec: IndicatorSpec) -> ExponentialMovingAverage:
    if spec.kind == "ema":
        return ExponentialMovingAverage(spec.period)
    raise ValueError(f"Unsupported indicator kind {spec.kind!r}.")


def _bar_view(symbol: str, ts: datetime, bar: Bar, atr: float | None, indicators: dict) -> "BarView":
    from aegis.strategies.signal import BarView

    values = {
        name: float(indicator.value) if indicator.initialized else None for name, indicator in indicators.items()
    }
    return BarView(
        symbol=symbol,
        ts=ts,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
        atr=atr,
        indicators=values,
    )


def _load(path: str):
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Expected 'module:Class', got {path!r}.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def resolve_exit_reason(
    *,
    stamped: str | None,
    closing_tags: tuple[str, ...] | list[str],
    exit_price: float,
    stop: float | None,
    target: float | None,
) -> str:
    """Name the fill that closed the position.

    A target or stop tag wins over a forced-exit stamp. Otherwise a winner
    that actually filled its bracket was recorded as ``daily loss cutoff``
    when an earlier flatten, on a flat book, had left that stamp behind.
    """
    tags = {str(tag) for tag in closing_tags}
    if "TARGET" in tags:
        return "target"
    if "STOP" in tags:
        return "stop"
    if stamped:
        return stamped
    if stop is not None and target is not None:
        if abs(exit_price - stop) <= abs(exit_price - target):
            return "stop"
        return "target"
    return "unknown"


def _closing_tags(strategy: HarnessStrategy, event) -> tuple[str, ...]:
    order_id = getattr(event, "closing_order_id", None)
    if order_id is None:
        return ()
    try:
        order = strategy.cache.order(order_id)
    except Exception:
        return ()
    if order is None or not order.tags:
        return ()
    return tuple(str(tag) for tag in order.tags)


def _utc(nanos: int) -> datetime:
    return datetime.fromtimestamp(nanos / 1_000_000_000, tz=timezone.utc)


# Re-export the plug-in seam. Signal modules import these from
# ``aegis.strategies.signal`` so they do not load Nautilus. The names live
# here because the harness is the only caller that may turn a signal into an order.
from aegis.strategies.policy import check_signal_source as check_signal_source  # noqa: E402
from aegis.strategies.plugin import register_signal as register_signal  # noqa: E402
from aegis.strategies.signal import BarView as BarView  # noqa: E402
from aegis.strategies.signal import Signal as Signal  # noqa: E402
from aegis.strategies.signal import SignalAction as SignalAction  # noqa: E402
from aegis.strategies.signal import SignalModel as SignalModel  # noqa: E402
