"""Long-only EMA crossover that cannot size or place a trade on its own.

On a bullish cross the strategy asks the installed :class:`RiskEngine` for a
quantity, stop, and target. The engine's decision is what gets sent. A bearish
cross, the 15:55 flatten, the daily loss cutoff, and the kill switch all exit.
Stops and targets are an OCO bracket so a stop fill cancels the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import ContingencyType, OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from aegis.backtest.report import TradeRecord
from aegis.backtest.risk import AtrState, PortfolioSnapshot, RiskEngine, get_risk_engine
from aegis.data.sessions import eastern_date
from aegis.errors import RiskEngineNotInstalled


class EmaCrossConfig(StrategyConfig, frozen=True):
    instrument_ids: list[str]
    bar_types: list[str]
    fast_period: int = 20
    slow_period: int = 50
    atr_period: int = 14


@dataclass
class _Plan:
    stop: float
    target: float


class EmaCrossStrategy(Strategy):
    """Baseline signal. Risk limits live on the engine, not on this config."""

    def __init__(self, config: EmaCrossConfig) -> None:
        super().__init__(config)
        self._fast: dict[str, ExponentialMovingAverage] = {}
        self._slow: dict[str, ExponentialMovingAverage] = {}
        self._atr: dict[str, AtrState] = {}
        self._prev: dict[str, tuple[float, float] | None] = {}
        self._plans: dict[str, _Plan] = {}
        self._exit_reasons: dict[str, str] = {}
        self._halted_days: set = set()
        self._day = None
        self._day_start_equity = 0.0
        self._risk: RiskEngine | None = None
        self._disabled = False
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[TradeRecord] = []

    def on_start(self) -> None:
        if self.config.fast_period >= self.config.slow_period:
            self.log.error("fast_period must be lower than slow_period; refusing to trade")
            self._disabled = True
            return
        try:
            self._risk = get_risk_engine()
        except RiskEngineNotInstalled:
            self.log.error("deterministic risk engine is not installed; refusing to trade")
            self._disabled = True
            return
        for instrument_id, bar_type in zip(self.config.instrument_ids, self.config.bar_types, strict=True):
            symbol = InstrumentId.from_str(instrument_id).symbol.value
            fast = ExponentialMovingAverage(self.config.fast_period)
            slow = ExponentialMovingAverage(self.config.slow_period)
            parsed = BarType.from_str(bar_type)
            self.register_indicator_for_bars(parsed, fast)
            self.register_indicator_for_bars(parsed, slow)
            self.subscribe_bars(parsed)
            self._fast[symbol] = fast
            self._slow[symbol] = slow
            self._atr[symbol] = AtrState(self.config.atr_period)
            self._prev[symbol] = None

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
        fast = self._fast[symbol]
        slow = self._slow[symbol]
        equity = self._equity(instrument_id.venue)
        self.equity_curve.append((ts, equity))
        self._roll_day(ts, equity)
        snapshot = PortfolioSnapshot(
            equity=equity,
            day_start_equity=self._day_start_equity,
            open_symbols=self._open_symbols(),
        )
        session = eastern_date(ts)
        crossed_up, crossed_down = self._cross(symbol, fast, slow)
        exit_reason = self._forced_exit_reason(ts, session, snapshot)
        if exit_reason is not None:
            self._flatten(instrument_id, exit_reason)
            return
        if self._in_position(instrument_id):
            if crossed_down:
                self._flatten(instrument_id, "signal")
            return
        if not crossed_up or atr is None or not fast.initialized or not slow.initialized:
            return
        if self._busy(instrument_id):
            return
        decision = self._risk.evaluate_entry(
            symbol=symbol,
            ts=ts,
            side="long",
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
        self._submit_bracket(instrument_id, decision.quantity, decision.stop_price, decision.target_price)

    def on_position_closed(self, event) -> None:
        symbol = event.instrument_id.symbol.value
        plan = self._plans.pop(symbol, None)
        reason = self._exit_reasons.pop(symbol, None)
        entry = float(event.avg_px_open)
        exit_price = float(event.avg_px_close)
        if reason is None and plan is not None:
            if abs(exit_price - plan.stop) <= abs(exit_price - plan.target):
                reason = "stop"
            else:
                reason = "target"
        reason = reason or "unknown"
        risk = None if plan is None else entry - plan.stop
        r_multiple = (exit_price - entry) / risk if risk and risk > 0 else None
        pnl = float(event.realized_pnl.as_double())
        quantity = int(event.peak_qty.as_double())
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

    def _submit_bracket(self, instrument_id: InstrumentId, quantity: int, stop: float, target: float) -> None:
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return
        symbol = instrument_id.symbol.value
        order_list = self.order_factory.bracket(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
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
        self._plans[symbol] = _Plan(stop=stop, target=target)

    def _flatten(self, instrument_id: InstrumentId, reason: str) -> None:
        symbol = instrument_id.symbol.value
        self._exit_reasons[symbol] = reason
        if reason == "daily loss cutoff":
            if self.equity_curve:
                self._halted_days.add(eastern_date(self.equity_curve[-1][0]))
        has_orders = len(self.cache.orders_open(instrument_id=instrument_id)) > 0
        in_position = not self.portfolio.is_flat(instrument_id)
        if not has_orders and not in_position:
            return
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

    def _cross(
        self,
        symbol: str,
        fast: ExponentialMovingAverage,
        slow: ExponentialMovingAverage,
    ) -> tuple[bool, bool]:
        crossed_up = False
        crossed_down = False
        if fast.initialized and slow.initialized:
            previous = self._prev[symbol]
            if previous is not None:
                crossed_up = previous[0] <= previous[1] and fast.value > slow.value
                crossed_down = previous[0] >= previous[1] and fast.value < slow.value
            self._prev[symbol] = (fast.value, slow.value)
        return crossed_up, crossed_down

    def _roll_day(self, ts: datetime, equity: float) -> None:
        session = eastern_date(ts)
        if self._day != session:
            self._day = session
            self._day_start_equity = equity

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


def _utc(nanos: int) -> datetime:
    return datetime.fromtimestamp(nanos / 1_000_000_000, tz=timezone.utc)
