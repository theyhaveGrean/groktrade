"""Paper session: live bars in, risk-gated bracket orders out.

The 15:55 America/New_York flatten runs from the bar clock and from a wall-clock
poll, so a quiet feed still flattens. Daily loss uses broker equity.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Protocol
from uuid import uuid4

from aegis.backtest.risk import PortfolioSnapshot, RiskDecision, RiskEngine, RiskLimits, install_risk_engine
from aegis.config import Settings
from aegis.data.sessions import eastern_date, to_eastern
from aegis.errors import AegisError
from aegis.execution.audit import AuditLog
from aegis.execution.guards import assert_paper_execution_allowed
from aegis.execution.killswitch import arm_kill_file, kill_reason
from aegis.execution.models import (
    AccountView,
    AssetView,
    BracketOrder,
    MinuteBar,
    OrderView,
    PositionView,
    ReconcileReport,
    iter_orders,
    is_open_status,
)
from aegis.execution.reconcile import sync_and_reconcile
from aegis.execution.signal import EmaCrossoverSignal
from aegis.execution.state import PaperState
from aegis.execution.validate import PreTradeValidator

log = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 7
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)


class Broker(Protocol):
    def account(self) -> AccountView: ...
    def asset(self, symbol: str) -> AssetView: ...
    def positions(self) -> list[PositionView]: ...
    def open_orders(self) -> list[OrderView]: ...
    def submit_bracket(self, order: BracketOrder) -> OrderView: ...
    def cancel_all_orders(self) -> None: ...
    def flatten_all(self) -> None: ...
    def flatten_symbol(self, symbol: str) -> None: ...


class BarSource(Protocol):
    def historical_bars(self, symbols: list[str], start: datetime, end: datetime) -> list[MinuteBar]: ...
    def stream(self, symbols: list[str], on_bar: Callable[[MinuteBar], None]) -> None: ...


@dataclass
class StatusReport:
    base_url: str
    kill_reason: str | None
    equity: float | None
    last_equity: float | None
    buying_power: float | None
    day_start_equity: float
    session_date: str | None
    halted_session: str | None
    mismatch: bool
    problems: tuple[str, ...]
    positions: tuple[tuple[str, int], ...]
    open_orders: tuple[str, ...]

    def render(self) -> str:
        def money(value: float | None) -> str:
            return "unavailable" if value is None else f"{value:.2f}"

        lines = [
            "trading_mode: paper",
            f"alpaca_base_url: {self.base_url}",
            f"kill_switch: {self.kill_reason or 'off'}",
            f"equity: {money(self.equity)}",
            f"last_equity: {money(self.last_equity)}",
            f"buying_power: {money(self.buying_power)}",
            f"day_start_equity: {self.day_start_equity:.2f}",
            f"session_date: {self.session_date or '-'}",
            f"halted_session: {self.halted_session or '-'}",
            f"reconciliation: {'MISMATCH' if self.mismatch else 'ok'}",
        ]
        if self.equity is not None and self.day_start_equity > 0:
            change = (self.equity - self.day_start_equity) / self.day_start_equity
            lines.append(f"day_change: {change:.2%}")
        if self.positions:
            lines.append("positions:")
            lines.extend(f"  {symbol} {qty}" for symbol, qty in self.positions)
        else:
            lines.append("positions: none")
        if self.open_orders:
            lines.append("open_orders:")
            lines.extend(f"  {order_id}" for order_id in self.open_orders)
        else:
            lines.append("open_orders: none")
        for problem in self.problems:
            lines.append(f"mismatch: {problem}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_mode": "paper",
            "alpaca_base_url": self.base_url,
            "kill_switch": self.kill_reason or "off",
            "equity": self.equity,
            "last_equity": self.last_equity,
            "buying_power": self.buying_power,
            "day_start_equity": self.day_start_equity,
            "session_date": self.session_date,
            "halted_session": self.halted_session,
            "mismatch": self.mismatch,
            "problems": list(self.problems),
            "positions": [{"symbol": symbol, "qty": qty} for symbol, qty in self.positions],
            "open_orders": list(self.open_orders),
        }


class PaperSession:
    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        symbols: list[str],
        *,
        strategy: str = "ema_crossover",
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        assert_paper_execution_allowed(settings)
        if strategy != "ema_crossover":
            raise AegisError(
                f"Unknown strategy {strategy!r}. This paper build runs ema_crossover."
            )
        if not symbols:
            raise AegisError("Paper trading needs at least one symbol.")
        from aegis.data.universe import validate_symbol

        self.settings = settings
        self.broker = broker
        self.symbols = [validate_symbol(symbol.strip().upper()) for symbol in symbols]
        self.strategy_name = strategy
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.risk = RiskEngine(RiskLimits(kill_switch=settings.risk_kill_switch))
        install_risk_engine(self.risk)
        self.audit = AuditLog(settings.paper_dir / "audit.jsonl")
        self.state = PaperState.load(settings.paper_dir / "state.json")
        self.signal = EmaCrossoverSignal(self.symbols, fast_period, slow_period, atr_period)
        self.validator = PreTradeValidator(self.risk)
        self.reconcile_seconds = settings.paper_reconcile_seconds
        self.poll_seconds = settings.paper_poll_seconds
        self._lock = threading.RLock()
        self._last_reconcile: datetime | None = None
        self._last_flatten_reason: str | None = None
        self._flattening = False
        self._was_mismatch = self.state.mismatch
        self._account: AccountView | None = None
        self._positions: list[PositionView] = []
        self._orders: list[OrderView] = []
        self._arm_if_requested()

    def serve(self, feed: BarSource) -> None:
        """Warm the EMA on recent IEX bars, then trade the live stream."""
        end = self._clock()
        start = end - timedelta(days=WARMUP_CALENDAR_DAYS)
        history = feed.historical_bars(self.symbols, start, end)
        warmed = 0
        with self._lock:
            for bar in sorted(history, key=lambda item: (item.ts, item.symbol)):
                if bar.symbol not in self.signal.symbols or not _regular_session(bar.ts):
                    continue
                self.signal.update(bar)
                warmed += 1
        log.info("warmed EMA on %s IEX bars", warmed)
        self.poll()
        stop = threading.Event()

        def ticker() -> None:
            while not stop.wait(self.poll_seconds):
                try:
                    self.poll()
                except Exception:
                    log.exception("paper poll failed")

        thread = threading.Thread(target=ticker, name="aegis-paper-poll", daemon=True)
        thread.start()
        try:
            feed.stream(self.symbols, self.on_bar)
        finally:
            stop.set()
            thread.join(timeout=1)

    def on_bar(self, bar: MinuteBar) -> None:
        with self._lock:
            self._on_bar(bar)

    def poll(self, now: datetime | None = None) -> None:
        with self._lock:
            self._poll(now or self._clock())

    def consider_entry(
        self,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        atr: float | None,
    ) -> RiskDecision:
        with self._lock:
            return self._consider_entry(symbol=symbol, ts=ts, side=side, entry=entry, atr=atr)

    def reconcile_now(self, now: datetime | None = None) -> ReconcileReport:
        with self._lock:
            return self._reconcile_now(now or self._clock())

    def flatten_all(self, now: datetime | None = None, reason: str = "flatten") -> None:
        with self._lock:
            self._flatten_all(now or self._clock(), reason, force=True)

    def _on_bar(self, bar: MinuteBar) -> None:
        if bar.symbol not in self.signal.symbols:
            return
        if not _regular_session(bar.ts):
            self._poll(self._clock())
            return
        intent = self.signal.update(bar)
        self._poll(bar.ts)
        wall = self._clock()
        if self.risk.must_flatten(wall) and not self.risk.must_flatten(bar.ts):
            self._flatten_all(wall, "flatten")
        flattening = self.risk.killed or self.risk.must_flatten(bar.ts) or self.risk.must_flatten(wall)
        if intent.crossed_down and not flattening:
            self._flatten_symbol(bar.symbol, "signal", bar.ts)
            return
        if intent.crossed_up:
            self._consider_entry(
                symbol=bar.symbol,
                ts=bar.ts,
                side="long",
                entry=bar.close,
                atr=intent.atr,
            )

    def _poll(self, now: datetime) -> None:
        self._arm_if_requested()
        fresh = self._ensure_fresh()
        if self.risk.killed:
            self._flatten_all(now, self.risk.kill_reason or "kill switch")
            return
        if self.risk.must_flatten(now) or self.risk.must_flatten(self._clock()):
            self._flatten_all(now, "flatten")
            return
        if not fresh or self._account is None:
            return
        self._baseline(now)
        session = eastern_date(now).isoformat()
        snapshot = self._snapshot(now)
        loss = self.state.halted_session == session or self.risk.daily_loss_breached(snapshot)
        if loss:
            self.state.halted_session = session
            self.state.save()
            self._flatten_all(now, "daily loss cutoff")
            return
        self._maybe_reconcile(now)

    def _consider_entry(
        self,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        atr: float | None,
    ) -> RiskDecision:
        self._arm_if_requested()
        if not self._ensure_fresh() or self._account is None:
            decision = RiskDecision(approved=False, reason="broker unavailable")
            self._audit_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry)
            return decision
        self._baseline(ts)
        session = eastern_date(ts).isoformat()
        if self.state.halted_session == session:
            decision = RiskDecision(approved=False, reason="daily loss cutoff", flatten=True)
            self._finish_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry, asset=None)
            return decision
        if self.risk.must_flatten(self._clock()) and not self.risk.must_flatten(ts):
            decision = RiskDecision(approved=False, reason="flatten", flatten=True)
            self._finish_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry, asset=None)
            return decision
        snapshot = self._snapshot(ts)
        # Risk, buying power, and the trading block are decided before the asset
        # lookup. A synthetic tradable asset keeps that first pass from hiding a
        # risk refusal behind "symbol not tradable". The real asset is required
        # before an order can be sent.
        preliminary = self.validator.evaluate(
            symbol=symbol,
            ts=ts,
            side=side,
            entry=entry,
            atr=atr,
            snapshot=snapshot,
            buying_power=self._account.buying_power,
            trading_blocked=self._account.trading_blocked,
            asset=AssetView(symbol=symbol, tradable=True, status="active"),
            mismatch=False,
        )
        if not preliminary.approved:
            self._finish_decision(preliminary, symbol=symbol, ts=ts, side=side, entry=entry, asset=None)
            return preliminary
        try:
            asset = self.broker.asset(symbol)
        except Exception as exc:
            decision = RiskDecision(approved=False, reason="symbol not tradable")
            self._audit_decision(
                decision,
                symbol=symbol,
                ts=ts,
                side=side,
                entry=entry,
                error=self._public_error(exc),
            )
            return decision
        decision = self.validator.evaluate(
            symbol=symbol,
            ts=ts,
            side=side,
            entry=entry,
            atr=atr,
            snapshot=snapshot,
            buying_power=self._account.buying_power,
            trading_blocked=self._account.trading_blocked,
            asset=asset,
            mismatch=self.state.mismatch,
        )
        if not decision.approved or decision.stop_price is None or decision.target_price is None:
            self._finish_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry, asset=asset)
            return decision
        bracket = BracketOrder(
            symbol=symbol,
            side=side,
            quantity=decision.quantity,
            stop_price=decision.stop_price,
            target_price=decision.target_price,
            client_order_id=f"aegis-{uuid4().hex}",
            entry_price=entry,
        )
        try:
            submitted = self.broker.submit_bracket(bracket)
        except Exception as exc:
            rejected = RiskDecision(approved=False, reason="broker rejected order")
            self._audit_decision(
                rejected,
                symbol=symbol,
                ts=ts,
                side=side,
                entry=entry,
                error=self._public_error(exc),
            )
            return rejected
        self._record_order(submitted, bracket, role="entry")
        self.state.save()
        self._audit_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry, asset=asset, order_id=submitted.id)
        self._reconcile_now(ts)
        return decision

    def _finish_decision(
        self,
        decision: RiskDecision,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        asset: AssetView | None,
    ) -> None:
        if decision.reason == "daily loss cutoff":
            self.state.halted_session = eastern_date(ts).isoformat()
            self.state.save()
        self._audit_decision(decision, symbol=symbol, ts=ts, side=side, entry=entry, asset=asset)
        if decision.flatten:
            self._flatten_all(ts, decision.reason)

    def _flatten_all(self, now: datetime, reason: str, *, force: bool = False) -> None:
        if self._flattening:
            return
        idle = (
            not self._positions
            and not self._orders
            and not self.state.positions
            and not self.state.orders
        )
        if idle and self._last_flatten_reason is not None and not force:
            return
        self._flattening = True
        try:
            self.broker.cancel_all_orders()
            self.broker.flatten_all()
            self.state.clear_exposure()
            self.state.save()
            self.audit.record(event="flatten", reason=reason, bar_ts=_iso(now), strategy=self.strategy_name)
            self._last_flatten_reason = reason
            try:
                self._reconcile_now(now)
            except Exception:
                log.exception("reconcile after flatten failed")
        finally:
            self._flattening = False

    def _flatten_symbol(self, symbol: str, reason: str, ts: datetime) -> None:
        self.broker.flatten_symbol(symbol)
        self.state.positions.pop(symbol, None)
        self.state.orders = {
            order_id: row for order_id, row in self.state.orders.items() if row.get("symbol") != symbol
        }
        self.state.save()
        self.audit.record(
            event="flatten_symbol",
            symbol=symbol,
            reason=reason,
            bar_ts=_iso(ts),
            strategy=self.strategy_name,
        )
        try:
            self._reconcile_now(ts)
        except Exception:
            log.exception("reconcile after symbol flatten failed")

    def _maybe_reconcile(self, now: datetime) -> None:
        if self._last_reconcile is not None:
            elapsed = (now - self._last_reconcile).total_seconds()
            if elapsed < self.reconcile_seconds:
                return
        self._reconcile_now(now)

    def _reconcile_now(self, now: datetime) -> ReconcileReport:
        positions = list(self.broker.positions())
        orders = list(self.broker.open_orders())
        self._positions = positions
        self._orders = orders
        report = sync_and_reconcile(self.state, positions, orders)
        self.state.mismatch = report.mismatch
        self.state.save()
        if report.mismatch:
            log.error("broker reconciliation mismatch: %s", report.summary())
            self.audit.record(
                event="reconcile_mismatch",
                problems=list(report.problems),
                bar_ts=_iso(now),
                strategy=self.strategy_name,
            )
        elif self._was_mismatch:
            self.audit.record(event="reconcile_ok", bar_ts=_iso(now), strategy=self.strategy_name)
        self._was_mismatch = report.mismatch
        self._last_reconcile = now
        return report

    def _snapshot(self, ts: datetime) -> PortfolioSnapshot:
        assert self._account is not None
        symbols = {position.symbol for position in self._positions if position.qty != 0}
        for order in iter_orders(self._orders):
            if order.symbol and is_open_status(order.status):
                symbols.add(order.symbol)
        return PortfolioSnapshot(
            equity=self._account.equity,
            day_start_equity=self._baseline(ts),
            open_symbols=frozenset(symbols),
        )

    def _baseline(self, ts: datetime) -> float:
        assert self._account is not None
        session = eastern_date(ts).isoformat()
        if self.state.session_date != session or self.state.day_start_equity <= 0:
            seeded = self._account.last_equity if self._account.last_equity > 0 else self._account.equity
            self.state.session_date = session
            self.state.day_start_equity = seeded
            if self.state.halted_session and self.state.halted_session != session:
                self.state.halted_session = None
            self.state.save()
        return self.state.day_start_equity

    def _ensure_fresh(self) -> bool:
        try:
            self._account = self.broker.account()
            self._positions = list(self.broker.positions())
            self._orders = list(self.broker.open_orders())
            return True
        except Exception as exc:
            log.exception("broker refresh failed")
            self.audit.record(event="broker_error", error=self._public_error(exc), strategy=self.strategy_name)
            return False

    def _arm_if_requested(self) -> None:
        reason = kill_reason(self.settings)
        if reason and not self.risk.killed:
            self.risk.trip(reason)

    def _record_order(self, view: OrderView, bracket: BracketOrder, *, role: str) -> None:
        self.state.orders[view.id] = {
            "id": view.id,
            "symbol": view.symbol or bracket.symbol,
            "qty": view.qty or bracket.quantity,
            "side": view.side,
            "status": view.status,
            "role": role,
            "client_order_id": view.client_order_id,
            "entry": bracket.entry_price,
            "stop": bracket.stop_price,
            "target": bracket.target_price,
        }
        for leg in view.legs:
            leg_role = "stop" if leg.stop_price is not None and leg.limit_price is None else "target"
            if leg.role in {"stop", "target"}:
                leg_role = leg.role
            self._record_order(leg, bracket, role=leg_role)

    def _audit_decision(
        self,
        decision: RiskDecision,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        asset: AssetView | None = None,
        order_id: str | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "order_decision",
            "strategy": self.strategy_name,
            "symbol": symbol,
            "side": side,
            "approved": decision.approved,
            "reason": decision.reason,
            "quantity": decision.quantity,
            "entry": entry,
            "stop": decision.stop_price,
            "target": decision.target_price,
            "bar_ts": _iso(ts),
            "equity": None if self._account is None else self._account.equity,
            "buying_power": None if self._account is None else self._account.buying_power,
            "tradable": None if asset is None else asset.tradable,
        }
        if order_id:
            payload["order_id"] = order_id
        if error:
            payload["error"] = error
        self.audit.record(**payload)

    def _public_error(self, exc: Exception) -> str:
        text = str(exc)
        for secret in (self.settings.alpaca_api_key, self.settings.alpaca_secret_key):
            if secret is None:
                continue
            raw = secret.get_secret_value()
            if raw:
                text = text.replace(raw, "[redacted]")
        return text[:500]


def execute_kill(settings: Settings, broker: Broker, *, reason: str = "aegis paper kill") -> None:
    """Arm the kill file, cancel every order, and flatten. Safe to run twice."""
    assert_paper_execution_allowed(settings)
    path = arm_kill_file(settings, reason)
    audit = AuditLog(settings.paper_dir / "audit.jsonl")
    audit.record(event="kill", reason=reason, kill_file=str(path), strategy="ema_crossover")
    broker.cancel_all_orders()
    broker.flatten_all()
    state = PaperState.load(settings.paper_dir / "state.json")
    state.clear_exposure()
    left = [position for position in broker.positions() if position.qty != 0]
    lingering = [order.id for order in iter_orders(broker.open_orders()) if order.id]
    if left or lingering:
        state.mismatch = True
        state.save()
        problems = [f"still open after kill: positions={left} orders={lingering}"]
        audit.record(event="reconcile_mismatch", problems=problems, reason=reason)
        raise AegisError("Kill switch flattened incompletely; see the audit log.")
    state.mismatch = False
    state.save()
    audit.record(event="flatten", reason=reason, strategy="ema_crossover")


def build_status(settings: Settings, broker: Broker | None = None) -> StatusReport:
    assert_paper_execution_allowed(settings)
    state = PaperState.load(settings.paper_dir / "state.json")
    armed = kill_reason(settings)
    equity = last_equity = buying_power = None
    positions: list[PositionView] = []
    orders: list[OrderView] = []
    problems: tuple[str, ...] = ()
    mismatch = state.mismatch
    if broker is not None:
        account = broker.account()
        equity = account.equity
        last_equity = account.last_equity
        buying_power = account.buying_power
        positions = list(broker.positions())
        orders = list(broker.open_orders())
        report = sync_and_reconcile(state, positions, orders)
        mismatch = report.mismatch
        problems = report.problems
        state.mismatch = mismatch
        state.save()
    open_ids = tuple(sorted(order.id for order in iter_orders(orders) if order.id and is_open_status(order.status)))
    return StatusReport(
        base_url=settings.alpaca_base_url.strip().rstrip("/"),
        kill_reason=armed,
        equity=equity,
        last_equity=last_equity,
        buying_power=buying_power,
        day_start_equity=state.day_start_equity,
        session_date=state.session_date,
        halted_session=state.halted_session,
        mismatch=mismatch,
        problems=problems,
        positions=tuple((position.symbol, position.qty) for position in positions if position.qty),
        open_orders=open_ids,
    )


def _regular_session(ts: datetime) -> bool:
    local = to_eastern(ts)
    if local.weekday() >= 5:
        return False
    clock = local.time()
    return _SESSION_OPEN <= clock < _SESSION_CLOSE


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("Paper timestamps must be timezone-aware.")
    return ts.astimezone(timezone.utc).isoformat()
