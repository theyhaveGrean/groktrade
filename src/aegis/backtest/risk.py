"""Deterministic risk engine.

These limits are the spec, not suggestions. Strategies ask
:func:`get_risk_engine` and place only what :meth:`RiskEngine.evaluate_entry`
approves. Nothing in this module imports an LLM client, and the numeric caps
are not read from the environment. The one env flag is the human kill switch,
applied by the backtest runner when it installs the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

_ENGINE: RiskEngine | None = None


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade: float = 0.01
    max_positions: int = 5
    max_position_equity_fraction: float = 0.25
    daily_loss_cutoff: float = 0.05
    reward_to_risk: float = 2.5
    atr_multiplier: float = 2.0
    entry_start: time = time(9, 35)
    entry_end: time = time(15, 30)
    flatten_time: time = time(15, 55)
    kill_switch: bool = False


@dataclass(frozen=True)
class PortfolioSnapshot:
    equity: float
    day_start_equity: float
    open_symbols: frozenset[str] = field(default_factory=frozenset)

    @property
    def open_position_count(self) -> int:
        return len(self.open_symbols)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    quantity: int = 0
    stop_price: float | None = None
    target_price: float | None = None
    flatten: bool = False


@dataclass
class AtrState:
    """Wilder ATR. The first ``period`` true ranges seed a simple average."""

    period: int
    prev_close: float | None = None
    _seed: list[float] = field(default_factory=list)
    atr: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self.prev_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )
        self.prev_close = close
        if self.atr is None:
            self._seed.append(true_range)
            if len(self._seed) == self.period:
                self.atr = sum(self._seed) / self.period
            return self.atr
        self.atr = (self.atr * (self.period - 1) + true_range) / self.period
        return self.atr


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.killed = bool(self.limits.kill_switch)
        self.kill_reason = "kill switch" if self.killed else None

    def trip(self, reason: str) -> None:
        """Latch the emergency stop. A human has to call :meth:`reset`."""
        self.killed = True
        self.kill_reason = reason

    def reset(self) -> None:
        """Clear the kill switch. Agents must not be given this method."""
        self.killed = False
        self.kill_reason = None

    def must_flatten(self, ts: datetime) -> bool:
        local = _eastern(ts)
        return local.time() >= self.limits.flatten_time

    def entry_window_open(self, ts: datetime) -> tuple[bool, str]:
        local = _eastern(ts)
        if local.weekday() >= 5:
            return False, "weekend"
        if local.time() < self.limits.entry_start:
            return False, "before entry window"
        if local.time() > self.limits.entry_end:
            return False, "after entry window"
        return True, "open"

    def daily_loss_breached(self, snapshot: PortfolioSnapshot) -> bool:
        baseline = snapshot.day_start_equity
        if baseline <= 0:
            return False
        loss = (baseline - snapshot.equity) / baseline
        return loss >= self.limits.daily_loss_cutoff

    def stop_price(self, side: str, entry: float, atr: float) -> float:
        distance = atr * self.limits.atr_multiplier
        if side == "long":
            return _cents(entry - distance)
        if side == "short":
            return _cents(entry + distance)
        raise ValueError(f"Unsupported side {side!r}.")

    def target_price(self, side: str, entry: float, stop: float) -> float:
        """Round the target away from entry so the ratio stays at least 2.5:1."""
        risk = abs(entry - stop)
        offset = self.limits.reward_to_risk * risk
        if side == "long":
            return _cents_away(entry + offset, entry, direction=1)
        if side == "short":
            return _cents_away(entry - offset, entry, direction=-1)
        raise ValueError(f"Unsupported side {side!r}.")

    def position_quantity(self, equity: float, entry: float, stop: float) -> int:
        risk_per_share = abs(entry - stop)
        if equity <= 0 or entry <= 0 or risk_per_share <= 0:
            return 0
        by_risk = int((equity * self.limits.max_risk_per_trade) // risk_per_share)
        by_notional = int((equity * self.limits.max_position_equity_fraction) // entry)
        return max(0, min(by_risk, by_notional))

    def evaluate_entry(
        self,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        atr: float | None,
        snapshot: PortfolioSnapshot,
    ) -> RiskDecision:
        if self.killed:
            return _deny(self.kill_reason or "kill switch", flatten=True)
        if self.daily_loss_breached(snapshot):
            return _deny("daily loss cutoff", flatten=True)
        if self.must_flatten(ts):
            return _deny("flatten window", flatten=True)
        open_now, why = self.entry_window_open(ts)
        if not open_now:
            return _deny(why)
        if symbol in snapshot.open_symbols:
            return _deny("already in position")
        if snapshot.open_position_count >= self.limits.max_positions:
            return _deny("max simultaneous positions")
        if atr is None or atr <= 0 or entry <= 0:
            return _deny("atr or entry is not positive")
        if side not in {"long", "short"}:
            return _deny("unsupported side")
        stop = self.stop_price(side, entry, atr)
        if abs(entry - stop) < 0.01:
            return _deny("stop distance rounds to zero")
        target = self.target_price(side, entry, stop)
        reward = abs(target - entry)
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0 or reward / risk_per_share < self.limits.reward_to_risk - 1e-9:
            return _deny("reward-to-risk below 2.5")
        quantity = self.position_quantity(snapshot.equity, entry, stop)
        if quantity < 1:
            return _deny("position size below 1 share")
        if quantity * risk_per_share > snapshot.equity * self.limits.max_risk_per_trade + 1e-6:
            return _deny("risk exceeds 1% of equity")
        if quantity * entry > snapshot.equity * self.limits.max_position_equity_fraction + 1e-6:
            return _deny("notional exceeds 25% of equity")
        return RiskDecision(
            approved=True,
            reason="approved",
            quantity=quantity,
            stop_price=stop,
            target_price=target,
        )


def install_risk_engine(engine: RiskEngine) -> None:
    """Publish the engine the strategy is allowed to see. The runner calls this."""
    global _ENGINE
    _ENGINE = engine


def get_risk_engine() -> RiskEngine:
    if _ENGINE is None:
        from aegis.errors import RiskEngineNotInstalled

        raise RiskEngineNotInstalled(
            "Deterministic risk engine is not installed. Refusing to trade."
        )
    return _ENGINE


def clear_risk_engine() -> None:
    global _ENGINE
    _ENGINE = None


def _eastern(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Risk checks require a timezone-aware timestamp.")
    return ts.astimezone(NY)


def _cents(value: float) -> float:
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(str(round(value, 8))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _cents_away(value: float, entry: float, *, direction: int) -> float:
    from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

    quantum = Decimal("0.01")
    raw = Decimal(str(round(value, 8)))
    rounding = ROUND_CEILING if direction > 0 else ROUND_FLOOR
    rounded = raw.quantize(quantum, rounding=rounding)
    entry_cents = Decimal(str(_cents(entry)))
    if direction > 0 and rounded <= entry_cents:
        rounded = entry_cents + quantum
    if direction < 0 and rounded >= entry_cents:
        rounded = entry_cents - quantum
    return float(rounded)


def _deny(reason: str, *, flatten: bool = False) -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, flatten=flatten)
