"""Live EMA crossover. The cross rule is the backtest function, not a second copy."""

from __future__ import annotations

from dataclasses import dataclass

from nautilus_trader.indicators import ExponentialMovingAverage

from aegis.backtest.risk import AtrState
from aegis.backtest.strategy import detect_ema_cross
from aegis.execution.models import MinuteBar


@dataclass(frozen=True)
class BarIntent:
    symbol: str
    close: float
    atr: float | None
    crossed_up: bool
    crossed_down: bool
    ready: bool


class EmaCrossoverSignal:
    """Long-only 20/50 signal by default, matching :class:`EmaCrossConfig`."""

    name = "ema_crossover"

    def __init__(self, symbols: list[str], fast_period: int = 20, slow_period: int = 50, atr_period: int = 14) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be lower than slow_period.")
        if atr_period < 1:
            raise ValueError("atr_period must be at least 1.")
        self.symbols = frozenset(symbols)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self._fast = {symbol: ExponentialMovingAverage(fast_period) for symbol in symbols}
        self._slow = {symbol: ExponentialMovingAverage(slow_period) for symbol in symbols}
        self._atr = {symbol: AtrState(atr_period) for symbol in symbols}
        self._prev: dict[str, tuple[float, float] | None] = {symbol: None for symbol in symbols}

    def update(self, bar: MinuteBar) -> BarIntent:
        symbol = bar.symbol
        if symbol not in self.symbols:
            raise KeyError(symbol)
        fast = self._fast[symbol]
        slow = self._slow[symbol]
        fast.update_raw(bar.close)
        slow.update_raw(bar.close)
        atr = self._atr[symbol].update(bar.high, bar.low, bar.close)
        ready = bool(fast.initialized and slow.initialized)
        crossed_up, crossed_down = detect_ema_cross(
            self._prev[symbol],
            float(fast.value),
            float(slow.value),
            ready=ready,
        )
        if ready:
            self._prev[symbol] = (float(fast.value), float(slow.value))
        return BarIntent(
            symbol=symbol,
            close=bar.close,
            atr=atr,
            crossed_up=crossed_up,
            crossed_down=crossed_down,
            ready=ready,
        )
