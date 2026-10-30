"""MACD momentum plug-in.

Fast and slow EMAs are harness indicators. The signal line is a causal EMA
of their difference, updated only after both price EMAs exist.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import model_validator

from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.plugin import register_signal
from aegis.strategies.signal import BarView, EmaTracker, IndicatorSpec, Signal, SignalModel


class MacdParams(StrategyParams):
    """MACD cross. Periods are bars of ``bar_timeframe``."""

    fast_period: Annotated[int, SearchMeta(kind="int", default_grid=(8, 12), minimum=2, maximum=50)] = 12
    slow_period: Annotated[int, SearchMeta(kind="int", default_grid=(26, 40), minimum=3, maximum=120)] = 26
    signal_period: Annotated[int, SearchMeta(kind="int", default_grid=(9,), minimum=2, maximum=40)] = 9
    atr_period: Annotated[int, SearchMeta(kind="int", default_grid=(14,), minimum=2)] = 14
    atr_multiplier: Annotated[float, SearchMeta(kind="float", default_grid=(2.0,), minimum=0.1)] = 2.0
    bar_timeframe: Annotated[
        str,
        SearchMeta(kind="choice", default_grid=("5min", "15min"), choices=("1min", "5min", "15min", "30min", "60min")),
    ] = "5min"

    @model_validator(mode="after")
    def _periods(self) -> MacdParams:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be lower than slow_period.")
        return self


class MacdMomentum(SignalModel):
    def __init__(self, params: MacdParams) -> None:
        super().__init__(params)
        self._signal: dict[str, EmaTracker] = {}
        self._prev: dict[str, tuple[float, float]] = {}

    def indicator_specs(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec(name="fast", kind="ema", period=self.params.fast_period),
            IndicatorSpec(name="slow", kind="ema", period=self.params.slow_period),
        ]

    def warmup_bars(self) -> int:
        return int(self.params.slow_period + self.params.signal_period)

    def on_bar(self, bar: BarView) -> Signal:
        fast = bar.indicators.get("fast")
        slow = bar.indicators.get("slow")
        if fast is None or slow is None:
            return Signal.none()
        macd = fast - slow
        tracker = self._signal.get(bar.symbol)
        if tracker is None:
            tracker = EmaTracker(int(self.params.signal_period))
            self._signal[bar.symbol] = tracker
        signal_value = tracker.update(macd)
        action = Signal.none()
        if tracker.initialized:
            previous = self._prev.get(bar.symbol)
            if previous is not None:
                if previous[0] <= previous[1] and macd > signal_value:
                    action = Signal.enter_long()
                elif previous[0] >= previous[1] and macd < signal_value:
                    action = Signal.exit()
            self._prev[bar.symbol] = (macd, signal_value)
        return action


register_signal(
    name="macd",
    summary="Long-only MACD momentum. Bullish cross enters; bearish cross exits.",
    params_model=MacdParams,
    model=MacdMomentum,
)
