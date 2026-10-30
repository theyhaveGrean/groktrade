"""Opening-range breakout plug-in.

The range is the high and low of bars whose open falls in the first
``opening_minutes`` of the regular session. A later close above that high
enters long once per session. A close below the range low exits.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import model_validator

from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.resample import timeframe_minutes
from aegis.strategies.plugin import register_signal
from aegis.strategies.signal import BarView, Signal, SignalModel


class OpeningRangeParams(StrategyParams):
    opening_minutes: Annotated[int, SearchMeta(kind="int", default_grid=(5, 15, 30), minimum=1, maximum=60)] = 15
    atr_period: Annotated[int, SearchMeta(kind="int", default_grid=(14,), minimum=2)] = 14
    atr_multiplier: Annotated[float, SearchMeta(kind="float", default_grid=(2.0,), minimum=0.1)] = 2.0
    bar_timeframe: Annotated[
        str,
        SearchMeta(kind="choice", default_grid=("5min", "15min"), choices=("1min", "5min", "15min", "30min", "60min")),
    ] = "5min"

    @model_validator(mode="after")
    def _range_matches_bars(self) -> OpeningRangeParams:
        size = timeframe_minutes(self.bar_timeframe)
        if self.opening_minutes < size or self.opening_minutes % size != 0:
            raise ValueError("opening_minutes must be a positive multiple of bar_timeframe.")
        return self


class OpeningRangeBreakout(SignalModel):
    def __init__(self, params: OpeningRangeParams) -> None:
        super().__init__(params)
        self._day: dict[str, object] = {}
        self._high: dict[str, float] = {}
        self._low: dict[str, float] = {}
        self._signaled: dict[str, bool] = {}

    def warmup_bars(self) -> int:
        size = timeframe_minutes(self.params.bar_timeframe)
        return max(1, int(self.params.opening_minutes // size))

    def on_bar(self, bar: BarView) -> Signal:
        day = bar.session_date
        if self._day.get(bar.symbol) != day:
            self._day[bar.symbol] = day
            self._high[bar.symbol] = bar.high
            self._low[bar.symbol] = bar.low
            self._signaled[bar.symbol] = False
        if bar.in_opening_range(int(self.params.opening_minutes)):
            self._high[bar.symbol] = max(self._high[bar.symbol], bar.high)
            self._low[bar.symbol] = min(self._low[bar.symbol], bar.low)
            return Signal.none()
        if bar.minute_of_day < 9 * 60 + 30:
            return Signal.none()
        # ATR warmup can reject the first breakout bar. Keep the signal armed
        # until a bar the risk engine is able to size.
        if bar.close > self._high[bar.symbol] and not self._signaled[bar.symbol]:
            if bar.atr is None:
                return Signal.none()
            self._signaled[bar.symbol] = True
            return Signal.enter_long()
        if self._signaled[bar.symbol] and bar.close < self._low[bar.symbol]:
            return Signal.exit()
        return Signal.none()


register_signal(
    name="opening_range_breakout",
    summary="Long-only opening-range breakout. One attempt per session.",
    params_model=OpeningRangeParams,
    model=OpeningRangeBreakout,
)
