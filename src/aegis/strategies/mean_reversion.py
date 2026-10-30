"""VWAP and Bollinger z-score mean reversion.

Enter long when the close is at least ``z_entry`` standard deviations below
the trailing mean and also below the session VWAP. Exit when the z-score
recovers to ``z_exit``. The window is causal: it ends at the current close.
"""

from __future__ import annotations

from collections import deque
from typing import Annotated

from pydantic import model_validator

from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.plugin import register_signal
from aegis.strategies.signal import BarView, Signal, SignalModel


class MeanReversionParams(StrategyParams):
    period: Annotated[int, SearchMeta(kind="int", default_grid=(20,), minimum=5, maximum=80)] = 20
    z_entry: Annotated[float, SearchMeta(kind="float", default_grid=(1.5, 2.0), minimum=0.5, maximum=4.0)] = 2.0
    z_exit: Annotated[float, SearchMeta(kind="float", default_grid=(0.0,), minimum=-1.0, maximum=1.0)] = 0.0
    atr_period: Annotated[int, SearchMeta(kind="int", default_grid=(14,), minimum=2)] = 14
    atr_multiplier: Annotated[float, SearchMeta(kind="float", default_grid=(2.0,), minimum=0.1)] = 2.0
    bar_timeframe: Annotated[
        str,
        SearchMeta(kind="choice", default_grid=("5min", "15min"), choices=("1min", "5min", "15min", "30min", "60min")),
    ] = "5min"

    @model_validator(mode="after")
    def _band(self) -> MeanReversionParams:
        if self.z_exit <= -self.z_entry:
            raise ValueError("z_exit must sit above the short entry band.")
        return self


class MeanReversion(SignalModel):
    def __init__(self, params: MeanReversionParams) -> None:
        super().__init__(params)
        self._closes: dict[str, deque[float]] = {}
        self._day: dict[str, object] = {}
        self._tpv: dict[str, float] = {}
        self._volume: dict[str, float] = {}

    def warmup_bars(self) -> int:
        return int(self.params.period)

    def on_bar(self, bar: BarView) -> Signal:
        symbol = bar.symbol
        if self._day.get(symbol) != bar.session_date:
            self._day[symbol] = bar.session_date
            self._tpv[symbol] = 0.0
            self._volume[symbol] = 0.0
        typical = (bar.high + bar.low + bar.close) / 3.0
        self._tpv[symbol] += typical * bar.volume
        self._volume[symbol] += bar.volume
        vwap = self._tpv[symbol] / self._volume[symbol] if self._volume[symbol] > 0 else bar.close
        window = self._closes.setdefault(symbol, deque(maxlen=int(self.params.period)))
        window.append(bar.close)
        period = int(self.params.period)
        if len(window) < period:
            return Signal.none()
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / period
        if variance <= 0:
            return Signal.none()
        z_score = (bar.close - mean) / variance**0.5
        if z_score <= -abs(self.params.z_entry) and bar.close < vwap:
            return Signal.enter_long()
        if z_score >= self.params.z_exit:
            return Signal.exit()
        return Signal.none()


register_signal(
    name="mean_reversion",
    summary="Long-only VWAP and Bollinger z-score mean reversion.",
    params_model=MeanReversionParams,
    model=MeanReversion,
)
