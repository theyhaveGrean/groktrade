"""Volatility breakout.

Enter long when the close exceeds the highest high of the prior ``lookback``
bars. Exit when the close falls through the lowest low of that same prior
window. The current bar is not part of the window.
"""

from __future__ import annotations

from collections import deque
from typing import Annotated

from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.plugin import register_signal
from aegis.strategies.signal import BarView, Signal, SignalModel


class VolatilityBreakoutParams(StrategyParams):
    lookback: Annotated[int, SearchMeta(kind="int", default_grid=(10, 20), minimum=2, maximum=80)] = 20
    atr_period: Annotated[int, SearchMeta(kind="int", default_grid=(14,), minimum=2)] = 14
    atr_multiplier: Annotated[float, SearchMeta(kind="float", default_grid=(2.0,), minimum=0.1)] = 2.0
    bar_timeframe: Annotated[
        str,
        SearchMeta(kind="choice", default_grid=("5min", "15min"), choices=("1min", "5min", "15min", "30min", "60min")),
    ] = "5min"


class VolatilityBreakout(SignalModel):
    def __init__(self, params: VolatilityBreakoutParams) -> None:
        super().__init__(params)
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}

    def warmup_bars(self) -> int:
        return int(self.params.lookback)

    def on_bar(self, bar: BarView) -> Signal:
        lookback = int(self.params.lookback)
        highs = self._highs.setdefault(bar.symbol, deque(maxlen=lookback))
        lows = self._lows.setdefault(bar.symbol, deque(maxlen=lookback))
        action = Signal.none()
        if len(highs) == lookback:
            if bar.close > max(highs):
                action = Signal.enter_long()
            elif bar.close < min(lows):
                action = Signal.exit()
        highs.append(bar.high)
        lows.append(bar.low)
        return action


register_signal(
    name="volatility_breakout",
    summary="Long-only close through the prior N-bar high, exit through the prior low.",
    params_model=VolatilityBreakoutParams,
    model=VolatilityBreakout,
)
