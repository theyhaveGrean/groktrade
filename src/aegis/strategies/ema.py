"""EMA crossover plug-in.

The signal reads two Nautilus EMAs that the harness registers. It does not
size or place orders. ``report_name`` stays ``ema_crossover`` so existing
backtest reports keep that label; the registry name is ``ema_cross``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import model_validator

from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.plugin import register_signal
from aegis.strategies.signal import BarView, IndicatorSpec, Signal, SignalModel


def detect_ema_cross(
    previous: tuple[float, float] | None,
    fast: float,
    slow: float,
    *,
    ready: bool,
) -> tuple[bool, bool]:
    """Return ``(crossed_up, crossed_down)`` for one new EMA pair.

    ``previous`` is the last ``(fast, slow)`` pair seen while both averages
    were warmed up. The caller stores the new pair after this returns.
    """
    if not ready or previous is None:
        return False, False
    crossed_up = previous[0] <= previous[1] and fast > slow
    crossed_down = previous[0] >= previous[1] and fast < slow
    return crossed_up, crossed_down


class EmaCrossParams(StrategyParams):
    """Long-only EMA cross. ``fast_period`` must stay below ``slow_period``."""

    fast_period: Annotated[int, SearchMeta(kind="int", default_grid=(8, 12, 20), minimum=2)] = 20
    slow_period: Annotated[int, SearchMeta(kind="int", default_grid=(30, 50, 80), minimum=3)] = 50
    atr_period: Annotated[int, SearchMeta(kind="int", default_grid=(14,), minimum=2)] = 14
    atr_multiplier: Annotated[
        float,
        SearchMeta(kind="float", default_grid=(1.5, 2.0, 2.5), minimum=0.1),
    ] = 2.0

    @model_validator(mode="after")
    def _fast_before_slow(self) -> EmaCrossParams:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be lower than slow_period.")
        return self


class EmaCrossover(SignalModel):
    """Bullish cross enters. Bearish cross exits. Stops come from the harness."""

    def __init__(self, params: EmaCrossParams) -> None:
        super().__init__(params)
        self._prev: dict[str, tuple[float, float]] = {}

    def indicator_specs(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec(name="fast", kind="ema", period=self.params.fast_period),
            IndicatorSpec(name="slow", kind="ema", period=self.params.slow_period),
        ]

    def warmup_bars(self) -> int:
        return int(self.params.slow_period)

    def on_bar(self, bar: BarView) -> Signal:
        fast = bar.indicators.get("fast")
        slow = bar.indicators.get("slow")
        previous = self._prev.get(bar.symbol)
        ready = fast is not None and slow is not None
        crossed_up, crossed_down = detect_ema_cross(
            previous,
            0.0 if fast is None else fast,
            0.0 if slow is None else slow,
            ready=ready,
        )
        if crossed_up:
            action = Signal.enter_long()
        elif crossed_down:
            action = Signal.exit()
        else:
            action = Signal.none()
        if ready:
            self._prev[bar.symbol] = (fast, slow)
        return action


register_signal(
    name="ema_cross",
    summary="Long-only EMA crossover with an ATR stop and a 2.5R target.",
    params_model=EmaCrossParams,
    model=EmaCrossover,
    report_name="ema_crossover",
)
