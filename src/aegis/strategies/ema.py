"""EMA crossover plugin.

The Nautilus strategy class stays in ``aegis.backtest.strategy``. This module
is the registry entry: parameters, search space, and the call that resamples
bars before the backtest.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import model_validator

from aegis.backtest.engine import run_ema_on_bars
from aegis.backtest.report import PerformanceReport
from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.strategies.params import SearchMeta, StrategyParams
from aegis.strategies.registry import StrategyPlugin, register
from aegis.strategies.resample import resample_bars, timeframe_minutes


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


def run_ema_cross(bars, params: EmaCrossParams, **kwargs: Any) -> PerformanceReport:
    """Resample to ``params.bar_timeframe`` and run the EMA backtest."""
    minutes = timeframe_minutes(params.bar_timeframe)
    prepared = resample_bars(bars, params.bar_timeframe)
    kill_switch = bool(kwargs.pop("kill_switch", False))
    return run_ema_on_bars(
        prepared,
        fast_period=params.fast_period,
        slow_period=params.slow_period,
        atr_period=params.atr_period,
        atr_multiplier=params.atr_multiplier,
        bar_minutes=minutes,
        risk_engine=RiskEngine(
            RiskLimits(atr_multiplier=params.atr_multiplier, kill_switch=kill_switch)
        ),
        **kwargs,
    )


register(
    StrategyPlugin(
        name="ema_cross",
        summary="Long-only EMA crossover with an ATR stop and a 2.5R target.",
        params_model=EmaCrossParams,
        run=run_ema_cross,
    )
)
