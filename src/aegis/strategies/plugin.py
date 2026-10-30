"""Register a signal model as a :class:`~aegis.strategies.registry.StrategyPlugin`.

Strategy modules call :func:`register_signal`. The ``run`` closure, which
resamples bars and hands them to the harness, lives here so signal modules
do not import Nautilus or the order path.
"""

from __future__ import annotations

from typing import Any

from aegis.strategies.params import StrategyParams
from aegis.strategies.registry import StrategyPlugin, register
from aegis.strategies.signal import SignalModel


def register_signal(
    *,
    name: str,
    summary: str,
    params_model: type[StrategyParams],
    model: type[SignalModel],
    report_name: str | None = None,
) -> StrategyPlugin:
    """Register ``model`` under ``name``. ``aegis optimize`` calls ``run``."""
    label = report_name or name

    def run(bars, params: Any, **kwargs: Any):
        from aegis.backtest.risk import RiskEngine, RiskLimits
        from aegis.strategies.harness import run_signal_on_bars
        from aegis.strategies.resample import resample_bars, timeframe_minutes

        parsed = params if isinstance(params, params_model) else params_model.model_validate(params)
        kill_switch = bool(kwargs.pop("kill_switch", False))
        risk_engine = kwargs.pop("risk_engine", None)
        multiplier = float(getattr(parsed, "atr_multiplier", 2.0))
        if risk_engine is None:
            risk_engine = RiskEngine(RiskLimits(atr_multiplier=multiplier, kill_switch=kill_switch))
        prepared = resample_bars(bars, parsed.bar_timeframe)
        atr_period = int(getattr(parsed, "atr_period", 14))
        holdout = kwargs.pop("holdout")
        return run_signal_on_bars(
            prepared,
            holdout=holdout,
            signal_model=model,
            params=parsed,
            strategy_name=label,
            bar_minutes=timeframe_minutes(parsed.bar_timeframe),
            atr_period=atr_period,
            risk_engine=risk_engine,
            **kwargs,
        )

    return register(
        StrategyPlugin(
            name=name,
            summary=summary,
            params_model=params_model,
            run=run,
        )
    )
