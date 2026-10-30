"""Strategy registry. Plugins register by name at import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from aegis.backtest.report import PerformanceReport
from aegis.holdout import DateRange
from aegis.strategies.params import StrategyParams


@dataclass(frozen=True)
class StrategyPlugin:
    """One runnable strategy.

    ``run`` receives 1-minute bars. It should resample with
    :func:`aegis.strategies.resample.resample_bars` when ``params.bar_timeframe``
    is coarser than one minute, then hand that frame to the backtest.
    """

    name: str
    summary: str
    params_model: type[StrategyParams]
    run: Callable[..., PerformanceReport]


_REGISTRY: dict[str, StrategyPlugin] = {}


def register(plugin: StrategyPlugin) -> StrategyPlugin:
    if plugin.name in _REGISTRY:
        raise ValueError(f"Strategy {plugin.name!r} is already registered.")
    _REGISTRY[plugin.name] = plugin
    return plugin


def get_strategy(name: str) -> StrategyPlugin:
    _ensure_loaded()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown strategy {name!r}. Known strategies: {known}.")
    return _REGISTRY[name]


def list_strategies() -> tuple[StrategyPlugin, ...]:
    _ensure_loaded()
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def _ensure_loaded() -> None:
    # Importing the package registers built-in plugins. A second import is free.
    import aegis.strategies as _strategies

    del _strategies
