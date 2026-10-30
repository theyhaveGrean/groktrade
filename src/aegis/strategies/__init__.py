"""Strategy plug-ins.

Each module under this package exposes a Pydantic ``StrategyParams`` subclass
and registers a :class:`~aegis.strategies.registry.StrategyPlugin`. Searchable
fields carry :class:`~aegis.strategies.params.SearchMeta`. The signal itself
is a :class:`~aegis.strategies.signal.SignalModel`. Orders stay on the harness.

The optimizer never imports a concrete strategy. It asks the registry.
"""

from aegis.strategies.ema import EmaCrossParams
from aegis.strategies.macd import MacdParams
from aegis.strategies.mean_reversion import MeanReversionParams
from aegis.strategies.opening_range import OpeningRangeParams
from aegis.strategies.params import StrategyParams, describe_search_space, parameter_grid
from aegis.strategies.registry import StrategyPlugin, get_strategy, list_strategies, register
from aegis.strategies.volatility_breakout import VolatilityBreakoutParams

__all__ = [
    "EmaCrossParams",
    "MacdParams",
    "MeanReversionParams",
    "OpeningRangeParams",
    "StrategyParams",
    "StrategyPlugin",
    "VolatilityBreakoutParams",
    "describe_search_space",
    "get_strategy",
    "list_strategies",
    "parameter_grid",
    "register",
]
