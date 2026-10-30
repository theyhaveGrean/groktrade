"""Strategy plug-ins.

Each module under this package exposes a Pydantic ``Params`` model and
registers a :class:`~aegis.strategies.registry.StrategyPlugin` by name.
Searchable fields carry :class:`~aegis.strategies.params.SearchMeta` (or the
equivalent ``Field(json_schema_extra={"search": ...})``). The optimizer never
imports a concrete strategy; it asks the registry.

Built-in name: ``ema_cross``.
"""

from aegis.strategies.ema import EmaCrossParams
from aegis.strategies.params import StrategyParams, describe_search_space, parameter_grid
from aegis.strategies.registry import StrategyPlugin, get_strategy, list_strategies, register

__all__ = [
    "EmaCrossParams",
    "StrategyParams",
    "StrategyPlugin",
    "describe_search_space",
    "get_strategy",
    "list_strategies",
    "parameter_grid",
    "register",
]
