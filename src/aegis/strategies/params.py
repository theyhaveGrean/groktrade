"""Pydantic parameter models with search-space metadata.

A strategy declares defaults on a ``StrategyParams`` subclass. Each tunable
field carries a :class:`SearchMeta` annotation. The optimizer builds its grid
from ``default_grid`` and replaces any axis the CLI passes as an override.
Combinations that fail model validation are dropped, so constraints such as
``fast_period < slow_period`` live on the model rather than in the optimizer.

``json_schema_extra={"search": {...}}`` is accepted as well, for models that
prefer to attach the spec on ``Field``.
"""

from __future__ import annotations

import itertools
import random
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo


class SearchMeta:
    """Annotated metadata describing how one parameter is searched."""

    def __init__(
        self,
        *,
        kind: str,
        default_grid: tuple[Any, ...] | list[Any],
        minimum: float | None = None,
        maximum: float | None = None,
        choices: tuple[Any, ...] | list[Any] | None = None,
    ) -> None:
        if kind not in {"int", "float", "choice"}:
            raise ValueError(f"Unsupported search kind {kind!r}.")
        if not default_grid:
            raise ValueError("default_grid must not be empty.")
        self.kind = kind
        self.default_grid = tuple(default_grid)
        self.minimum = minimum
        self.maximum = maximum
        self.choices = None if choices is None else tuple(choices)


class StrategyParams(BaseModel):
    """Base parameter model. Every strategy inherits the bar-size axis."""

    model_config = ConfigDict(extra="forbid")

    bar_timeframe: Annotated[
        str,
        SearchMeta(
            kind="choice",
            default_grid=("1min", "5min", "15min"),
            choices=("1min", "5min", "15min", "30min", "60min"),
        ),
    ] = "1min"

    def label(self) -> str:
        parts = []
        for name, value in self.model_dump().items():
            if isinstance(value, float):
                parts.append(f"{name}={value:g}")
            else:
                parts.append(f"{name}={value}")
        return " ".join(parts)


def search_spec(field: FieldInfo) -> SearchMeta | None:
    for item in field.metadata:
        if isinstance(item, SearchMeta):
            return item
    extra = field.json_schema_extra
    if isinstance(extra, dict) and isinstance(extra.get("search"), dict):
        return SearchMeta(**extra["search"])
    return None


def parameter_grid(
    model: type[StrategyParams],
    overrides: dict[str, list[Any]] | None = None,
    *,
    method: str = "grid",
    samples: int = 20,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Valid parameter dicts for ``model``.

    ``overrides`` replaces ``default_grid`` for the named fields. ``method``
    is ``grid`` or ``random`` (a seeded sample of the grid, without replacement).
    """
    if method not in {"grid", "random"}:
        raise ValueError("method must be 'grid' or 'random'.")
    axes = _axes(model, overrides or {})
    names = [name for name, _values in axes]
    pool: list[dict[str, Any]] = []
    skipped: list[str] = []
    for combo in itertools.product(*(values for _name, values in axes)):
        candidate = dict(zip(names, combo, strict=True))
        try:
            parsed = model.model_validate(candidate)
        except ValidationError as exc:
            skipped.append(str(exc.errors()[0]["msg"]))
            continue
        pool.append(parsed.model_dump())
    if not pool:
        detail = skipped[0] if skipped else "no combinations were produced"
        raise ValueError(f"No valid parameter combinations for {model.__name__}: {detail}")
    pool.sort(key=lambda item: _stable_key(item))
    if method == "random":
        if samples < 1:
            raise ValueError("samples must be positive.")
        if samples < len(pool):
            pool = random.Random(seed).sample(pool, samples)
            pool.sort(key=lambda item: _stable_key(item))
    return pool


def describe_search_space(model: type[StrategyParams]) -> str:
    """Human-readable axes, defaults, and how many grid points validate."""
    lines = [f"{model.__name__}"]
    for name, field in model.model_fields.items():
        spec = search_spec(field)
        default = field.default
        if spec is None:
            lines.append(f"  {name}: fixed at {default}")
            continue
        choice = "" if spec.choices is None else f" choices={list(spec.choices)}"
        bounds = ""
        if spec.minimum is not None or spec.maximum is not None:
            bounds = f" bounds=[{spec.minimum}, {spec.maximum}]"
        lines.append(
            f"  {name}: default={default} grid={list(spec.default_grid)}{choice}{bounds}"
        )
    try:
        count = len(parameter_grid(model, method="grid"))
    except ValueError as exc:
        lines.append(f"Valid grid points: 0 ({exc})")
    else:
        lines.append(f"Valid grid points: {count}")
    return "\n".join(lines)


def _axes(model: type[StrategyParams], overrides: dict[str, list[Any]]) -> list[tuple[str, list[Any]]]:
    unknown = sorted(set(overrides) - set(model.model_fields))
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} for {model.__name__}. "
            f"Known: {', '.join(model.model_fields)}."
        )
    axes: list[tuple[str, list[Any]]] = []
    for name, field in model.model_fields.items():
        spec = search_spec(field)
        if spec is None:
            if name in overrides:
                raise ValueError(f"{name} is not searchable on {model.__name__}.")
            continue
        raw_values = overrides[name] if name in overrides else list(spec.default_grid)
        if not raw_values:
            raise ValueError(f"Search axis {name} has no values.")
        coerced = [_check_value(name, field, spec, value) for value in raw_values]
        axes.append((name, coerced))
    if not axes:
        raise ValueError(f"{model.__name__} has no searchable parameters.")
    return axes


def _check_value(name: str, field: FieldInfo, spec: SearchMeta, value: Any) -> Any:
    try:
        coerced = TypeAdapter(field.annotation).validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"Invalid value {value!r} for {name}: {exc.errors()[0]['msg']}") from exc
    if spec.choices is not None and coerced not in spec.choices:
        raise ValueError(f"{name}={coerced!r} is not one of {list(spec.choices)}.")
    if isinstance(coerced, bool) or not isinstance(coerced, (int, float)):
        return coerced
    if spec.minimum is not None and coerced < spec.minimum:
        raise ValueError(f"{name}={coerced} is below the minimum {spec.minimum}.")
    if spec.maximum is not None and coerced > spec.maximum:
        raise ValueError(f"{name}={coerced} is above the maximum {spec.maximum}.")
    return coerced


def _stable_key(params: dict[str, Any]) -> str:
    return "|".join(f"{name}={params[name]}" for name in sorted(params))
