"""Plug-in contract the Implementer codes against.

The safety-critical harness belongs in ``aegis.strategies.harness`` and is
owned by the strategy refactor. This module is the stable seam:

- generated code imports :class:`Signal`, :class:`BarView`, and
  :class:`SignalModel` from here
- :func:`check_signal_source` runs the local AST policy and, when the harness
  defines the same function, that policy too
- :class:`SignalRegistry` is the stub registry. The harness registry is used
  when ``register_signal`` / ``get_signal`` are present

A signal plug-in returns an action and a reason. It does not choose quantity,
stop, or target. The sandbox feeds entry signals through
``aegis.backtest.risk.RiskEngine`` with default :class:`RiskLimits`. The
harness, once it lands, is the production caller of that engine. Agents never
edit the harness, the risk engine, the data package, or holdout code.

Reconciliation checklist when ``harness.py`` arrives:

- base class ``SignalModel`` with ``on_bar(bar, *, position_open) -> Signal``
- pydantic ``Signal`` (action, reason; no size) and ``BarView``
- ``check_signal_source(source)`` returning a report, a list of strings, or an
  object with ``ok`` and ``violations``
- ``register_signal`` / ``get_signal``
- Params search metadata under ``json_schema_extra['x-aegis-search']``
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from aegis.agents.policy import (
    SEARCH_META,
    PolicyReport,
    PolicyViolation,
    check_candidate,
    check_signal_source as _local_check,
)
from aegis.holdout import DateRange

HARNESS_MODULE = "aegis.strategies.harness"

SignalAction = Literal["enter_long", "enter_short", "exit", "hold"]


class Signal(BaseModel):
    """What a plug-in is allowed to say. Size and prices stay with the risk engine."""

    model_config = ConfigDict(extra="forbid")

    action: SignalAction
    reason: str = Field(min_length=1)


class BarView(BaseModel):
    """One bar the harness (or the sandbox probe) hands to a plug-in."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class SearchSpace(BaseModel):
    """Bounds an optimizer may search later. These are signal parameters, not risk limits."""

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    step: float | None = None
    log: bool = False


class SignalModelProtocol(Protocol):
    """Structural stub for ``aegis.strategies.harness.SignalModel``.

    A plug-in has a registry name, a pydantic params instance, and ``on_bar``.
    It does not size trades. When the harness base class lands, the sandbox
    accepts that class as well as :class:`SignalModel`.
    """

    name: str
    params: BaseModel

    def reset(self) -> None:
        """Drop indicator state between walks."""
        ...

    def on_bar(self, bar: BarView, *, position_open: bool) -> Signal:
        """Return a signal for this bar. Do not size or submit an order."""
        ...


class SignalModel(ABC):
    """Concrete stub base. Subclass this until the harness base is the one to use.

    ``name`` is the registry key. ``params`` is a pydantic model instance.
    The class matches :class:`SignalModelProtocol`.
    """

    name: str

    def reset(self) -> None:
        """Drop indicator state. The harness calls this between walks."""
        return None

    @abstractmethod
    def on_bar(self, bar: BarView, *, position_open: bool) -> Signal:
        """Return a signal for this bar. Do not size or submit an order."""


class SignalRegistry:
    """In-process registry. The sandbox child may fill one; the parent does not import plug-ins."""

    def __init__(self) -> None:
        self._by_name: dict[str, type[SignalModel]] = {}

    def register(self, cls: type[SignalModel]) -> type[SignalModel]:
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("SignalModel.name must be a non-empty string.")
        self._by_name[name] = cls
        return cls

    def get(self, name: str) -> type[SignalModel]:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"No signal named {name!r} is registered.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def clear(self) -> None:
        self._by_name.clear()


REGISTRY = SignalRegistry()


def harness_module():
    """Import the real harness, or ``None`` when that refactor has not landed."""
    try:
        return importlib.import_module(HARNESS_MODULE)
    except ImportError:
        return None


def signal_model_bases() -> tuple[type, ...]:
    """Bases the sandbox accepts. The stub is always included."""
    bases: list[type] = [SignalModel]
    module = harness_module()
    harness_base = None if module is None else getattr(module, "SignalModel", None)
    if isinstance(harness_base, type) and harness_base is not SignalModel:
        bases.append(harness_base)
    return tuple(bases)


def is_signal_model(cls: object) -> bool:
    if not isinstance(cls, type):
        return False
    for base in signal_model_bases():
        try:
            matches = issubclass(cls, base)
        except TypeError:
            # A Protocol base is not always acceptable to issubclass.
            continue
        if matches and cls is not base:
            return True
    return False


def register_signal(cls: type[SignalModel]) -> type[SignalModel]:
    """Register on the harness when it exposes ``register_signal``, else on the stub."""
    module = harness_module()
    harness_register = None if module is None else getattr(module, "register_signal", None)
    if callable(harness_register):
        harness_register(cls)
        return cls
    return REGISTRY.register(cls)


def get_signal(name: str) -> type[SignalModel]:
    module = harness_module()
    harness_get = None if module is None else getattr(module, "get_signal", None)
    if callable(harness_get):
        return harness_get(name)
    return REGISTRY.get(name)


def search_space(params_model: type[BaseModel]) -> dict[str, SearchSpace]:
    """Read ``x-aegis-search`` metadata off a Params model."""
    found: dict[str, SearchSpace] = {}
    for name, field in params_model.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict) or SEARCH_META not in extra:
            continue
        payload = extra[SEARCH_META]
        if isinstance(payload, dict):
            found[name] = SearchSpace.model_validate(payload)
    return found


def policy_implementation() -> str:
    module = harness_module()
    if module is not None and callable(getattr(module, "check_signal_source", None)):
        return HARNESS_MODULE
    return "stub"


def check_signal_source(
    source: str,
    *,
    filename: str = "candidate.py",
    holdout: DateRange | None = None,
    kind: str = "signal",
) -> PolicyReport:
    """Local AST policy, plus the harness policy when that function exists."""
    local = _local_check(source, filename=filename, holdout=holdout, kind=kind)
    harness_fn = _harness_policy()
    if harness_fn is None:
        return local
    extra = _call_harness_policy(harness_fn, source, filename)
    violations = [*local.violations, *extra]
    return PolicyReport(ok=not violations, violations=violations, implementation=HARNESS_MODULE)


def check_pair(
    signal_source: str,
    test_source: str,
    *,
    holdout: DateRange | None = None,
) -> PolicyReport:
    """Check both scratch files. Harness policy is applied to the signal source."""
    local = check_candidate(signal_source, test_source, holdout=holdout)
    harness_fn = _harness_policy()
    if harness_fn is None:
        return local
    extra = _call_harness_policy(harness_fn, signal_source, "candidate.py")
    violations = [*local.violations, *extra]
    return PolicyReport(ok=not violations, violations=violations, implementation=HARNESS_MODULE)


def _harness_policy() -> Callable | None:
    module = harness_module()
    if module is None:
        return None
    fn = getattr(module, "check_signal_source", None)
    return fn if callable(fn) else None


def _call_harness_policy(fn: Callable, source: str, filename: str) -> list[PolicyViolation]:
    attempts = (
        lambda: fn(source),
        lambda: fn(source, filename=filename),
        lambda: fn(source=source),
    )
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return _coerce_policy(attempt())
        except TypeError as exc:
            if "argument" not in str(exc) and "positional" not in str(exc):
                raise
            last_error = exc
    if last_error is not None:
        return [
            PolicyViolation(
                code="harness_policy",
                message=f"Could not call aegis.strategies.harness.check_signal_source: {last_error}",
            )
        ]
    return []


def _coerce_policy(result: object) -> list[PolicyViolation]:
    if result is None:
        return []
    if isinstance(result, PolicyReport):
        return list(result.violations)
    ok = getattr(result, "ok", None)
    violations = getattr(result, "violations", None)
    if isinstance(violations, list):
        coerced: list[PolicyViolation] = []
        for item in violations:
            if isinstance(item, PolicyViolation):
                coerced.append(item)
            elif isinstance(item, str):
                coerced.append(PolicyViolation(code="harness", message=item))
            else:
                message = getattr(item, "message", None) or str(item)
                code = getattr(item, "code", None) or "harness"
                line = getattr(item, "line", None)
                coerced.append(PolicyViolation(code=str(code), message=str(message), line=line))
        if ok is False and not coerced:
            coerced.append(PolicyViolation(code="harness", message="Harness policy reported a failure."))
        return coerced
    if isinstance(result, list):
        return [PolicyViolation(code="harness", message=str(item)) for item in result]
    if ok is True:
        return []
    return [PolicyViolation(code="harness", message=str(result))]


EXAMPLE_SIGNAL_SOURCE = '''from pydantic import BaseModel, Field

from aegis.agents.signal_contract import BarView, Signal, SignalModel


class Params(BaseModel):
    lookback: int = Field(
        2,
        json_schema_extra={"x-aegis-search": {"low": 2, "high": 10, "step": 1}},
    )


class MomentumSignal(SignalModel):
    """Scratch example. The risk engine sizes any entry this returns."""

    name = "momentum_scratch"

    def __init__(self, params: Params | None = None) -> None:
        self.params = params or Params()
        self._closes: list[float] = []

    def reset(self) -> None:
        self._closes = []

    def on_bar(self, bar: BarView, *, position_open: bool) -> Signal:
        self._closes.append(bar.close)
        if position_open:
            return Signal(action="exit", reason="scratch test exits on the next bar")
        if len(self._closes) <= self.params.lookback:
            return Signal(action="hold", reason="warming up")
        previous = self._closes[-1 - self.params.lookback]
        if bar.close > previous:
            return Signal(action="enter_long", reason="close above lookback")
        return Signal(action="hold", reason="no setup")
'''


EXAMPLE_TEST_SOURCE = '''import candidate
from datetime import datetime, timezone

from aegis.agents.signal_contract import BarView


def _bar(price: float) -> BarView:
    return BarView(
        symbol="AAPL",
        ts=datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000,
    )


def test_holds_while_warming():
    model = candidate.MomentumSignal()
    signal = model.on_bar(_bar(10), position_open=False)
    assert signal.action == "hold"


def test_enters_when_price_rises():
    model = candidate.MomentumSignal()
    model.on_bar(_bar(10), position_open=False)
    model.on_bar(_bar(10), position_open=False)
    signal = model.on_bar(_bar(12), position_open=False)
    assert signal.action == "enter_long"
    assert signal.reason
'''
