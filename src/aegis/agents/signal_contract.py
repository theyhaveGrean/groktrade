"""Plug-in seam for the research sandbox.

The classes and the two gates live on the harness stack:

- :class:`aegis.strategies.signal.SignalModel` (re-exported by ``aegis.strategies.harness``)
- :func:`aegis.strategies.policy.check_signal_source`
- :func:`aegis.strategies.plugin.register_signal`

This module re-exports them without importing the harness file. That file
imports Nautilus, and the sandbox child has a small address-space limit.
Generated code imports ``aegis.strategies.signal`` and ``aegis.strategies.params``.
"""

from __future__ import annotations

from collections.abc import Callable

from aegis.agents.policy import PolicyReport, PolicyViolation
from aegis.agents.policy import check_candidate
from aegis.agents.policy import check_signal_source as _local_check
from aegis.holdout import DateRange
from aegis.strategies.signal import BarView, Signal, SignalAction, SignalModel

HARNESS_MODULE = "aegis.strategies.harness"

__all__ = [
    "BarView",
    "EXAMPLE_SIGNAL_SOURCE",
    "EXAMPLE_TEST_SOURCE",
    "Signal",
    "SignalAction",
    "SignalModel",
    "check_pair",
    "check_signal_source",
    "get_signal",
    "is_signal_model",
    "policy_implementation",
    "register_signal",
]


def is_signal_model(cls: object) -> bool:
    return isinstance(cls, type) and issubclass(cls, SignalModel) and cls is not SignalModel


def register_signal(*args, **kwargs):
    """Register on the harness registry. Same function ``harness.register_signal`` calls."""
    from aegis.strategies.plugin import register_signal as _register

    return _register(*args, **kwargs)


def get_signal(name: str) -> type[SignalModel]:
    from aegis.strategies.registry import get_strategy

    try:
        plugin = get_strategy(name)
    except ValueError as exc:
        raise KeyError(str(exc)) from exc
    model = plugin.model
    if not isinstance(model, type) or not issubclass(model, SignalModel):
        raise KeyError(f"No signal model is registered as {name!r}.")
    return model


def policy_implementation() -> str:
    """The harness owns the policy. The sandbox calls the same function object."""
    return HARNESS_MODULE


def check_signal_source(
    source: str,
    *,
    filename: str = "candidate.py",
    holdout: DateRange | None = None,
    kind: str = "signal",
) -> PolicyReport:
    """Agent AST policy, plus the harness allowlist."""
    local = _local_check(source, filename=filename, holdout=holdout, kind=kind)
    extra = _call_harness_policy(_harness_policy(), source, filename)
    violations = [*local.violations, *extra]
    return PolicyReport(ok=not violations, violations=violations, implementation=HARNESS_MODULE)


def check_pair(
    signal_source: str,
    test_source: str,
    *,
    holdout: DateRange | None = None,
) -> PolicyReport:
    """Check both scratch files. The harness policy runs on the signal source."""
    local = check_candidate(signal_source, test_source, holdout=holdout)
    extra = _call_harness_policy(_harness_policy(), signal_source, "candidate.py")
    violations = [*local.violations, *extra]
    return PolicyReport(ok=not violations, violations=violations, implementation=HARNESS_MODULE)


def _harness_policy() -> Callable:
    from aegis.strategies.policy import check_signal_source as harness_check

    return harness_check


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
    if ok is None and hasattr(result, "passed"):
        ok = bool(getattr(result, "passed"))
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


EXAMPLE_SIGNAL_SOURCE = '''from aegis.strategies.params import StrategyParams
from aegis.strategies.signal import BarView, Signal, SignalModel


class Params(StrategyParams):
    lookback: int = 2


class MomentumSignal(SignalModel):
    """Scratch example. The harness sizes any entry. Flat while warming up."""

    def __init__(self, params: Params) -> None:
        super().__init__(params)
        self._closes: list[float] = []

    def on_bar(self, bar: BarView) -> Signal:
        self._closes.append(bar.close)
        if len(self._closes) <= int(self.params.lookback):
            return Signal.none()
        previous = self._closes[-1 - int(self.params.lookback)]
        if bar.close > previous:
            return Signal.enter_long()
        return Signal.none()
'''


EXAMPLE_TEST_SOURCE = '''import candidate
from datetime import datetime, timezone

from aegis.strategies.signal import BarView


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
    model = candidate.MomentumSignal(candidate.Params())
    signal = model.on_bar(_bar(10))
    assert signal.action == "none"


def test_enters_when_price_rises():
    model = candidate.MomentumSignal(candidate.Params())
    model.on_bar(_bar(10))
    model.on_bar(_bar(10))
    signal = model.on_bar(_bar(12))
    assert signal.action == "enter_long"
'''
