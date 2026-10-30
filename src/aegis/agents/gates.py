"""Decisions the orchestrator is not allowed to make.

Risk limits are not a parameter of these functions. A request to override
them, or to ignore a failed naysayer verdict or a failed sandbox, is recorded
and dropped. Nothing here starts a backtest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aegis.agents.models import AdvancementRequest, Critique, ImplementationResult, ObjectiveAssignment

LOCKED_CONSTRAINTS = (
    "Risk limits stay on the default aegis.backtest.risk.RiskLimits. Agents cannot change them.",
    "The holdout window cannot be read.",
    "A failed naysayer verdict or a failed sandbox cannot be overridden.",
    "The implementer may only write a signal module, its Params model, and tests in the scratch workspace.",
    "The harness, risk engine, data package, and holdout module are read-only.",
    "A walk-forward backtest runs only after the naysayer and the sandbox pass, and only inside the harness.",
)

_CONTRADICTION = re.compile(
    r"holdout|skip the naysayer|ignore the naysayer|override validation|override risk|"
    r"risk per trade|max_risk|disable the kill|kill switch off|read_bars|loosen",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SanitizedAssignment:
    assignment: ObjectiveAssignment
    dropped_constraints: tuple[str, ...]


@dataclass(frozen=True)
class GateDecision:
    advance_to_backtest: bool
    backtest_started: bool
    stop_reason: str
    rejected_risk_override: bool
    rejected_validation_override: bool
    unresolved_critical: tuple[str, ...]


def unresolved_critical(critique: Critique) -> list[str]:
    return [finding.code for finding in critique.findings if finding.unresolved_critical]


def sanitize_assignment(assignment: ObjectiveAssignment) -> SanitizedAssignment:
    """Keep the model's safe constraints and always append the locked ones."""
    kept: list[str] = []
    dropped: list[str] = []
    for constraint in assignment.constraints:
        if _CONTRADICTION.search(constraint):
            dropped.append(constraint)
        else:
            kept.append(constraint)
    for locked in LOCKED_CONSTRAINTS:
        if locked not in kept:
            kept.append(locked)
    cleaned = assignment.model_copy(update={"constraints": kept})
    return SanitizedAssignment(assignment=cleaned, dropped_constraints=tuple(dropped))


def contradicts_locked_rules(text: str) -> bool:
    """True when text asks to skip the naysayer, read the holdout, or loosen risk."""
    return _CONTRADICTION.search(text) is not None


def decide_advancement(
    request: AdvancementRequest,
    critique: Critique,
    implementation: ImplementationResult | None,
) -> GateDecision:
    """Apply the phase-3 gate. ``backtest_started`` is always false.

    There is no risk-limit argument. Callers cannot pass a looser cap through
    this function, and a true override flag does not change the decision.
    """
    critical = tuple(unresolved_critical(critique))
    rejected_risk = bool(request.override_risk_limits)
    validation_failed = bool(critical) or critique.verdict == "fail" or (
        implementation is None or implementation.status != "passed"
    )
    rejected_validation = bool(request.override_validation) and validation_failed
    # An override flag with nothing to override is still rejected so the
    # request cannot be mistaken for a successful bypass.
    if request.override_validation and not validation_failed:
        rejected_validation = True

    if critical:
        reason = (
            "Unresolved critical naysayer findings ("
            + ", ".join(critical)
            + "); backtest not started."
        )
        return _blocked(reason, rejected_risk, rejected_validation or bool(request.override_validation), critical)
    if critique.verdict == "fail":
        return _blocked(
            "Naysayer verdict is fail; backtest not started.",
            rejected_risk,
            True if request.override_validation else rejected_validation,
            critical,
        )
    if implementation is None or implementation.status != "passed":
        status = "missing" if implementation is None else implementation.status
        return _blocked(
            f"Signal module did not pass the sandbox ({status}); backtest not started.",
            rejected_risk,
            True if request.override_validation else rejected_validation,
            critical,
        )
    if not request.advance_to_backtest:
        return GateDecision(
            advance_to_backtest=False,
            backtest_started=False,
            stop_reason="Gates passed and the orchestrator did not request advancement. Backtest not started.",
            rejected_risk_override=rejected_risk,
            rejected_validation_override=rejected_validation,
            unresolved_critical=critical,
        )
    return GateDecision(
        advance_to_backtest=True,
        backtest_started=False,
        stop_reason="Gates passed. The harness walk-forward may start.",
        rejected_risk_override=rejected_risk,
        rejected_validation_override=rejected_validation,
        unresolved_critical=critical,
    )


def _blocked(
    reason: str,
    rejected_risk: bool,
    rejected_validation: bool,
    critical: tuple[str, ...],
) -> GateDecision:
    return GateDecision(
        advance_to_backtest=False,
        backtest_started=False,
        stop_reason=reason,
        rejected_risk_override=rejected_risk,
        rejected_validation_override=rejected_validation,
        unresolved_critical=critical,
    )
