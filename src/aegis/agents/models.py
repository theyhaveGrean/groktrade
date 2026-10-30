"""Structured outputs for the phase-3 research workflow.

The model fills these objects. Gates in :mod:`aegis.agents.gates` decide what
happens next, and they read the findings rather than trusting a verdict the
model can flip.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Rule(_Strict):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, str | int | float] = Field(default_factory=dict)


class StrategyProposal(_Strict):
    """Researcher output. Entry, exit, assumptions, and a hypothesis that can fail."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    summary: str = Field(min_length=1)
    entry_rules: list[Rule] = Field(min_length=1)
    exit_rules: list[Rule] = Field(min_length=1)
    market_assumptions: list[str] = Field(min_length=1)
    falsifiable_hypothesis: str = Field(min_length=12)
    indicators: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)


class Finding(_Strict):
    code: str = Field(min_length=1)
    severity: Literal["critical", "major", "minor"]
    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=8)
    resolved: bool = False
    resolution: str | None = None

    @property
    def unresolved_critical(self) -> bool:
        if self.severity != "critical":
            return False
        if not self.resolved:
            return True
        return not (self.resolution or "").strip()


class Critique(_Strict):
    """Naysayer output. Pass/fail is restated from the findings."""

    verdict: Literal["pass", "fail"]
    findings: list[Finding] = Field(default_factory=list)
    stress_tests: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _verdict_follows_critical_findings(self) -> Critique:
        if any(finding.unresolved_critical for finding in self.findings):
            self.verdict = "fail"
        return self


class ObjectiveAssignment(_Strict):
    objective: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    notes: str = ""


class AdvancementRequest(_Strict):
    """What the orchestrator asks for. The gate may refuse it."""

    advance_to_backtest: bool
    override_risk_limits: bool = False
    override_validation: bool = False
    reason: str = Field(min_length=1)


class ImplementationDraft(_Strict):
    """Sources the implementer claims to have written. The sandbox re-runs them."""

    strategy_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    signal_source: str = Field(min_length=1, max_length=60_000)
    test_source: str = Field(min_length=1, max_length=60_000)
    notes: str = ""


class ImplementationResult(_Strict):
    strategy_name: str
    code_path: str
    test_path: str
    status: Literal["passed", "failed", "rejected"]
    content_hash: str
    tests_run: int
    tests_passed: int
    sandbox_log: str
    notes: str
    candidate_dir: str


class SandboxResult(_Strict):
    status: Literal["passed", "failed", "rejected"]
    policy_ok: bool
    violations: list[str] = Field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    failures: list[str] = Field(default_factory=list)
    log: str = ""
    timed_out: bool = False
    resource_killed: bool = False
    risk_applied: bool = False
    risk_quantity: int | None = None
    risk_max_risk_per_trade: float | None = None
    secrets_in_env: bool = False
    signal_class: str | None = None


class ResearchRunResult(_Strict):
    run_id: str
    objective: str
    iterations: int
    proposal: StrategyProposal | None = None
    critique: Critique | None = None
    implementation: ImplementationResult | None = None
    advance_to_backtest: bool
    backtest_started: bool
    stop_reason: str
    audit_path: str
    candidate_hash: str | None = None
    policy_implementation: str
    rejected_risk_override: bool = False
    rejected_validation_override: bool = False
    trial_ledger: str | None = None
