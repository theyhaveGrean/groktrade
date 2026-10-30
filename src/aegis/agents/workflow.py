"""Phase-3 research cycle.

Steps, in order: assign the objective, propose, initial critique, implement
and test. The loop may repeat the proposal and critique up to
``max_iterations``. The sandbox result and the naysayer findings decide
whether a candidate is eligible to advance. This module does not import the
backtest engine and does not start a backtest.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from aegis.agents.audit import AuditLog
from aegis.agents.budget import RunBudget
from aegis.agents.candidates import store_candidate
from aegis.agents.gates import (
    contradicts_locked_rules,
    decide_advancement,
    sanitize_assignment,
    unresolved_critical,
)
from aegis.agents.llm import LLMClient, OpenAIChatBackend
from aegis.agents.models import (
    AdvancementRequest,
    Critique,
    ImplementationDraft,
    ObjectiveAssignment,
    ResearchRunResult,
    StrategyProposal,
)
from aegis.agents.prompts import prompt_hash
from aegis.agents.redaction import Redactor
from aegis.agents.runner import AgentSpec, run_agent
from aegis.agents.sandbox import run_sandbox, write_scratch
from aegis.agents.signal_contract import policy_implementation
from aegis.agents.tools import RunState, ToolContext, tools_for
from aegis.config import Settings
from aegis.errors import BudgetExceeded, ResearchError

_ASSIGN = AgentSpec("orchestrator", ObjectiveAssignment)
_REVIEW = AgentSpec("orchestrator", AdvancementRequest)
_RESEARCHER = AgentSpec("researcher", StrategyProposal)
_NAYSAYER = AgentSpec("naysayer", Critique)
_IMPLEMENTER = AgentSpec("implementer", ImplementationDraft)


def run_research(
    settings: Settings,
    *,
    objective: str,
    max_iterations: int,
    client: LLMClient | None = None,
    store_dir: Path | None = None,
    audit_dir: Path | None = None,
) -> ResearchRunResult:
    """Run workflow steps 1-4. Stop before any backtest."""
    if max_iterations < 1:
        raise ResearchError("max iterations must be at least 1.")
    objective = objective.strip()
    if not objective:
        raise ResearchError("A research objective is required.")
    settings.assert_paper_mode()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    store = store_dir or settings.research_store_dir
    audits = audit_dir or settings.research_audit_dir
    scratch = store / run_id / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    redactor = Redactor()
    objective = redactor.scrub(objective)
    audit = AuditLog(audits / f"{run_id}.jsonl", run_id, redactor)
    budget = _budget(settings)
    state = RunState(objective=objective, max_iterations=max_iterations)
    context = ToolContext(
        scratch=scratch,
        universe_path=settings.universe_path,
        holdout=settings.holdout,
        cache_dir=settings.data_cache_dir,
        state=state,
        timeout_seconds=settings.research_sandbox_timeout_seconds,
        memory_mb=settings.research_sandbox_memory_mb,
    )
    backend = client or OpenAIChatBackend()
    version = settings.research_prompt_version
    policy_name = policy_implementation()
    proposal: StrategyProposal | None = None
    critique: Critique | None = None
    implementation = None
    audit.write(
        "run_start",
        "orchestrator",
        {
            "objective": redactor.scrub(objective),
            "max_iterations": max_iterations,
            "prompt_version": version,
            "prompt_hashes": {role: prompt_hash(role, version) for role in ("orchestrator", "researcher", "implementer", "naysayer")},
            "policy_implementation": policy_name,
            "models": {role: settings.model_for_agent(role) for role in ("orchestrator", "researcher", "implementer", "naysayer")},
        },
    )
    if contradicts_locked_rules(objective):
        audit.write(
            "gate",
            "orchestrator",
            {
                "note": (
                    "The objective text mentions holdout access or a risk change. "
                    "Locked constraints still apply and tools will refuse the data."
                )
            },
        )
    try:
        assignment = _call(settings, _ASSIGN, _assignment_message(objective), backend, context, budget, audit, redactor, version)
        assert isinstance(assignment, ObjectiveAssignment)
        sanitized = sanitize_assignment(assignment)
        if sanitized.dropped_constraints:
            audit.write(
                "gate",
                "orchestrator",
                {
                    "rejected_constraints": list(sanitized.dropped_constraints),
                    "note": "Dropped constraints that contradicted fixed risk limits or the holdout.",
                },
            )
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            state.iteration = iteration
            state.budget = budget.snapshot()
            proposal = _call(
                settings,
                _RESEARCHER,
                _proposal_message(sanitized.assignment, critique),
                backend,
                context,
                budget,
                audit,
                redactor,
                version,
            )
            assert isinstance(proposal, StrategyProposal)
            state.proposal_json = proposal.model_dump_json()
            critique = _call(
                settings,
                _NAYSAYER,
                _critique_message(proposal),
                backend,
                context,
                budget,
                audit,
                redactor,
                version,
            )
            assert isinstance(critique, Critique)
            state.critique_json = critique.model_dump_json()
            state.unresolved_critical = unresolved_critical(critique)
            if not state.unresolved_critical:
                break
        assert proposal is not None and critique is not None
        draft = _call(
            settings,
            _IMPLEMENTER,
            _implement_message(proposal, critique),
            backend,
            context,
            budget,
            audit,
            redactor,
            version,
        )
        assert isinstance(draft, ImplementationDraft)
        write_scratch(scratch, "candidate.py", draft.signal_source)
        write_scratch(scratch, "test_candidate.py", draft.test_source)
        sandbox = run_sandbox(
            scratch,
            holdout=settings.holdout,
            timeout_seconds=settings.research_sandbox_timeout_seconds,
            memory_mb=settings.research_sandbox_memory_mb,
            cache_dir=settings.data_cache_dir,
        )
        implementation = store_candidate(
            store_dir=store,
            run_id=run_id,
            strategy_name=draft.strategy_name,
            signal_source=draft.signal_source,
            test_source=draft.test_source,
            proposal=proposal,
            critique=critique,
            sandbox=sandbox,
            policy_implementation=policy_name,
            prompt_version=version,
            notes=draft.notes,
        )
        state.candidate_hash = implementation.content_hash
        state.budget = budget.snapshot()
        audit.write(
            "candidate",
            "implementer",
            {
                "content_hash": implementation.content_hash,
                "status": implementation.status,
                "candidate_dir": implementation.candidate_dir,
                "policy_implementation": policy_name,
            },
        )
        review = _call(
            settings,
            _REVIEW,
            _review_message(proposal, critique, implementation),
            backend,
            context,
            budget,
            audit,
            redactor,
            version,
        )
        assert isinstance(review, AdvancementRequest)
        decision = decide_advancement(review, critique, implementation)
        audit.write(
            "gate",
            "orchestrator",
            {
                "requested_advance": review.advance_to_backtest,
                "requested_risk_override": review.override_risk_limits,
                "requested_validation_override": review.override_validation,
                "rejected_risk_override": decision.rejected_risk_override,
                "rejected_validation_override": decision.rejected_validation_override,
                "advance_to_backtest": decision.advance_to_backtest,
                "backtest_started": decision.backtest_started,
                "stop_reason": decision.stop_reason,
                "unresolved_critical": list(decision.unresolved_critical),
            },
        )
        result = ResearchRunResult(
            run_id=run_id,
            objective=objective,
            iterations=iterations,
            proposal=proposal,
            critique=critique,
            implementation=implementation,
            advance_to_backtest=decision.advance_to_backtest,
            backtest_started=False,
            stop_reason=decision.stop_reason,
            audit_path=str(audit.path),
            candidate_hash=implementation.content_hash,
            policy_implementation=policy_name,
            rejected_risk_override=decision.rejected_risk_override,
            rejected_validation_override=decision.rejected_validation_override,
        )
        audit.write("run_finish", "orchestrator", {"stop_reason": result.stop_reason, "advance_to_backtest": result.advance_to_backtest})
        return result
    except BudgetExceeded as exc:
        audit.write("budget", None, {"message": str(exc), "budget": budget.snapshot()})
        exc.partial = ResearchRunResult(
            run_id=run_id,
            objective=objective,
            iterations=state.iteration,
            proposal=proposal,
            critique=critique,
            implementation=implementation,
            advance_to_backtest=False,
            backtest_started=False,
            stop_reason=str(exc),
            audit_path=str(audit.path),
            candidate_hash=None if implementation is None else implementation.content_hash,
            policy_implementation=policy_name,
        )
        raise
    finally:
        audit.close()


def _call(settings, spec, message, client, context, budget, audit, redactor, version):
    context.state.budget = budget.snapshot()
    return run_agent(
        spec,
        message,
        client=client,
        model=settings.model_for_agent(spec.name),
        tools=tools_for(spec.name),
        context=context,
        budget=budget,
        audit=audit,
        redactor=redactor,
        prompt_version=version,
    )


def _budget(settings: Settings) -> RunBudget:
    return RunBudget(
        max_input_tokens=settings.research_max_input_tokens,
        max_output_tokens=settings.research_max_output_tokens,
        max_cost_usd=settings.research_max_cost_usd,
        max_llm_calls=settings.research_max_llm_calls,
        price_input_per_mtok=settings.research_price_input_per_mtok,
        price_output_per_mtok=settings.research_price_output_per_mtok,
    )


def _assignment_message(objective: str) -> str:
    return (
        "Step 1. Assign this research objective. Restate it, list success criteria, "
        "and list constraints. Do not relax risk limits or request holdout data.\n\n"
        f"Objective:\n{objective}"
    )


def _proposal_message(assignment: ObjectiveAssignment, critique: Critique | None) -> str:
    lines = [
        "Step 2. Propose one testable strategy.",
        f"Objective: {assignment.objective}",
        "Constraints:",
        *[f"- {item}" for item in assignment.constraints],
    ]
    if critique is not None:
        lines.append("The previous critique is unresolved. Revise the hypothesis to address it.")
        lines.append(critique.model_dump_json())
    return "\n".join(lines)


def _critique_message(proposal: StrategyProposal) -> str:
    return (
        "Step 3. Initial critique of this proposal, before any backtest.\n\n"
        + proposal.model_dump_json(indent=2)
    )


def _implement_message(proposal: StrategyProposal, critique: Critique) -> str:
    return (
        "Step 4. Implement the proposal as a signal plug-in and tests. "
        "The sandbox will re-run whatever you submit. A critical critique does not "
        "give you permission to change risk limits or read the holdout.\n\n"
        "Proposal:\n"
        + proposal.model_dump_json(indent=2)
        + "\n\nCritique:\n"
        + critique.model_dump_json(indent=2)
    )


def _review_message(proposal: StrategyProposal, critique: Critique, implementation) -> str:
    return (
        "The implementation step finished. Request advancement only if you believe the gates pass. "
        "Leave override_risk_limits and override_validation false. The runtime will not start a backtest.\n\n"
        f"Proposal: {proposal.name}\n"
        f"Critique verdict: {critique.verdict}\n"
        f"Unresolved critical: {unresolved_critical(critique)}\n"
        f"Sandbox status: {implementation.status}\n"
        f"Content hash: {implementation.content_hash}\n"
        f"Reason from the sandbox: {implementation.sandbox_log}"
    )
