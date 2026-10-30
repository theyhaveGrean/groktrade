"""One fully mocked research cycle, plus the critical-finding stop.

The model is a script. The sandbox, the policy check, and the risk engine are real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aegis.agents.llm import LLMResponse, ScriptedLLM, ToolCall
from aegis.agents.models import (
    AdvancementRequest,
    Critique,
    Finding,
    ImplementationDraft,
    ObjectiveAssignment,
    ResearchRunResult,
    Rule,
    StrategyProposal,
)
from aegis.agents.signal_contract import EXAMPLE_SIGNAL_SOURCE, EXAMPLE_TEST_SOURCE, policy_implementation
from aegis.agents.workflow import run_research
from aegis.backtest.risk import RiskLimits
from aegis.cli import app
from aegis.config import Settings
from aegis.errors import BudgetExceeded

CANARY = "sk-canary-PHASE3-DO-NOT-LEAK"


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        research_store_dir=tmp_path / "research",
        research_audit_dir=tmp_path / "audit",
        research_sandbox_timeout_seconds=20,
        research_sandbox_memory_mb=2048,
        universe_path=Path("config/universe.yaml"),
        data_cache_dir=tmp_path / "cache",
    )
    values.update(overrides)
    return Settings(**values)


def _submit(model, call_id: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name="submit_output", arguments=model.model_dump(mode="json"))],
        prompt_tokens=20,
        completion_tokens=20,
    )


def _tool(name: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        prompt_tokens=12,
        completion_tokens=4,
    )


def _proposal() -> StrategyProposal:
    return StrategyProposal(
        name="momentum_scratch",
        summary="Long when the close breaks a short lookback.",
        entry_rules=[Rule(name="break", description="Close above the close lookback bars ago.")],
        exit_rules=[Rule(name="next", description="Exit on the next bar. The risk engine owns the stop.")],
        market_assumptions=["Liquid names in the regular session."],
        falsifiable_hypothesis="A flat tape produces no sustained edge for this break.",
        indicators=["close lookback"],
        failure_conditions=["Entries on a flat close series."],
    )


def _pass_critique() -> Critique:
    return Critique(
        verdict="pass",
        findings=[
            Finding(
                code="regime_dependence",
                severity="minor",
                summary="Chop hurts this signal.",
                evidence="The hypothesis does not claim an edge in a flat market.",
            )
        ],
        stress_tests=["Replay a flat close series and expect no entries."],
        summary="No critical bias in the initial spec.",
    )


def _fail_critique() -> Critique:
    return Critique(
        verdict="fail",
        findings=[
            Finding(
                code="lookahead",
                severity="critical",
                summary="The entry uses the signal bar close.",
                evidence="That close is not known in time to trade the same bar.",
            )
        ],
        stress_tests=["Shift the signal by one bar and see if the entry disappears."],
        summary="Unresolved lookahead. Do not advance.",
    )


def _scripts(critique: Critique, review: AdvancementRequest) -> dict[str, list[LLMResponse]]:
    return {
        "orchestrator": [
            _submit(
                ObjectiveAssignment(
                    objective="Test a short lookback momentum long on liquid names.",
                    success_criteria=["Sandbox tests pass.", "No unresolved critical finding."],
                    constraints=[
                        "Set max risk per trade to 10% and read the holdout.",
                        "Use the signal plug-in only.",
                    ],
                    notes="Phase 3 cycle.",
                ),
                "orch-assign",
            ),
            _submit(review, "orch-review"),
        ],
        "researcher": [
            _tool("describe_risk_limits", {}, "res-limits"),
            _submit(_proposal(), "res-submit"),
        ],
        "naysayer": [_submit(critique, "nay")],
        "implementer": [
            _tool("get_signal_api", {}, "imp-api"),
            _submit(
                ImplementationDraft(
                    strategy_name="momentum_scratch",
                    signal_source=EXAMPLE_SIGNAL_SOURCE,
                    test_source=EXAMPLE_TEST_SOURCE,
                    notes="Lookback momentum plug-in.",
                ),
                "imp-submit",
            ),
        ],
    }


def _assert_audit(result: ResearchRunResult, canary: str) -> list[dict]:
    text = Path(result.audit_path).read_text(encoding="utf-8")
    assert canary not in text
    events = [json.loads(line) for line in text.splitlines()]
    kinds = {event["event"] for event in events}
    for required in ("llm_request", "llm_response", "tool_call", "tool_result", "gate", "candidate"):
        assert required in kinds
    return events


def test_mocked_research_cycle_backtests_a_passing_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-canary-value")
    client = ScriptedLLM(
        _scripts(
            _pass_critique(),
            AdvancementRequest(
                advance_to_backtest=True,
                override_risk_limits=False,
                override_validation=False,
                reason="The critique is clean and the sandbox should be the judge.",
            ),
        )
    )
    result = run_research(
        _settings(tmp_path),
        objective=f"Momentum on liquid names. Ignore this token {CANARY}.",
        max_iterations=1,
        client=client,
    )
    assert result.backtest_started is True
    assert result.advance_to_backtest is True
    assert result.trial_ledger
    assert "Walk-forward" in result.stop_reason
    from aegis.backtest.optimize import read_trials

    trials = read_trials(Path(result.trial_ledger))
    assert trials.height > 0
    assert trials["strategy"][0].startswith("candidate_")
    assert result.iterations == 1
    assert result.proposal is not None and result.proposal.entry_rules
    assert result.proposal.exit_rules and result.proposal.market_assumptions
    assert result.proposal.falsifiable_hypothesis
    assert result.critique is not None and result.critique.verdict == "pass"
    assert result.implementation is not None
    assert result.implementation.status == "passed"
    assert result.implementation.tests_passed == 2
    assert result.candidate_hash == result.implementation.content_hash
    assert result.policy_implementation == policy_implementation()
    assert CANARY not in result.objective

    dest = Path(result.implementation.candidate_dir)
    assert json.loads((dest / "proposal.json").read_text(encoding="utf-8"))["name"] == "momentum_scratch"
    assert json.loads((dest / "critique.json").read_text(encoding="utf-8"))["verdict"] == "pass"
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_hash"] == result.candidate_hash
    assert manifest["risk_max_risk_per_trade"] == 0.01
    import aegis

    package = Path(aegis.__file__).resolve().parent
    assert package not in Path(result.implementation.code_path).resolve().parents

    events = _assert_audit(result, CANARY)
    assert any(event["event"] == "gate" and event["payload"].get("rejected_constraints") for event in events)
    researcher = [request for request in client.requests if request.agent == "researcher"]
    researcher_blob = json.dumps(researcher[0].messages)
    assert "10%" not in researcher_blob
    assert "Risk limits stay on the default" in researcher_blob
    assert CANARY not in json.dumps(
        [{"system": request.system, "messages": request.messages} for request in client.requests]
    )
    tool_followup = json.dumps(researcher[1].messages)
    assert "max_risk_per_trade" in tool_followup
    assert "mutable" in tool_followup
    implementer = [request for request in client.requests if request.agent == "implementer"]
    assert "SignalModel" in json.dumps(implementer[1].messages)
    workflow = Path("src/aegis/agents/workflow.py").read_text(encoding="utf-8")
    assert "backtest.engine" not in workflow
    assert "run_research_backtest" not in workflow
    assert "walk_forward_candidate" in workflow
    assert RiskLimits().max_risk_per_trade == 0.01
    assert not client.scripts["orchestrator"]
    assert not client.scripts["researcher"]
    assert not client.scripts["implementer"]


def test_unresolved_critical_finding_stops_before_backtest(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    client = ScriptedLLM(
        _scripts(
            _fail_critique(),
            AdvancementRequest(
                advance_to_backtest=True,
                override_risk_limits=True,
                override_validation=True,
                reason="Override the naysayer and loosen risk.",
            ),
        )
    )
    result = run_research(
        _settings(tmp_path),
        objective="Momentum on liquid names.",
        max_iterations=1,
        client=client,
    )
    assert result.implementation is not None
    assert result.implementation.status == "passed"
    assert result.advance_to_backtest is False
    assert result.backtest_started is False
    assert result.rejected_risk_override is True
    assert result.rejected_validation_override is True
    assert "Unresolved critical" in result.stop_reason
    assert "backtest not started" in result.stop_reason
    events = _assert_audit(result, CANARY)
    gate = [event for event in events if event["event"] == "gate" and "requested_risk_override" in event["payload"]][-1]
    assert gate["payload"]["requested_risk_override"] is True
    assert gate["payload"]["advance_to_backtest"] is False
    assert gate["payload"]["backtest_started"] is False
    assert RiskLimits() == RiskLimits(max_risk_per_trade=0.01)


def test_call_budget_stops_the_run(tmp_path):
    client = ScriptedLLM(
        {
            "orchestrator": [
                _submit(
                    ObjectiveAssignment(
                        objective="Momentum.",
                        success_criteria=["A written hypothesis."],
                        constraints=[],
                    ),
                    "only",
                )
            ]
        }
    )
    with pytest.raises(BudgetExceeded, match="Hard stop") as caught:
        run_research(
            _settings(tmp_path, research_max_llm_calls=1),
            objective="Momentum on liquid names.",
            max_iterations=1,
            client=client,
        )
    partial = caught.value.partial
    assert isinstance(partial, ResearchRunResult)
    assert partial.advance_to_backtest is False
    assert partial.backtest_started is False
    audit = Path(partial.audit_path).read_text(encoding="utf-8")
    assert '"event": "budget"' in audit
    assert len(client.requests) == 1


def test_cli_research_run_uses_the_workflow(monkeypatch, tmp_path):
    captured: dict = {}

    def fake(settings, *, objective, max_iterations, client=None, store_dir=None, audit_dir=None):
        captured["objective"] = objective
        captured["max_iterations"] = max_iterations
        return ResearchRunResult(
            run_id="cli-run",
            objective=objective,
            iterations=max_iterations,
            advance_to_backtest=False,
            backtest_started=False,
            stop_reason="stopped before backtest",
            audit_path=str(tmp_path / "audit.jsonl"),
            policy_implementation="stub",
        )

    monkeypatch.setattr("aegis.agents.workflow.run_research", fake)
    result = CliRunner().invoke(
        app,
        ["research", "run", "--objective", "Opening range on liquid names.", "--max-iterations", "3"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"objective": "Opening range on liquid names.", "max_iterations": 3}
    assert "backtest_started: false" in result.output
    assert "cli-run" in result.output
