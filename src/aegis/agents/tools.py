"""Tools each agent is allowed to call.

Handlers receive a :class:`ToolContext`. That object has paths and the
holdout window. It does not carry :class:`aegis.config.Settings`, so a tool
cannot read ``openai_api_key`` off it. A future tool that needs a key reads
the environment inside the handler and must not put the value in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from aegis.agents.models import SandboxResult
from aegis.agents.sandbox import run_sandbox, write_scratch
from aegis.agents.signal_contract import EXAMPLE_SIGNAL_SOURCE, EXAMPLE_TEST_SOURCE, check_signal_source
from aegis.backtest.risk import RiskLimits
from aegis.data.universe import load_universe, resolve_universe_path, select_as_of
from aegis.errors import ResearchError
from aegis.holdout import DateRange, assert_no_holdout_overlap

_NONE = {"type": "object", "properties": {}, "additionalProperties": False}


@dataclass
class RunState:
    objective: str = ""
    iteration: int = 0
    max_iterations: int = 1
    proposal_json: str | None = None
    critique_json: str | None = None
    unresolved_critical: list[str] = field(default_factory=list)
    candidate_hash: str | None = None
    budget: dict[str, float | int] = field(default_factory=dict)


@dataclass
class ToolContext:
    scratch: Path
    universe_path: Path
    holdout: DateRange
    cache_dir: Path
    state: RunState
    timeout_seconds: float
    memory_mb: int


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    agents: frozenset[str]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tools_for(agent: str) -> list[Tool]:
    return [tool for tool in TOOLS if agent in tool.agents]


def describe_risk_limits(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    limits = RiskLimits()
    return {
        "max_risk_per_trade": limits.max_risk_per_trade,
        "max_positions": limits.max_positions,
        "max_position_equity_fraction": limits.max_position_equity_fraction,
        "daily_loss_cutoff": limits.daily_loss_cutoff,
        "reward_to_risk": limits.reward_to_risk,
        "atr_multiplier": limits.atr_multiplier,
        "entry_start": limits.entry_start.isoformat(timespec="minutes"),
        "entry_end": limits.entry_end.isoformat(timespec="minutes"),
        "flatten_time": limits.flatten_time.isoformat(timespec="minutes"),
        "kill_switch": limits.kill_switch,
        "mutable": False,
        "note": "These limits are fixed in code. Agents cannot change them.",
    }


def list_universe(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Symbol names and listing windows. This does not read bars."""
    as_of = ctx.holdout.start - timedelta(days=1)
    path = resolve_universe_path(ctx.universe_path)
    members = select_as_of(load_universe(path), as_of)
    return {
        "as_of": as_of.isoformat(),
        "symbols": [
            {"symbol": member.symbol, "start": member.start.isoformat(), "end": None if member.end is None else member.end.isoformat()}
            for member in members
        ],
    }


def check_research_window(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Refuse a window that overlaps the holdout. Returns no bars either way."""
    from datetime import date

    start = date.fromisoformat(str(args["start"]))
    end = date.fromisoformat(str(args["end"]))
    assert_no_holdout_overlap(DateRange(start, end), ctx.holdout, purpose="agent research")
    return {"allowed": True, "start": start.isoformat(), "end": end.isoformat()}


def read_run_state(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    state = ctx.state
    return {
        "objective": state.objective,
        "iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "unresolved_critical": list(state.unresolved_critical),
        "candidate_hash": state.candidate_hash,
        "budget": dict(state.budget),
        "note": "Risk limits and failed validations cannot be overridden. Backtests are not started in this phase.",
    }


def get_signal_api(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "signal_source": EXAMPLE_SIGNAL_SOURCE,
        "test_source": EXAMPLE_TEST_SOURCE,
        "writable_files": ["candidate.py", "test_candidate.py"],
        "base_class": "aegis.strategies.signal.SignalModel",
        "params_base": "aegis.strategies.params.StrategyParams",
        "policy": "aegis.strategies.harness.check_signal_source",
        "registry": "aegis.strategies.harness.register_signal",
        "read_only": [
            "aegis/strategies/harness.py",
            "aegis/backtest/risk.py",
            "aegis/data/",
            "aegis/holdout.py",
        ],
        "note": "Copy the shape, not a strategy that submits orders. Params search metadata uses x-aegis-search.",
    }


def write_signal(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    _only_plugin_file(str(args.get("path") or "candidate.py"), "candidate.py")
    path = write_scratch(ctx.scratch, "candidate.py", str(args["source"]))
    return {"path": path.name, "bytes": path.stat().st_size}


def write_tests(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    _only_plugin_file(str(args.get("path") or "test_candidate.py"), "test_candidate.py")
    path = write_scratch(ctx.scratch, "test_candidate.py", str(args["source"]))
    return {"path": path.name, "bytes": path.stat().st_size}


def _only_plugin_file(requested: str, allowed: str) -> None:
    """The implementer may write one scratch plug-in file, never the platform."""
    normalized = requested.replace("\\", "/").strip()
    if normalized != allowed or "/" in normalized or normalized.startswith("."):
        raise ResearchError(
            "Implementer can only write candidate.py or test_candidate.py in the scratch workspace, "
            f"not {requested!r}. The risk engine, harness, data package, and holdout are read-only."
        )


def run_sandbox_tool(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    result = run_sandbox(
        ctx.scratch,
        holdout=ctx.holdout,
        timeout_seconds=ctx.timeout_seconds,
        memory_mb=ctx.memory_mb,
        cache_dir=ctx.cache_dir,
    )
    return _sandbox_payload(result)


def read_proposal(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if not ctx.state.proposal_json:
        raise ResearchError("No proposal has been recorded yet.")
    return {"proposal": ctx.state.proposal_json}


def read_signal_source(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    path = ctx.scratch / "candidate.py"
    if not path.is_file():
        raise ResearchError("No candidate.py is in the scratch workspace.")
    return {"source": path.read_text(encoding="utf-8")}


def scan_signal(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Static policy scan. Does not execute the module and does not read bars."""
    source = args.get("source")
    if not source:
        path = ctx.scratch / "candidate.py"
        if not path.is_file():
            raise ResearchError("No candidate.py is in the scratch workspace.")
        source = path.read_text(encoding="utf-8")
    report = check_signal_source(str(source), filename="candidate.py", holdout=ctx.holdout, kind="signal")
    return {
        "ok": report.ok,
        "implementation": report.implementation,
        "violations": [item.message for item in report.violations],
    }


def _sandbox_payload(result: SandboxResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "policy_ok": result.policy_ok,
        "tests_run": result.tests_run,
        "tests_passed": result.tests_passed,
        "violations": result.violations,
        "failures": result.failures,
        "risk_applied": result.risk_applied,
        "log": result.log,
    }


_SOURCE = {
    "type": "object",
    "properties": {"source": {"type": "string"}},
    "required": ["source"],
    "additionalProperties": False,
}

_WINDOW = {
    "type": "object",
    "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
    "required": ["start", "end"],
    "additionalProperties": False,
}

_SCAN = {
    "type": "object",
    "properties": {"source": {"type": "string"}},
    "additionalProperties": False,
}


TOOLS: tuple[Tool, ...] = (
    Tool(
        "describe_risk_limits",
        "Read the fixed risk limits. They cannot be changed.",
        _NONE,
        frozenset({"researcher", "implementer", "naysayer"}),
        describe_risk_limits,
    ),
    Tool(
        "list_universe",
        "List universe symbols and listing windows. Does not return prices or bars.",
        _NONE,
        frozenset({"researcher"}),
        list_universe,
    ),
    Tool(
        "check_research_window",
        "Check whether a date window is outside the holdout. Returns no market data.",
        _WINDOW,
        frozenset({"researcher"}),
        check_research_window,
    ),
    Tool(
        "read_run_state",
        "Read the objective, iteration, unresolved critique, and remaining budget.",
        _NONE,
        frozenset({"orchestrator"}),
        read_run_state,
    ),
    Tool(
        "get_signal_api",
        "Return the signal-module template, the two writable filenames, and the read-only paths.",
        _NONE,
        frozenset({"implementer"}),
        get_signal_api,
    ),
    Tool(
        "write_signal",
        "Write candidate.py in the scratch workspace.",
        _SOURCE,
        frozenset({"implementer"}),
        write_signal,
    ),
    Tool(
        "write_tests",
        "Write test_candidate.py in the scratch workspace.",
        _SOURCE,
        frozenset({"implementer"}),
        write_tests,
    ),
    Tool(
        "run_sandbox",
        "Policy-check and test the scratch files. The workflow runs this again on the submitted source.",
        _NONE,
        frozenset({"implementer"}),
        run_sandbox_tool,
    ),
    Tool(
        "read_proposal",
        "Read the current strategy proposal.",
        _NONE,
        frozenset({"naysayer"}),
        read_proposal,
    ),
    Tool(
        "read_signal_source",
        "Read candidate.py from the scratch workspace if the implementer has written it.",
        _NONE,
        frozenset({"naysayer"}),
        read_signal_source,
    ),
    Tool(
        "scan_signal",
        "Run the AST policy check on signal source. Does not execute it.",
        _SCAN,
        frozenset({"naysayer"}),
        scan_signal,
    ),
)
