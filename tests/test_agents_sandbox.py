"""Sandbox, policy, and candidate storage. No model calls."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from aegis.agents.candidates import content_hash, store_candidate
from aegis.agents.models import Critique, Finding, Rule, SandboxResult, StrategyProposal
from aegis.agents.sandbox import run_sandbox, scratch_file, write_scratch
from aegis.agents.signal_contract import (
    EXAMPLE_SIGNAL_SOURCE,
    EXAMPLE_TEST_SOURCE,
    Signal,
    SignalModel,
    check_pair,
    check_signal_source,
    get_signal,
    policy_implementation,
    register_signal,
)
from aegis.agents.signal_contract import REGISTRY
from aegis.backtest.risk import PortfolioSnapshot, RiskEngine, RiskLimits
from aegis.errors import ResearchError
from aegis.holdout import DateRange

HOLD = DateRange(date(2026, 7, 1), date(2026, 10, 31))


def _proposal() -> StrategyProposal:
    return StrategyProposal(
        name="momentum_scratch",
        summary="Long when the close breaks a short lookback.",
        entry_rules=[Rule(name="break", description="Close above the close lookback bars ago.")],
        exit_rules=[Rule(name="next", description="Exit on the next bar. The risk engine owns the stop.")],
        market_assumptions=["Liquid names in the regular session."],
        falsifiable_hypothesis="A flat tape produces no sustained edge for this break.",
        indicators=["close lookback"],
        failure_conditions=["Entries during a flat close series."],
    )


def _critique() -> Critique:
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
        stress_tests=["Replay a flat series."],
        summary="No critical issue in the spec.",
    )


def _run(scratch: Path, signal: str, tests: str, *, timeout: float = 20) -> SandboxResult:
    write_scratch(scratch, "candidate.py", signal)
    write_scratch(scratch, "test_candidate.py", tests)
    return run_sandbox(
        scratch,
        holdout=HOLD,
        timeout_seconds=timeout,
        memory_mb=256,
        cache_dir=scratch / "cache",
    )


def test_example_module_passes_policy_and_sandbox(tmp_path):
    report = check_pair(EXAMPLE_SIGNAL_SOURCE, EXAMPLE_TEST_SOURCE, holdout=HOLD)
    assert report.ok
    assert policy_implementation() in {"stub", "aegis.strategies.harness"}
    result = _run(tmp_path / "scratch", EXAMPLE_SIGNAL_SOURCE, EXAMPLE_TEST_SOURCE)
    assert result.status == "passed"
    assert result.policy_ok
    assert result.tests_run == 2
    assert result.tests_passed == 2
    assert result.risk_applied
    assert result.secrets_in_env is False
    assert result.signal_class == "MomentumSignal"
    probe = RiskEngine().evaluate_entry(
        symbol="AAPL",
        ts=datetime(2024, 1, 3, 10, 0, tzinfo=ZoneInfo("America/New_York")),
        side="long",
        entry=100.0,
        atr=1.0,
        snapshot=PortfolioSnapshot(equity=100_000.0, day_start_equity=100_000.0),
    )
    assert result.risk_quantity == probe.quantity
    assert result.risk_max_risk_per_trade == RiskLimits().max_risk_per_trade == 0.01


def test_policy_rejects_orders_network_risk_and_holdout():
    order = EXAMPLE_SIGNAL_SOURCE.replace(
        'return Signal(action="hold", reason="no setup")',
        'self.submit_order()\n        return Signal(action="hold", reason="no setup")',
    )
    network = "import socket\n" + EXAMPLE_SIGNAL_SOURCE
    risk = EXAMPLE_SIGNAL_SOURCE.replace(
        "class Params(BaseModel):",
        "class Params(BaseModel):\n    max_risk_per_trade: float = 0.5",
    )
    holdout = EXAMPLE_SIGNAL_SOURCE.replace(
        "warming up",
        "warming up on 2026-07-04",
    )
    data = EXAMPLE_SIGNAL_SOURCE.replace(
        "from pydantic import BaseModel, Field",
        "from aegis.data.cache import BarCache\nfrom pydantic import BaseModel, Field",
    )
    for source, code in (
        (order, "forbidden_api"),
        (network, "import"),
        (risk, "risk_param"),
        (holdout, "holdout"),
        (data, "import"),
    ):
        report = check_signal_source(source, holdout=HOLD, kind="signal")
        assert not report.ok, source[:80]
        assert any(item.code == code for item in report.violations)

    names = EXAMPLE_SIGNAL_SOURCE.replace("self._closes: list[float] = []", "RiskLimits()")
    report = check_signal_source(names, holdout=HOLD, kind="signal")
    assert any(item.code == "forbidden_api" for item in report.violations)


def test_sandbox_rejects_order_api_without_running_tests(tmp_path):
    source = '''from pydantic import BaseModel, Field

from aegis.agents.signal_contract import BarView, Signal, SignalModel


class Params(BaseModel):
    lookback: int = Field(2, json_schema_extra={"x-aegis-search": {"low": 2, "high": 5}})


class BadSignal(SignalModel):
    name = "bad_signal"

    def __init__(self, params: Params | None = None) -> None:
        self.params = params or Params()

    def on_bar(self, bar: BarView, *, position_open: bool) -> Signal:
        self.submit_order()
        return Signal(action="hold", reason="should be rejected first")
'''
    result = _run(tmp_path / "scratch", source, EXAMPLE_TEST_SOURCE)
    assert result.status == "rejected"
    assert result.tests_run == 0
    assert result.policy_ok is False


def test_sandbox_times_out_on_an_infinite_test(tmp_path):
    looping = "def test_loops():\n    while True:\n        pass\n"
    result = _run(tmp_path / "scratch", EXAMPLE_SIGNAL_SOURCE, looping, timeout=2)
    assert result.status == "failed"
    assert result.timed_out or result.resource_killed


def test_scratch_writes_stay_out_of_the_package(tmp_path):
    with pytest.raises(ResearchError):
        scratch_file(tmp_path, "harness.py")
    with pytest.raises(ResearchError):
        scratch_file(tmp_path, "../candidate.py")
    package_scratch = Path("src/aegis/strategies")
    with pytest.raises(ResearchError):
        scratch_file(package_scratch, "candidate.py")
    assert not package_scratch.exists()
    path = write_scratch(tmp_path, "candidate.py", EXAMPLE_SIGNAL_SOURCE)
    assert path.parent == tmp_path.resolve()


def test_candidate_is_stored_with_proposal_and_critique(tmp_path):
    sandbox = SandboxResult(
        status="passed",
        policy_ok=True,
        tests_run=2,
        tests_passed=2,
        risk_applied=True,
        risk_quantity=250,
        risk_max_risk_per_trade=0.01,
        signal_class="MomentumSignal",
        log="ok",
    )
    record = store_candidate(
        store_dir=tmp_path,
        run_id="run",
        strategy_name="momentum_scratch",
        signal_source=EXAMPLE_SIGNAL_SOURCE,
        test_source=EXAMPLE_TEST_SOURCE,
        proposal=_proposal(),
        critique=_critique(),
        sandbox=sandbox,
        policy_implementation="stub",
        prompt_version="v1",
        notes="",
    )
    dest = Path(record.candidate_dir)
    assert record.content_hash == content_hash(EXAMPLE_SIGNAL_SOURCE)
    assert (dest / "candidate.py").is_file()
    assert (dest / "test_candidate.py").is_file()
    proposal = json.loads((dest / "proposal.json").read_text(encoding="utf-8"))
    critique = json.loads((dest / "critique.json").read_text(encoding="utf-8"))
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert proposal["falsifiable_hypothesis"]
    assert proposal["entry_rules"] and proposal["exit_rules"] and proposal["market_assumptions"]
    assert critique["verdict"] == "pass"
    assert manifest["content_hash"] == record.content_hash
    assert manifest["files"]["proposal"] == "proposal.json"
    assert manifest["files"]["critique"] == "critique.json"


def test_registry_stub_round_trip():
    class Tiny(SignalModel):
        name = "tiny_stub"

        def on_bar(self, bar, *, position_open: bool) -> Signal:
            return Signal(action="hold", reason="noop")

    register_signal(Tiny)
    try:
        assert get_signal("tiny_stub") is Tiny
    finally:
        REGISTRY.clear()


def test_runtime_guard_blocks_network_env_and_escape(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-canary-PHASE3-DO-NOT-LEAK\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    scratch = tmp_path / "scratch"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from aegis.agents.sandbox_guard import apply_runtime_guards",
                f"scratch = Path({str(scratch)!r})",
                "scratch.mkdir(parents=True, exist_ok=True)",
                f"apply_runtime_guards(scratch=scratch, deny_read=[Path({str(env_file)!r})])",
                "results = []",
                "try:",
                "    import socket",
                "    socket.socket()",
                "    results.append('socket-open')",
                "except Exception:",
                "    results.append('socket-blocked')",
                "try:",
                f"    open({str(env_file)!r})",
                "    results.append('env-open')",
                "except Exception:",
                "    results.append('env-blocked')",
                "try:",
                f"    open({str(outside)!r}, 'w')",
                "    results.append('write-open')",
                "except Exception:",
                "    results.append('write-blocked')",
                "print(','.join(results))",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(**{key: value for key, value in __import__("os").environ.items() if key not in {"OPENAI_API_KEY", "MASSIVE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"}})
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "socket-blocked,env-blocked,write-blocked"
    assert not outside.exists()
