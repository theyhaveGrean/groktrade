"""Child entrypoint. The parent launches this with a scrubbed environment.

Do not import the workflow, the OpenAI client, or settings here. Settings would
read ``.env`` and pull secrets into a process that executes untrusted code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import traceback
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path

from aegis.agents.policy import RISK_PARAM_NAMES, SEARCH_META
from aegis.agents.signal_contract import check_pair, is_signal_model
from aegis.holdout import DateRange


def _apply_limits(cpu_seconds: int, memory_mb: int) -> None:
    cpu = max(1, int(cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    memory_bytes = max(64, int(memory_mb)) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000, 2_000_000))
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ValueError, OSError):
        pass


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _secrets_present() -> bool:
    import os

    names = ("OPENAI_API_KEY", "MASSIVE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    return any(os.environ.get(name) for name in names)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inspect_signal(module) -> list[str]:
    from pydantic import BaseModel

    problems: list[str] = []
    params_cls = getattr(module, "Params", None)
    if not isinstance(params_cls, type) or not issubclass(params_cls, BaseModel):
        problems.append("Params must be a pydantic BaseModel.")
        return problems
    for name, field in params_cls.model_fields.items():
        if name in RISK_PARAM_NAMES:
            problems.append(f"Params.{name} collides with a risk-engine field.")
        extra = field.json_schema_extra
        if isinstance(extra, dict) and SEARCH_META in extra:
            space = extra[SEARCH_META]
            if not isinstance(space, dict):
                problems.append(f"Params.{name} search metadata is not an object.")
                continue
            low = space.get("low")
            high = space.get("high")
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not low < high:
                problems.append(f"Params.{name} search bounds must satisfy low < high.")
    found = [value for value in vars(module).values() if is_signal_model(value)]
    if len(found) != 1:
        problems.append(f"Expected one SignalModel subclass, found {len(found)}.")
        return problems
    try:
        instance = found[0]()
        instance.reset()
    except Exception as exc:  # noqa: BLE001 — untrusted constructor
        problems.append(f"Signal model failed to construct: {type(exc).__name__}: {exc}")
    return problems


def _risk_probe() -> tuple[bool, int | None, float | None, str | None]:
    """Prove the default risk engine is what would size an entry. The plug-in is not asked."""
    from zoneinfo import ZoneInfo

    from aegis.backtest.risk import PortfolioSnapshot, RiskEngine, RiskLimits

    engine = RiskEngine()
    limits = RiskLimits()
    if engine.limits != limits:
        return False, None, None, "RiskEngine was not constructed with default limits."
    ny = ZoneInfo("America/New_York")
    decision = engine.evaluate_entry(
        symbol="AAPL",
        ts=datetime(2024, 1, 3, 10, 0, tzinfo=ny),
        side="long",
        entry=100.0,
        atr=1.0,
        snapshot=PortfolioSnapshot(equity=100_000.0, day_start_equity=100_000.0),
    )
    if not decision.approved or decision.quantity < 1:
        return False, decision.quantity, limits.max_risk_per_trade, decision.reason
    return True, decision.quantity, limits.max_risk_per_trade, None


def _run_tests(module) -> tuple[int, int, list[str]]:
    names = sorted(
        name for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))
    )
    failures: list[str] = []
    passed = 0
    for name in names:
        try:
            getattr(module, name)()
        except Exception as exc:  # noqa: BLE001 — report untrusted test failures
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
    return len(names), passed, failures


def _signal_class_name(module) -> str | None:
    found = [value for value in vars(module).values() if is_signal_model(value)]
    if len(found) == 1:
        return found[0].__name__
    return None


def run(scratch: Path, config: dict) -> dict:
    signal_path = scratch / "candidate.py"
    test_path = scratch / "test_candidate.py"
    holdout = DateRange(
        date_cls.fromisoformat(config["holdout_start"]),
        date_cls.fromisoformat(config["holdout_end"]),
    )
    signal_source = signal_path.read_text(encoding="utf-8")
    test_source = test_path.read_text(encoding="utf-8")
    report = check_pair(signal_source, test_source, holdout=holdout)
    base = {
        "policy_ok": report.ok,
        "violations": [item.message for item in report.violations],
        "tests_run": 0,
        "tests_passed": 0,
        "failures": [],
        "timed_out": False,
        "resource_killed": False,
        "risk_applied": False,
        "risk_quantity": None,
        "risk_max_risk_per_trade": None,
        "secrets_in_env": _secrets_present(),
        "signal_class": None,
        "log": "",
    }
    if base["secrets_in_env"]:
        base["status"] = "failed"
        base["log"] = "Sandbox environment contained a secret. Refusing to execute."
        return base
    if not report.ok:
        base["status"] = "rejected"
        base["log"] = "Policy check rejected the candidate before execution."
        return base

    # Probe the risk engine before the import jail. The plug-in never gets to
    # import aegis.backtest, so it cannot swap this engine out.
    applied, quantity, max_risk, probe_error = _risk_probe()
    base["risk_applied"] = applied
    base["risk_quantity"] = quantity
    base["risk_max_risk_per_trade"] = max_risk
    if probe_error or not applied:
        base["status"] = "failed"
        base["log"] = probe_error or "Risk engine probe did not approve the reference entry."
        return base

    from aegis.agents.sandbox_guard import apply_runtime_guards

    deny = [Path(item) for item in config.get("deny_read", [])]
    apply_runtime_guards(scratch=scratch, deny_read=deny)
    try:
        module = _load_module(signal_path, "candidate")
        problems = _inspect_signal(module)
        base["signal_class"] = _signal_class_name(module)
        if problems:
            base["status"] = "rejected"
            base["policy_ok"] = False
            base["violations"] = [*base["violations"], *problems]
            base["log"] = "Loaded module failed the signal-model inspection."
            return base
        tests = _load_module(test_path, "test_candidate")
        total, passed, failures = _run_tests(tests)
        base["tests_run"] = total
        base["tests_passed"] = passed
        base["failures"] = failures
        if total < 1 or failures or passed != total:
            base["status"] = "failed"
            base["log"] = "Sandbox tests failed." if failures else "Sandbox ran zero tests."
            return base
        base["status"] = "passed"
        base["log"] = f"{passed} test(s) passed. Risk engine quantity on the probe was {quantity}."
        return base
    except Exception as exc:  # noqa: BLE001 — untrusted code
        base["status"] = "failed"
        base["log"] = f"{type(exc).__name__}: {exc}"
        base["failures"] = [base["log"]]
        return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one signal candidate inside the sandbox.")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    _apply_limits(int(config["cpu_seconds"]), int(config["memory_mb"]))
    try:
        payload = run(Path(args.scratch), config)
    except Exception as exc:  # noqa: BLE001 — always emit JSON for the parent
        payload = {
            "status": "failed",
            "policy_ok": False,
            "violations": [],
            "tests_run": 0,
            "tests_passed": 0,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "log": traceback.format_exc(limit=4),
            "timed_out": False,
            "resource_killed": False,
            "risk_applied": False,
            "risk_quantity": None,
            "risk_max_risk_per_trade": None,
            "secrets_in_env": False,
            "signal_class": None,
        }
    _emit(payload)
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
