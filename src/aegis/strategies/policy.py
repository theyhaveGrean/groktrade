"""Static check for signal modules.

The agent sandbox can call :func:`check_signal_source` on generated text
before import. Imports must sit on a short allowlist. Names that place or
cancel orders are rejected even if the import slipped through.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

INFRASTRUCTURE = {
    "__init__.py",
    "harness.py",
    "signal.py",
    "policy.py",
    "registry.py",
    "params.py",
    "resample.py",
    "plugin.py",
}

ALLOWED_MODULES = {
    "__future__",
    "math",
    "typing",
    "typing_extensions",
    "dataclasses",
    "datetime",
    "enum",
    "collections",
    "collections.abc",
    "zoneinfo",
    "pydantic",
    "aegis.strategies.params",
    "aegis.strategies.signal",
    "aegis.strategies.plugin",
    "aegis.strategies.registry",
    "aegis.strategies.resample",
}

FORBIDDEN_NAMES = {
    "submit_order",
    "submit_order_list",
    "order_factory",
    "cancel_order",
    "cancel_all_orders",
    "close_position",
    "close_all_positions",
    "place_order",
    "send_order",
    "modify_order",
    "nautilus_trader",
    "alpaca",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "open",
    "breakpoint",
}


class PolicyViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    line: int
    message: str


class PolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    files_checked: list[str] = Field(default_factory=list)
    violations: list[PolicyViolation] = Field(default_factory=list)


def check_signal_source(source: str, *, filename: str = "<signal>") -> PolicyResult:
    """Parse ``source`` and report allowlist or order-API violations."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        line = exc.lineno or 1
        return PolicyResult(
            passed=False,
            files_checked=[filename],
            violations=[PolicyViolation(path=filename, line=line, message=f"Syntax error: {exc.msg}")],
        )
    violations: list[PolicyViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, filename, node.lineno, violations)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(
                    PolicyViolation(
                        path=filename,
                        line=node.lineno or 1,
                        message="Relative imports are not allowed in a signal module.",
                    )
                )
                continue
            module = node.module or ""
            _check_module(module, filename, node.lineno, violations)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(
                PolicyViolation(
                    path=filename,
                    line=node.lineno or 1,
                    message=f"Signal modules cannot reference {node.id}.",
                )
            )
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            violations.append(
                PolicyViolation(
                    path=filename,
                    line=node.lineno or 1,
                    message=f"Signal modules cannot reference {node.attr}.",
                )
            )
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in FORBIDDEN_NAMES:
                violations.append(
                    PolicyViolation(
                        path=filename,
                        line=node.lineno or 1,
                        message=f"Signal modules cannot call {name}.",
                    )
                )
    return PolicyResult(passed=not violations, files_checked=[filename], violations=violations)


def check_signal_package(directory: Path | None = None) -> PolicyResult:
    """Check every non-infrastructure module under ``aegis.strategies``."""
    root = directory or Path(__file__).resolve().parent
    files = sorted(
        path for path in root.glob("*.py") if path.name not in INFRASTRUCTURE and not path.name.startswith("_")
    )
    violations: list[PolicyViolation] = []
    checked: list[str] = []
    for path in files:
        result = check_signal_source(path.read_text(encoding="utf-8"), filename=str(path))
        checked.extend(result.files_checked)
        violations.extend(result.violations)
    return PolicyResult(passed=not violations, files_checked=checked, violations=violations)


def _check_module(module: str, filename: str, line: int | None, violations: list[PolicyViolation]) -> None:
    if module in ALLOWED_MODULES:
        return
    violations.append(
        PolicyViolation(
            path=filename,
            line=line or 1,
            message=f"Import of {module!r} is outside the signal-module allowlist.",
        )
    )
