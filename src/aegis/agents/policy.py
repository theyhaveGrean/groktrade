"""AST policy for agent-authored signal modules.

The check is import-allowlist plus a ban on order, risk, data, and holdout
APIs. It does not execute the module. The sandbox runs the same check in a
subprocess before it imports anything the agent wrote.

When ``aegis.strategies.harness.check_signal_source`` exists, callers merge
that report with this one. A candidate has to pass both.
"""

from __future__ import annotations

import ast
import re
from datetime import date

from pydantic import BaseModel, Field

from aegis.holdout import DateRange

SEARCH_META = "x-aegis-search"

ALLOWED_IMPORT_MODULES = frozenset(
    {
        "math",
        "statistics",
        "dataclasses",
        "typing",
        "enum",
        "datetime",
        "collections",
        "abc",
        "functools",
        "itertools",
        "operator",
        "decimal",
        "numbers",
        "copy",
        "re",
    }
)

ALLOWED_FROM: dict[str, frozenset[str]] = {
    "datetime": frozenset({"date", "datetime", "time", "timedelta", "timezone"}),
    "typing": frozenset(
        {"Any", "Literal", "Optional", "Iterable", "Sequence", "Mapping", "NamedTuple"}
    ),
    "collections": frozenset({"deque", "Counter"}),
    "dataclasses": frozenset({"dataclass", "field"}),
    "enum": frozenset({"Enum"}),
    "pydantic": frozenset({"BaseModel", "Field", "ConfigDict", "field_validator", "model_validator"}),
    "aegis.agents.signal_contract": frozenset({"BarView", "Signal", "SignalModel", "SearchSpace"}),
    "aegis.strategies.harness": frozenset({"BarView", "Signal", "SignalModel", "SearchSpace"}),
    "__future__": frozenset({"annotations"}),
}

# Runtime denylist. The AST allowlist is the strict gate. This list is defense
# in depth for modules Python has already imported (``sys.modules`` would
# otherwise satisfy ``import os`` without asking the AST). Interpreter and
# pydantic internals stay importable so the sandbox can boot.
RUNTIME_DENY_PREFIXES = (
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "http",
    "urllib",
    "requests",
    "httpx",
    "nautilus_trader",
    "ctypes",
    "pickle",
    "multiprocessing",
    "shutil",
    "runpy",
    "pty",
    "fcntl",
    "importlib",
    "aegis.config",
    "aegis.data",
    "aegis.holdout",
    "aegis.backtest",
    "aegis.cli",
    "aegis.experiments",
    "aegis.execution",
    "aegis.ml",
    "aegis.agents.llm",
    "aegis.agents.workflow",
    "aegis.agents.sandbox",
    "aegis.agents.tools",
    "aegis.agents.runner",
)

FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "importlib",
        "ctypes",
        "pickle",
        "multiprocessing",
        "shutil",
        "http",
        "urllib",
        "requests",
        "httpx",
        "nautilus_trader",
        "builtins",
        "runpy",
        "code",
        "codeop",
        "pty",
        "fcntl",
        "resource",
        "signal",
        "threading",
        "asyncio",
        "aegis.config",
        "aegis.data",
        "aegis.holdout",
        "aegis.backtest",
        "aegis.cli",
        "aegis.experiments",
        "aegis.execution",
        "aegis.ml",
    }
)

FORBIDDEN_NAMES = frozenset(
    {
        "submit_order",
        "submit_order_list",
        "order_factory",
        "close_all_positions",
        "cancel_all_orders",
        "place_order",
        "send_order",
        "bracket",
        "RiskLimits",
        "install_risk_engine",
        "clear_risk_engine",
        "get_risk_engine",
        "read_bars",
        "BarCache",
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "socket",
        "subprocess",
        "Popen",
        "urlopen",
        "system",
        "popen",
        "OPENAI_API_KEY",
        "MASSIVE_API_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    }
)

RISK_PARAM_NAMES = frozenset(
    {
        "max_risk_per_trade",
        "max_positions",
        "max_position_equity_fraction",
        "daily_loss_cutoff",
        "reward_to_risk",
        "atr_multiplier",
        "kill_switch",
        "quantity",
        "stop_price",
        "target_price",
    }
)

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


class PolicyViolation(BaseModel):
    code: str
    message: str
    line: int | None = None


class PolicyReport(BaseModel):
    ok: bool
    violations: list[PolicyViolation] = Field(default_factory=list)
    implementation: str = "stub"


def runtime_import_allowed(name: str) -> bool:
    """Whether the sandbox may import ``name`` while executing a plug-in.

    Dangerous modules are denied even when they are already in ``sys.modules``.
    Other imports are allowed so pydantic and the interpreter can finish
    booting. The AST allowlist is what restricts the plug-in source itself.
    """
    if name == "aegis.agents.signal_contract" or name == "aegis.strategies.harness":
        return True
    for prefix in RUNTIME_DENY_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return False
    if name.startswith("aegis.agents.") and name != "aegis.agents.signal_contract":
        return False
    if name.startswith("aegis.strategies.") and name not in {
        "aegis.strategies.harness",
    }:
        return False
    return True


def check_candidate(
    signal_source: str,
    test_source: str,
    *,
    holdout: DateRange | None = None,
) -> PolicyReport:
    """Policy for the pair of files the implementer is allowed to write."""
    signal = check_signal_source(signal_source, filename="candidate.py", holdout=holdout, kind="signal")
    tests = check_signal_source(
        test_source,
        filename="test_candidate.py",
        holdout=holdout,
        kind="test",
    )
    violations = [*signal.violations, *tests.violations]
    return PolicyReport(ok=not violations, violations=violations, implementation="stub")


def check_signal_source(
    source: str,
    *,
    filename: str = "candidate.py",
    holdout: DateRange | None = None,
    kind: str = "signal",
) -> PolicyReport:
    """Parse ``source`` and report policy violations. Never executes it."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return PolicyReport(
            ok=False,
            violations=[
                PolicyViolation(code="syntax", message=f"{filename}: {exc.msg}", line=exc.lineno)
            ],
        )
    visitor = _PolicyVisitor(filename=filename, holdout=holdout, kind=kind)
    visitor.visit(tree)
    if kind == "signal":
        visitor.finish_signal(tree)
    else:
        visitor.finish_tests(tree)
    return PolicyReport(ok=not visitor.violations, violations=list(visitor.violations))


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, *, filename: str, holdout: DateRange | None, kind: str) -> None:
        self.filename = filename
        self.holdout = holdout
        self.kind = kind
        self.violations: list[PolicyViolation] = []

    def finish_signal(self, tree: ast.AST) -> None:
        params = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and _bases_named(node, "BaseModel") and node.name == "Params"
        ]
        signals = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and _bases_named(node, "SignalModel")
        ]
        if len(params) != 1:
            self._add("params", f"{self.filename} must define exactly one Params(BaseModel) class.")
        if len(signals) != 1:
            self._add("signal_model", f"{self.filename} must define exactly one SignalModel subclass.")
        for node in params:
            self._check_params(node)
        for node in signals:
            methods = {stmt.name for stmt in node.body if isinstance(stmt, ast.FunctionDef)}
            if "on_bar" not in methods:
                self._add("signal_model", f"{node.name} must define on_bar.", line=node.lineno)

    def finish_tests(self, tree: ast.AST) -> None:
        tests = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        if not tests:
            self._add("tests", f"{self.filename} must define at least one test_* function.")

    def _check_params(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            names: list[str] = []
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names.append(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
            for name in names:
                if name in RISK_PARAM_NAMES:
                    self._add(
                        "risk_param",
                        f"Params.{name} is a risk-engine field. Plug-ins cannot carry risk limits.",
                        line=stmt.lineno,
                    )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self.kind == "test" and alias.name == "candidate":
                continue
            if alias.name not in ALLOWED_IMPORT_MODULES or _module_forbidden(alias.name):
                self._add("import", f"Import of {alias.name!r} is not allowed.", line=node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._add("import", "Relative imports are not allowed.", line=node.lineno)
            return
        module = node.module or ""
        allowed_names = ALLOWED_FROM.get(module)
        if allowed_names is None or _module_forbidden(module):
            self._add("import", f"Import from {module!r} is not allowed.", line=node.lineno)
            return
        for alias in node.names:
            if alias.name == "*":
                self._add("import", "Star imports are not allowed.", line=node.lineno)
            elif alias.name not in allowed_names:
                self._add(
                    "import",
                    f"{module}.{alias.name} is not on the signal-module allowlist.",
                    line=node.lineno,
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._add("forbidden_api", f"Name {node.id!r} is not allowed in a signal module.", line=node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_NAMES or node.attr in {"environ", "read_bars"}:
            self._add(
                "forbidden_api",
                f"Attribute {node.attr!r} is not allowed in a signal module.",
                line=node.lineno,
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._scan_text(node.value, node.lineno)
        self.generic_visit(node)

    def _scan_text(self, text: str, line: int | None) -> None:
        if self.holdout is None:
            return
        for match in _ISO_DATE.finditer(text):
            try:
                day = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            if self.holdout.contains(day):
                self._add(
                    "holdout",
                    f"Source mentions holdout date {day.isoformat()}. Agents cannot read that window.",
                    line=line,
                )

    def _add(self, code: str, message: str, line: int | None = None) -> None:
        self.violations.append(PolicyViolation(code=code, message=message, line=line))


def _bases_named(node: ast.ClassDef, name: str) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == name:
            return True
        if isinstance(base, ast.Attribute) and base.attr == name:
            return True
    return False


def _module_forbidden(module: str) -> bool:
    if module in FORBIDDEN_MODULES:
        return True
    for banned in FORBIDDEN_MODULES:
        if module == banned or module.startswith(banned + "."):
            return True
    return False
