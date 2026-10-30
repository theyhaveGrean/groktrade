"""Parent side of the signal sandbox.

The implementer can only write ``candidate.py`` and ``test_candidate.py``
inside a scratch directory. The child process gets a scrubbed environment,
a timeout, resource limits, no network, and no write access outside scratch.
This module never imports the untrusted file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import aegis
from aegis.agents.models import SandboxResult
from aegis.agents.redaction import SECRET_ENV_NAMES
from aegis.errors import ResearchError
from aegis.holdout import DateRange

WRITABLE_NAMES = frozenset({"candidate.py", "test_candidate.py"})
_PACKAGE_ROOT = Path(aegis.__file__).resolve().parent


def scratch_file(scratch: Path, name: str) -> Path:
    """Resolve a file the implementer is allowed to write. Refuse everything else."""
    if name not in WRITABLE_NAMES or "/" in name or "\\" in name:
        raise ResearchError(
            f"Implementer can only write candidate.py or test_candidate.py, not {name!r}."
        )
    scratch_resolved = scratch.resolve()
    path = (scratch_resolved / name).resolve()
    if path.parent != scratch_resolved:
        raise ResearchError(f"Refusing to write outside the scratch workspace: {name}.")
    if path == _PACKAGE_ROOT or _PACKAGE_ROOT in path.parents:
        raise ResearchError(f"Refusing to write inside the aegis package: {path}.")
    protected = {
        _PACKAGE_ROOT / "strategies" / "harness.py",
        _PACKAGE_ROOT / "backtest" / "risk.py",
        _PACKAGE_ROOT / "holdout.py",
        _PACKAGE_ROOT / "config.py",
    }
    if path in protected or _is_under(path, _PACKAGE_ROOT / "data") or _is_under(path, _PACKAGE_ROOT / "backtest"):
        raise ResearchError(f"Refusing to write a protected file: {path}.")
    scratch_resolved.mkdir(parents=True, exist_ok=True)
    return path


def write_scratch(scratch: Path, name: str, source: str) -> Path:
    path = scratch_file(scratch, name)
    if len(source) > 60_000:
        raise ResearchError(f"{name} exceeds the 60,000 character scratch limit.")
    path.write_text(source, encoding="utf-8")
    return path


def run_sandbox(
    scratch: Path,
    *,
    holdout: DateRange,
    timeout_seconds: float,
    memory_mb: int,
    cache_dir: Path | None = None,
) -> SandboxResult:
    """Execute the scratch candidate. Policy failures do not count as a pass."""
    scratch.mkdir(parents=True, exist_ok=True)
    for name in WRITABLE_NAMES:
        if not (scratch / name).is_file():
            return SandboxResult(
                status="rejected",
                policy_ok=False,
                violations=[f"Missing {name} in the scratch workspace."],
                log=f"Missing {name}.",
            )
    config_path = scratch / "_sandbox_config.json"
    config_path.write_text(
        json.dumps(
            {
                "cpu_seconds": max(1, int(timeout_seconds)),
                "memory_mb": memory_mb,
                "holdout_start": holdout.start.isoformat(),
                "holdout_end": holdout.end.isoformat(),
                "deny_read": [str(path) for path in _deny_read(cache_dir)],
            }
        ),
        encoding="utf-8",
    )
    env = _child_env()
    _assert_clean(env)
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "aegis.agents.sandbox_main", "--scratch", str(scratch), "--config", str(config_path)],
            cwd=scratch,
            env=env,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            status="failed",
            policy_ok=True,
            timed_out=True,
            log=_clip(exc.stderr or exc.stdout or "Sandbox timed out."),
            failures=["timeout"],
        )
    payload = _parse_stdout(completed.stdout)
    if payload is None:
        killed = completed.returncode is not None and completed.returncode < 0
        return SandboxResult(
            status="failed",
            policy_ok=False,
            resource_killed=killed,
            timed_out=killed,
            log=_clip(completed.stderr or completed.stdout or "Sandbox produced no JSON result."),
            failures=["no-json"],
        )
    if completed.returncode is not None and completed.returncode < 0:
        payload["resource_killed"] = True
        payload["timed_out"] = True
        payload["status"] = "failed"
    try:
        return SandboxResult.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — child JSON is untrusted
        return SandboxResult(
            status="failed",
            policy_ok=False,
            log=f"Sandbox result was not valid: {exc}",
            failures=[str(exc)],
        )


def _child_env() -> dict[str, str]:
    src = str(_PACKAGE_ROOT.parent)
    # PYTHONPATH points at this checkout. Site-packages stay on the default
    # path so pydantic (and the rest of the interpreter's install) import.
    # The plug-in itself is still held to the AST allowlist and the import jail.
    # Secrets are not copied. PYTHONSAFEPATH keeps the scratch directory off
    # sys.path so a candidate cannot shadow a stdlib module by filename.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": src,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    for name in SECRET_ENV_NAMES:
        env.pop(name, None)
    return env


def _assert_clean(env: dict[str, str]) -> None:
    leaked = [name for name in SECRET_ENV_NAMES if env.get(name)]
    if leaked:
        raise ResearchError(
            "Refusing to launch the sandbox because its environment still has secret variables."
        )


def _deny_read(cache_dir: Path | None) -> list[Path]:
    paths = [
        _PACKAGE_ROOT / "config.py",
        _PACKAGE_ROOT / "holdout.py",
        _PACKAGE_ROOT / "data",
        _PACKAGE_ROOT.parent.parent / ".env",
    ]
    if cache_dir is not None:
        paths.append(cache_dir)
    return paths


def _parse_stdout(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "status" in payload:
            return payload
    return None


def _clip(text: str, limit: int = 4_000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, FileNotFoundError):
        return False
    return True
