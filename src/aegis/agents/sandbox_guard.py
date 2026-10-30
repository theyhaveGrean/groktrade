"""Runtime jail used only inside the sandbox subprocess.

Importing this module does not patch the caller. :func:`apply_runtime_guards`
is what installs the patches, and the sandbox child is the only process that
should call it.
"""

from __future__ import annotations

import builtins
import io
import os
import socket
from pathlib import Path

from aegis.agents.policy import runtime_import_allowed


class SandboxGuardError(PermissionError):
    """A plug-in tried to leave the scratch workspace or open the network."""


def apply_runtime_guards(*, scratch: Path, deny_read: list[Path]) -> None:
    """Block network, secret files, writes outside ``scratch``, and denied imports."""
    _block_network()
    _restrict_open(scratch.resolve(), [path.resolve() for path in deny_read])
    _install_import_guard()


def _block_network() -> None:
    def _blocked(*_args, **_kwargs):
        raise SandboxGuardError("network is disabled in the strategy sandbox")

    class _BlockedSocket(socket.socket):
        def __init__(self, *args, **kwargs) -> None:
            raise SandboxGuardError("network is disabled in the strategy sandbox")

    socket.socket = _BlockedSocket  # type: ignore[misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.create_server = _blocked  # type: ignore[assignment]


def _restrict_open(scratch: Path, deny_read: list[Path]) -> None:
    real_open = builtins.open
    real_io_open = io.open

    def guarded(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        path = Path(file).resolve()
        _assert_path(path, mode, scratch, deny_read)
        return real_open(file, mode, *args, **kwargs)

    def guarded_io(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_io_open(file, mode, *args, **kwargs)
        path = Path(file).resolve()
        _assert_path(path, mode, scratch, deny_read)
        return real_io_open(file, mode, *args, **kwargs)

    builtins.open = guarded
    io.open = guarded_io  # type: ignore[assignment]

    def _blocked_os_open(*_args, **_kwargs):
        raise SandboxGuardError("os.open is disabled in the strategy sandbox")

    os.open = _blocked_os_open  # type: ignore[assignment]


def _assert_path(path: Path, mode: str, scratch: Path, deny_read: list[Path]) -> None:
    writing = any(flag in mode for flag in ("w", "a", "x", "+"))
    if writing and not _is_under(path, scratch):
        raise SandboxGuardError(f"sandbox write blocked: {path}")
    if path.name == ".env" or path.name.startswith(".env."):
        raise SandboxGuardError("sandbox read blocked: env file")
    for blocked in deny_read:
        if path == blocked or _is_under(path, blocked):
            raise SandboxGuardError(f"sandbox read blocked: {path}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _install_import_guard() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and not runtime_import_allowed(name):
            raise SandboxGuardError(f"import of {name!r} is not allowed in the strategy sandbox")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
