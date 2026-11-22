"""Process-level recovery helpers for fatal simulator failures.

This module intentionally depends only on the Python standard library.  A
fatal PhysX/CUDA failure can leave Kit's process state unsafe to tear down, so
the recovery path persists the evaluation cursor and replaces the process
without first closing ``SimulationApp``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
import sys
from typing import Any, NoReturn

MAX_INPROC_RESTARTS = 3
RESTART_COUNT_ENV = "ROBODOJO_FATAL_RESTART_COUNT"
RUN_ID_ENV = "ROBODOJO_RUN_ID"


def _restart_count() -> int:
    """Read the bounded restart counter, treating malformed values as zero."""

    try:
        return int(os.environ.get(RESTART_COUNT_ENV, "0"))
    except (TypeError, ValueError):
        return 0


def _persist_manifest(env: Any, restart_count: int) -> None:
    """Persist progress without allowing a persistence error to block recovery."""

    persist = getattr(env, "persist_resume_manifest", None)
    if not callable(persist):
        return
    try:
        persist(restart_count=restart_count)
    except Exception as exc:
        print(f"[FATAL] persist_resume_manifest failed: {exc}", flush=True)


def _flush_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _restore_monitored_stdio() -> None:
    """Restore original stdout/stderr fds if PhysXWarningMonitor redirected them.

    Looked up lazily via ``sys.modules`` so this module remains pure stdlib and
    never triggers simulator imports when monitoring is inactive.
    """
    monitor_mod = sys.modules.get("src.eval_client.physx_warning_monitor")
    if monitor_mod is None:
        return
    get_monitor = getattr(monitor_mod, "get_monitor", None)
    if not callable(get_monitor):
        return
    try:
        monitor = get_monitor()
        restore = getattr(monitor, "restore_stdio", None)
        if callable(restore):
            restore()
    except Exception:
        pass


def _hard_exit(exit_fn: Callable[[int], Any] | None = None) -> None:
    """Exit without running Python cleanup that may touch the broken runtime."""

    # Production ``os._exit`` never returns.  The optional callback keeps the
    # behavior easy to exercise in CPU-only tests without terminating pytest.
    (exit_fn or os._exit)(99)


def restart_or_exit(
    env: Any,
    fatal_msg: str,
    *,
    max_restarts: int = MAX_INPROC_RESTARTS,
    exec_fn: Callable[[str, Sequence[str]], Any] | None = None,
    exit_fn: Callable[[int], Any] | None = None,
    executable: str | None = None,
    argv: Sequence[str] | None = None,
) -> NoReturn:
    """Persist progress and replace the process, or request shell retry.

    ``exec_fn`` and ``exit_fn`` are injectable for CPU-only tests.  Production
    callers use :func:`os.execv` and :func:`os._exit`, preserving the original
    executable, arguments, and run id across an in-process replacement.
    """

    restart_count = _restart_count() + 1
    _persist_manifest(env, restart_count)
    print(
        f"[FATAL] PhysX kernel failure detected: {fatal_msg}; persisted manifest. "
        f"In-process restart attempt {restart_count}/{max_restarts}.",
        flush=True,
    )

    if restart_count > max_restarts:
        print(
            f"[FATAL] in-process restart cap reached ({max_restarts}); "
            "exiting with rc=99 for bash-level retry.",
            flush=True,
        )
        _flush_streams()
        _hard_exit(exit_fn)
        return

    os.environ[RESTART_COUNT_ENV] = str(restart_count)
    executable = executable or sys.executable
    argv = list(argv if argv is not None else sys.argv)
    command = [executable, *argv]
    _restore_monitored_stdio()
    print(f"[FATAL] os.execv self-restart with run_id={os.environ.get(RUN_ID_ENV)}", flush=True)
    _flush_streams()
    try:
        (exec_fn or os.execv)(executable, command)
    except Exception as exc:
        # An exec failure leaves the damaged runtime in place.  Do not fall
        # through to ordinary exception handling or atexit cleanup.
        print(f"[FATAL] os.execv failed: {exc}; exiting with rc=99.", flush=True)
        _flush_streams()
        _hard_exit(exit_fn)
        return
    # A test double may return even though a real execv never does.  Treat that
    # as a failed replacement and use the same hard shell-retry path.
    print("[FATAL] os.execv returned unexpectedly; exiting with rc=99.", flush=True)
    _hard_exit(exit_fn)


def exit_for_shell_restart(
    env: Any,
    fatal_msg: str,
    *,
    exit_fn: Callable[[int], Any] | None = None,
) -> NoReturn:
    """Persist progress and terminate so ``eval_policy.sh`` starts fresh Kit."""

    restart_count = _restart_count()
    _persist_manifest(env, restart_count)
    print(
        f"[FATAL] PhysX requested shell-level restart: {fatal_msg}; "
        "exiting with rc=99 for bash-level retry.",
        flush=True,
    )
    _flush_streams()
    _hard_exit(exit_fn)


# Keep the old private spellings available to callers that imported the
# helpers from an earlier revision.  They are aliases, not simulator imports.
_restart_or_exit = restart_or_exit
_exit_for_shell_restart = exit_for_shell_restart
