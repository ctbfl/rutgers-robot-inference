"""
How a policy server announces itself, and how the menu reaps the dead.

A policy server writes one JSON file into :data:`ruri.config.POLICIES_DIR` when
it becomes able to serve, and then forgets about it. It never deletes it, not
even on a clean exit. The menu sweeps the directory once a second, drops any
entry whose pid is gone, and unlinks the file.

That split is the point. There is exactly one writer role and exactly one
cleaner role, so a policy server carries no cleanup code at all -- no atexit,
no signal handlers -- and ``kill -9`` is not a special case, it is the only
case. The cost is that a dead server can linger in the menu for up to a
second, which is the same delay Ctrl-C gets.

The two components never talk to each other. A policy server registers whether
or not a menu is running, so a menu started later immediately sees every
server, including ones older than itself.

Registration is announced at the moment the server can actually answer, which
is after ``bind()`` -- not at process start. Loading a Pi0.5 checkpoint takes
about 22 seconds, and during those 22 seconds the process exists but cannot
serve anything; a client that found it in the menu would just time out.

Not handled: pid reuse. If a server is killed and the kernel hands its pid to
an unrelated process before the next sweep, that entry survives one extra
second. Recording the process start time from /proc would close it; it is not
worth the code today.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from typing import Any

from ruri.config import MENU_LOG, MENU_PORT, POLICIES_DIR, RUNTIME_DIR


logger = logging.getLogger(__name__)


def entry_path(name: str):
    return POLICIES_DIR / f"{name}.json"


def advertised_endpoint(bind_address: str, host: str | None = None) -> str:
    """
    Turn a bind address into one a client can dial.

    ``tcp://*:5555`` is meaningful only to the process that binds it, so the
    wildcard is replaced with a reachable hostname. Pass `host` explicitly when
    the machine's own idea of its name is not what the robot can resolve.
    """
    host = host or socket.gethostname()
    for wildcard in ("://*:", "://0.0.0.0:"):
        if wildcard in bind_address:
            return bind_address.replace(wildcard, f"://{host}:", 1)
    return bind_address


def register(name: str, endpoint: str, describe: dict[str, Any]) -> None:
    """
    Announce this server, then start the menu if nobody else has.

    Never raises: a policy server that cannot register is still a working
    policy server, and taking it down because a convenience process is missing
    would be the wrong trade.
    """
    try:
        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "endpoint": endpoint,
            "pid": os.getpid(),
            "describe": describe,
        }
        # Written whole, then renamed, so the menu never reads half a file.
        tmp = entry_path(name).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.rename(entry_path(name))
        logger.info("Registered %r at %s", name, endpoint)
    except Exception:
        logger.exception("Could not register %r; serving anyway", name)
        return

    try:
        ensure_menu_running()
    except Exception:
        logger.exception("Could not start the RURI menu; serving anyway")


def ensure_menu_running() -> bool:
    """
    Start the menu unless something already holds its port. Returns True if we
    spawned one.

    The port is the mutex. Several servers starting at once will all spawn a
    menu, but only one can bind; the losers exit immediately on their own, so
    no locking is needed here.
    """
    if _port_in_use(MENU_PORT):
        return False

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = open(MENU_LOG, "ab")
    subprocess.Popen(
        [sys.executable, "-m", "ruri.server.menu"],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        # Its own session, so that Ctrl-C in the terminal that happened to
        # start this policy server does not also kill the menu, and so that
        # the menu outlives whichever server spawned it.
        start_new_session=True,
    )
    logger.info("Started the RURI menu on port %d (log: %s)", MENU_PORT, MENU_LOG)
    return True


def _port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """A plain TCP connect: enough to tell whether anything holds the port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------
# Menu side
# ----------------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, but it exists.
        return True
    return True


def sweep() -> list[dict[str, Any]]:
    """
    Return the live entries, unlinking the dead ones on the way through.

    Unreadable files are treated as dead: a truncated or hand-edited entry
    cannot be dialled anyway, and leaving it would mean it is never reaped.
    """
    if not POLICIES_DIR.is_dir():
        return []

    live: list[dict[str, Any]] = []
    for path in sorted(POLICIES_DIR.glob("*.json")):
        try:
            entry = json.loads(path.read_text())
            pid = int(entry["pid"])
        except Exception:
            logger.warning("Discarding unreadable registry entry %s", path)
            path.unlink(missing_ok=True)
            continue

        if _alive(pid):
            live.append(entry)
        else:
            logger.info("Reaping %r (pid %d is gone)", entry.get("name", path.stem), pid)
            path.unlink(missing_ok=True)

    return live
