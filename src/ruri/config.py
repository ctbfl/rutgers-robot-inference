"""
Process-wide RURI settings.

Only the things more than one component has to agree on live here: where the
menu listens, and where policy servers announce themselves.

The runtime directory is deliberately under /tmp. It holds per-boot state --
which policy servers are up right now -- and /tmp is tmpfs on this machine, so
a reboot clears it without anyone having to remember to. It is suffixed with
the uid so that two users on the same box do not collide.
"""

from __future__ import annotations

import os
import pathlib


# Fixed so a client only ever has to know one address. Override for a second,
# isolated RURI on the same machine.
MENU_PORT = int(os.environ.get("RURI_MENU_PORT", "5550"))

RUNTIME_DIR = pathlib.Path(
    os.environ.get("RURI_RUNTIME_DIR", f"/tmp/ruri-{os.getuid()}")
)

# One JSON file per live policy server. Written by the server, read and
# reaped by the menu; the two never talk to each other.
POLICIES_DIR = RUNTIME_DIR / "policies"

# The menu is started in the background by whichever policy server finds the
# port free, so its output needs somewhere to go that is not a terminal.
MENU_LOG = RUNTIME_DIR / "menu.log"


def menu_bind_address() -> str:
    return f"tcp://*:{MENU_PORT}"


def menu_connect_address(host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{MENU_PORT}"
