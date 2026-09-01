"""
The RURI menu: which policies are up right now.

A client asks one fixed port what is available, gets back a name, an endpoint
and the full metadata for each live policy server, and then connects straight
to the one it wants. The menu is not on the data path -- it never sees an
observation or an action chunk -- so it costs the inference loop nothing.

It is started automatically by whichever policy server finds its port free
(see :mod:`ruri.server.registry`), so nobody has to remember to run it.

State
-----
None of its own. Every second it re-reads
:data:`ruri.config.POLICIES_DIR`, drops the entries whose pid is gone, and
answers from what is left. It never writes a registration and never contacts a
policy server, so there is nothing to get out of sync: a menu started at any
time immediately sees every server, including ones older than itself, and a
menu that dies loses nothing that is not on disk.

The sweep runs on its own clock rather than only when a request arrives, so a
dead entry is reaped whether or not anyone is looking.

Wire format
-----------
Plain msgpack, deliberately not :mod:`ruri.common.zmq`. That module pulls in
numpy for its ndarray codec, which triples this process's footprint (15 MiB ->
37 MiB) to carry arrays a menu never sends. The encodings agree for
array-free payloads, so a client using the normal RURI codec talks to the menu
without knowing the difference.

    {"type": "list"}  ->  {"policies": [{"name", "endpoint", "describe"}, ...]}
    {"type": "stop"}  ->  {"stopping": true}, then exits
"""

from __future__ import annotations

import logging
import time

import msgpack
import zmq

from ruri.config import MENU_PORT, POLICIES_DIR, menu_bind_address
from ruri.server.registry import sweep


logger = logging.getLogger("ruri.menu")

SWEEP_INTERVAL_S = 1.0
POLL_TIMEOUT_MS = 200


def handle(request: dict, live: list[dict]) -> dict:
    kind = request.get("type", "list")
    if kind == "list":
        return {
            "policies": [
                {"name": e.get("name"), "endpoint": e.get("endpoint"), "describe": e.get("describe")}
                for e in live
            ]
        }
    if kind == "stop":
        return {"stopping": True}
    return {"error": f"unknown request type {kind!r}; expected 'list' or 'stop'"}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind(menu_bind_address())
    except zmq.ZMQError as exc:
        # Lost the race with another policy server that spawned a menu too.
        # Expected, not an error: the port is the mutex.
        logger.info("Menu port %d already held (%s); exiting", MENU_PORT, exc)
        socket.close(linger=0)
        context.term()
        return 0

    logger.info("RURI menu on port %d, watching %s", MENU_PORT, POLICIES_DIR)

    live = sweep()
    last_sweep = time.monotonic()
    try:
        while True:
            if socket.poll(POLL_TIMEOUT_MS):
                try:
                    request = msgpack.unpackb(socket.recv(), raw=False)
                    response = handle(request, live)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Request failed")
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                # A REP socket must answer everything it receives or it locks
                # into a "must send" state and the next recv fails outright.
                socket.send(msgpack.packb(response, use_bin_type=True))
                if response.get("stopping"):
                    break

            now = time.monotonic()
            if now - last_sweep >= SWEEP_INTERVAL_S:
                live = sweep()
                last_sweep = now
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        context.term()

    logger.info("RURI menu stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
